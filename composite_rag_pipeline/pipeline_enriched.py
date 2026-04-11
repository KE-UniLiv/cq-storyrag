#!/usr/bin/env python3
"""
Enriched RAG pipeline: CQ backbone + per-beat bounded graph expansion.

Sits alongside pipeline_programmatic.py (KG / Hybrid) and pipeline_graph.py
as a fully comparable 4th variant.  Uses the same planner and SPARQL
retriever as KG mode, then enriches each beat's evidence with k-hop graph
expansion before generation.

Output artifacts — same naming convention as other variants:
  retriever/plan_with_evidence_Enriched.json
  generator/answers_Enriched.jsonl
  generator/story_Enriched.md
  generator/story_Enriched_clean.md
  params/graph_expansion_config.json

Graph expansion config knobs (all exposed as CLI flags):
  --max_hops    k-hop depth from seed entities   (default: 1)
  --max_nodes   max subgraph nodes per beat      (default: 50)
  --top_k       max triples kept after ranking   (default: 20)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---- Shared planner / util infrastructure --------------------------------
# Import from pipeline_programmatic rather than copying to avoid drift.
from pipeline_programmatic import (
    DEFAULT_GENERATOR_CFG,
    DEFAULT_GENERATOR_PARAMS,
    DEFAULT_RETRIEVER_CFG,
    DEFAULT_RETRIEVER_PARAMS,
    _make_plans_via_external_or_internal,
    _make_run_dirs,
    deep_update,
    read_json,
    slug,
    write_json,
    write_jsonl,
)

# ---- Retrieval -----------------------------------------------------------
from retriever.retriever_local_rdflib import run as retriever_run
from retriever.evidence_pack_builder import build_evidence_packs

# ---- Generation ----------------------------------------------------------
from generator.generator_enriched import generate as generator_generate

# ---- KG loading ----------------------------------------------------------
from rdflib import Graph


# ==========================================================================
# Graph expansion defaults
# ==========================================================================

DEFAULT_GRAPH_EXPANSION_CFG: Dict[str, Any] = {
    "max_hops":  1,   # 1-hop keeps expansion local and fast
    "max_nodes": 50,  # hard cap per beat — keeps output explainable
    "top_k":     20,  # triples kept after degree-ranking
}


# ==========================================================================
# Pipeline runner
# ==========================================================================

def run_enriched(
    *,
    kg_meta: Path,
    hy_meta: Path,
    narrative_plans: Path,
    rdf_files: List[Path],
    persona: str,
    length: str,
    items_per_beat: int,
    seed: int,
    generator_params: Dict[str, Any],
    retriever_params: Dict[str, Any],
    out_root: Path,
    retriever_cfg: Dict[str, Any],
    generator_cfg: Dict[str, Any],
    graph_expansion_cfg: Dict[str, Any],
    use_external_planner: bool,
    planner_path: Path,
    planner_match_strategy: str,
    run_root: Path = Path("runs"),
    run_tag: Optional[str] = None,
    persist_params: bool = False,
) -> None:
    """
    Execute the Enriched pipeline for one (persona, length, seed) triple.

    Steps
    -----
    1. Plan   – shared CQ planner (same as KG / Hybrid)
    2. Retrieve – SPARQL-only, no URL enrichment
    3. Expand   – per-beat bounded k-hop graph expansion (new)
    4. Generate – two-tier prompt: primary CQ evidence + supporting graph context
    5. Save     – all artifacts under run_root with "Enriched" suffix
    """
    run_dirs = _make_run_dirs(run_root, persona, length, seed, run_tag)
    planner_dir   = run_dirs["planner"]
    retriever_dir = run_dirs["retriever"]
    generator_dir = run_dirs["generator"]
    params_dir    = run_dirs["params"]

    # ------------------------------------------------------------------
    # 1) Plan — reuse the same CQ planner as KG / Hybrid
    #    The planner always produces both KG and Hybrid plans; we use KG.
    # ------------------------------------------------------------------
    plan_kg_path = planner_dir / "plan_KG.json"
    plan_hy_path = planner_dir / "plan_Hybrid.json"

    plan_kg, _ = _make_plans_via_external_or_internal(
        use_external=use_external_planner,
        planner_path=planner_path,
        match_strategy=planner_match_strategy,
        kg_meta=kg_meta,
        hy_meta=hy_meta,
        narrative_plans=narrative_plans,
        persona=persona,
        length=length,
        items_per_beat=items_per_beat,
        seed=seed,
        out_kg_path=plan_kg_path,
        out_hy_path=plan_hy_path,
    )
    print(f"✓ planner → {plan_kg_path}")

    if persist_params:
        try:
            write_json(params_dir / "retriever_params_used.json",     retriever_params)
            write_json(params_dir / "generator_params_used.json",     generator_params)
            write_json(params_dir / "graph_expansion_config.json",    graph_expansion_cfg)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 2) SPARQL retrieval — identical to KG mode (no URL enrichment)
    #    Graph expansion handles the extra context instead.
    # ------------------------------------------------------------------
    shared = retriever_cfg.get("shared", {})
    kgc    = retriever_cfg.get("kg", {})

    retr_out_path = retriever_dir / "plan_with_evidence_Enriched.json"

    out_sparql = retriever_run(
        plan=plan_kg,
        rdf_files=[str(p) for p in rdf_files],
        bindings=retriever_params,
        per_item_sample=int(shared.get("per_item_sample", 5)),
        require_sparql=bool(shared.get("require_sparql", True)),
        timeout_s=float(shared.get("timeout_s", 10.0)),
        log_dir=str(retriever_dir / "logs"),
        errors_jsonl=str(retriever_dir / "retriever_enriched.jsonl"),
        include_stack=True,
        include_executed_query=True,
        strict_bindings=False,
        execute_on_unbound=False,
        # No URL enrichment — graph expansion provides the extra context
        enrich_urls=False,
        fetch_url_content=False,
        url_timeout_s=float(shared.get("url_timeout_s", 5.0)),
        max_urls_per_item=int(shared.get("max_urls_per_item", 5)),
        content_max_bytes=int(shared.get("content_max_bytes", 250_000)),
        content_max_chars=int(shared.get("content_max_chars", 5000)),
        chunk_url_content=False,
        chunk_chars=int(shared.get("chunk_chars", 500)),
        chunk_overlap=int(shared.get("chunk_overlap", 50)),
        max_chunks_per_url=int(shared.get("max_chunks_per_url", 8)),
        max_url_chunks_total_per_item=int(shared.get("max_url_chunks_total_per_item", 20)),
    )
    print(f"✓ retriever (SPARQL) → {len(out_sparql.get('items', []))} items")

    # ------------------------------------------------------------------
    # 3) Per-beat graph expansion
    #    Load the KG once and augment each item with expanded_rows +
    #    expansion_trace.  The KG is already loaded by the retriever
    #    but not passed back, so we reload it here (typically < 1 s).
    # ------------------------------------------------------------------
    print("[Enriched] Loading KG for graph expansion...")
    kg = Graph()
    for f in rdf_files:
        kg.parse(str(f))
    print(f"[Enriched] KG loaded: {len(kg)} triples")

    enriched_evidence = build_evidence_packs(
        out_sparql,
        kg,
        max_hops=int(graph_expansion_cfg.get("max_hops",  1)),
        max_nodes=int(graph_expansion_cfg.get("max_nodes", 50)),
        top_k=int(graph_expansion_cfg.get("top_k",     20)),
    )

    write_json(retr_out_path, enriched_evidence)
    print(f"✓ evidence packs (enriched) → {retr_out_path}")

    # Summary of expansion coverage
    items = enriched_evidence.get("items", [])
    seeded   = sum(1 for it in items if it.get("expansion_trace", {}).get("seeds"))
    expanded = sum(len(it.get("expanded_rows", [])) for it in items)
    print(f"[Enriched] Seed coverage: {seeded}/{len(items)} items had seeds; "
          f"{expanded} expansion triples total")

    # ------------------------------------------------------------------
    # 4) Generation — two-tier prompt (primary CQ + supporting graph)
    # ------------------------------------------------------------------
    gc = generator_cfg
    story_path       = generator_dir / "story_Enriched.md"
    story_clean_path = generator_dir / "story_Enriched_clean.md"
    answers_path     = generator_dir / "answers_Enriched.jsonl"
    claims_path      = (
        generator_dir / "claims_Enriched.jsonl"
        if gc.get("make_claims", True) else None
    )

    story_md, answers = generator_generate(
        mode="Enriched",
        plan=plan_kg,
        plan_with_evidence=enriched_evidence,
        params=generator_params,
        llm_provider=gc.get("llm_provider", "ollama"),
        llm_model=gc.get("llm_model", "llama3.1-128k"),
        ollama_num_ctx=gc.get("ollama_num_ctx"),
        max_facts_per_beat=int(gc.get("max_facts_per_beat", 12)),
        max_expanded_facts=int(graph_expansion_cfg.get("top_k", 20)),
        context_budget_chars=int(gc.get("context_budget_chars", 50000)),
        enforce_citation_each_sentence=False,
        citation_style=gc.get("citation_style", "cqid"),
        claims_out=str(claims_path) if claims_path else None,
        story_clean_out=str(story_clean_path),
        run_id=slug(persona) + "-" + slug(length) + "-" + str(seed) + "-enriched",
    )

    story_path.write_text(story_md, encoding="utf-8")
    write_jsonl(answers_path, answers)
    print(f"✓ generator Enriched → {story_path} (+ clean {story_clean_path})")
    if claims_path:
        print(f"✓ claims Enriched → {claims_path}")


# ==========================================================================
# CLI
# ==========================================================================

def cli() -> None:
    Boolean = argparse.BooleanOptionalAction
    ap = argparse.ArgumentParser(
        description="Enriched RAG pipeline: CQ backbone + per-beat graph expansion"
    )

    # Required inputs — identical to pipeline_programmatic.py
    ap.add_argument("--kg_meta",         type=Path, required=True)
    ap.add_argument("--hy_meta",         type=Path, required=True)
    ap.add_argument("--narrative_plans", type=Path, required=True)
    ap.add_argument("--rdf",             type=Path, nargs="+", required=True)

    # Run identity
    ap.add_argument("--persona",        default="Emma")
    ap.add_argument("--length",         default="Medium")
    ap.add_argument("--items_per_beat", type=int, default=2)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--out_root",       type=Path, default=Path("."))
    ap.add_argument("--run_root",       default="runs",
                    help="Folder where timestamped run directories are created.")
    ap.add_argument("--run_tag",        default=None,
                    help="Optional label appended to the run folder name.")
    ap.add_argument("--persist_params", action="store_true",
                    help="Write resolved params into run/params/.")

    # Params (as JSON strings, same as grid runner passes)
    ap.add_argument("--generator_params_json", default=None)
    ap.add_argument("--retriever_params_json", default=None)

    # LLM
    ap.add_argument("--llm_provider",   default=None, choices=["ollama", "gemini"])
    ap.add_argument("--llm_model",      default=None)
    ap.add_argument("--ollama_num_ctx", type=int, default=None)

    # Generator overrides
    ap.add_argument("--max_facts_per_beat",   type=int, default=None)
    ap.add_argument("--context_budget_chars", type=int, default=None)
    ap.add_argument("--citation_style",       choices=["numeric", "cqid"], default=None)

    # Graph expansion knobs
    ap.add_argument("--max_hops",  type=int, default=1,
                    help="K-hop depth for graph expansion (default: 1)")
    ap.add_argument("--max_nodes", type=int, default=50,
                    help="Max nodes in expanded subgraph per beat (default: 50)")
    ap.add_argument("--top_k",     type=int, default=20,
                    help="Max triples kept after degree-ranking (default: 20)")

    # Planner
    ap.add_argument("--use_external_planner",    action=Boolean, default=True)
    ap.add_argument("--planner_path",            type=Path,
                    default=Path("planner/planner_dual_random.py"))
    ap.add_argument("--planner_match_strategy",  default="intersect")

    args = ap.parse_args()

    # ---- Merge configs ------------------------------------------------
    rcfg = copy.deepcopy(DEFAULT_RETRIEVER_CFG)
    gcfg = copy.deepcopy(DEFAULT_GENERATOR_CFG)
    graph_expansion_cfg = copy.deepcopy(DEFAULT_GRAPH_EXPANSION_CFG)

    retriever_params = copy.deepcopy(DEFAULT_RETRIEVER_PARAMS)
    generator_params = copy.deepcopy(DEFAULT_GENERATOR_PARAMS)

    if args.retriever_params_json:
        retriever_params = deep_update(retriever_params, json.loads(args.retriever_params_json))
    if args.generator_params_json:
        generator_params = deep_update(generator_params, json.loads(args.generator_params_json))

    # Apply CLI overrides to generator config
    if args.llm_provider:           gcfg["llm_provider"]        = args.llm_provider
    if args.llm_model:              gcfg["llm_model"]            = args.llm_model
    if args.ollama_num_ctx:         gcfg["ollama_num_ctx"]       = args.ollama_num_ctx
    if args.max_facts_per_beat:     gcfg["max_facts_per_beat"]   = args.max_facts_per_beat
    if args.context_budget_chars:   gcfg["context_budget_chars"] = args.context_budget_chars
    if args.citation_style:         gcfg["citation_style"]       = args.citation_style

    # Graph expansion — always taken from CLI
    graph_expansion_cfg["max_hops"]  = args.max_hops
    graph_expansion_cfg["max_nodes"] = args.max_nodes
    graph_expansion_cfg["top_k"]     = args.top_k

    run_enriched(
        kg_meta=args.kg_meta,
        hy_meta=args.hy_meta,
        narrative_plans=args.narrative_plans,
        rdf_files=args.rdf,
        persona=args.persona,
        length=args.length,
        items_per_beat=args.items_per_beat,
        seed=args.seed,
        generator_params=generator_params,
        retriever_params=retriever_params,
        out_root=args.out_root,
        retriever_cfg=rcfg,
        generator_cfg=gcfg,
        graph_expansion_cfg=graph_expansion_cfg,
        use_external_planner=args.use_external_planner,
        planner_path=args.planner_path,
        planner_match_strategy=args.planner_match_strategy,
        run_root=Path(args.run_root),
        run_tag=args.run_tag,
        persist_params=args.persist_params,
    )


if __name__ == "__main__":
    cli()
