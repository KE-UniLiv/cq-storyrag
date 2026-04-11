#!/usr/bin/env python3
"""
Beat-level story generator for the Enriched RAG pipeline
(CQ backbone + per-beat graph expansion).

Reuses LLM calls and text utilities from generator_dual.
Adds two-tier prompt construction:
  PRIMARY CONTEXT   – direct CQ SPARQL evidence (must anchor every beat)
  SUPPORTING CONTEXT – graph-expanded evidence (enriches but does not replace)

Output interface is identical to generator_dual.generate():
  returns (story_markdown, answers_stream)

answers_stream records include expansion_traces so downstream evaluation
can inspect what the graph expansion contributed per beat.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- Reuse shared utilities from generator_dual -------------------------
from generator.generator_dual import (
    _append_jsonl,
    _clean_story_text_remove_sections_and_citations,
    _ensure_sentence_citations_cqid,
    _score_row_for_fact_density,
    _split_sentences,
    _strip_meta_leadins,
    _trim_to_budget,
    load_persona_pack,
    llm_generate,
    log_prompt,
    make_prompt_record,
)

try:
    from generator.generator_dual import soften_readability
except ImportError:
    def soften_readability(text: str, **kwargs) -> str:  # type: ignore[misc]
        return text


# =============================================================================
# Row → readable text helpers
# =============================================================================

def _clean_n3_value(v: str) -> str:
    """Strip N3 encoding from a SPARQL result value."""
    v = v.strip()
    if v.startswith("<") and v.endswith(">"):
        return v[1:-1]
    if v.startswith('"') and "^^" in v:
        return v.split("^^")[0].strip('"')
    if len(v) >= 2 and v[0] == v[-1] == '"':
        return v[1:-1]
    return v


def _row_to_context_line(row: Dict[str, Any]) -> Optional[str]:
    """Convert a direct SPARQL result row to a clean context line."""
    parts = []
    for k, v in row.items():
        if isinstance(k, str) and k.startswith("__"):
            continue
        if isinstance(v, str) and v.strip():
            cleaned = _clean_n3_value(v)
            if cleaned:
                parts.append(cleaned)
    if not parts:
        return None
    return " — ".join(parts)


def _uri_to_label(uri: str) -> str:
    """Convert a URI to a human-readable label."""
    uri = str(uri).strip()
    if uri.startswith("<") and uri.endswith(">"):
        uri = uri[1:-1]
    if "#" in uri:
        label = uri.split("#")[-1]
    elif "/" in uri:
        label = uri.rstrip("/").split("/")[-1]
    else:
        label = uri
    return label.replace("_", " ").strip()


def _expanded_row_to_line(row: Dict[str, Any]) -> Optional[str]:
    """Format a graph-expansion triple as a readable context line."""
    s = _uri_to_label(str(row.get("subject", "")))
    p = _uri_to_label(str(row.get("predicate", "")))
    o = _uri_to_label(str(row.get("object", "")))
    if not s or not p or not o:
        return None
    return f"{s} {p} {o}"


# =============================================================================
# Prompt builder
# =============================================================================

def build_enriched_prompt(
    persona_description: str,
    beat_idx: int,
    beat_title: str,
    direct_lines: List[str],
    expanded_lines: List[str],
) -> Tuple[str, List[str]]:
    """
    Build a two-tier prompt for enriched generation.

    PRIMARY CONTEXT   – direct CQ SPARQL evidence; must anchor the beat
    SUPPORTING CONTEXT – graph-expanded evidence; may enrich but not replace

    Returns (prompt_text, combined_context_lines).
    """
    instruction = [
        "You are writing a factual, engaging story section. Do NOT roleplay as the audience.",
        f"Section context — Beat {beat_idx + 1}: {beat_title}",
        "",
        "Audience (write for them; do NOT roleplay):",
        f"- Description: {persona_description}",
        "- Tone/style: clear, precise, evidence-driven",
        "- Dos: lead with outcomes; use named entities, dates, and numbers; keep sentences tight",
        "- Don'ts: no first person, no speculation, no meta lead-ins; start directly.",
        "",
        "Use ONLY the CONTEXT below. Do not add any fact not present in CONTEXT.",
        "PRIMARY facts must anchor the narrative; each PRIMARY item should appear as a clause.",
        "SUPPORTING facts may add contextual texture — use them only where they directly "
        "relate to an already-established PRIMARY claim.",
        "",
        "Rules:",
        "- One paragraph; third person; no bullets or headings.",
        "- Begin with the event name to anchor the narrative.",
        "- Realize every PRIMARY fact as an explicit clause using exact predicate wording.",
        "- Do not invent facts not present in CONTEXT.",
        "- If a PRIMARY detail is missing, acknowledge the gap briefly rather than inventing it.",
    ]

    header = "\n".join(instruction)

    primary_block = (
        "PRIMARY CONTEXT (CQ direct evidence — answer this beat):\n"
        + ("\n".join(direct_lines) if direct_lines else "(none)")
    )

    supporting_block = (
        "SUPPORTING CONTEXT (graph expansion — use to enrich, not replace):\n"
        + ("\n".join(expanded_lines) if expanded_lines else "(none)")
    )

    prompt = textwrap.dedent(f"""
        {header}

        {primary_block}

        {supporting_block}

        Now write the story section in third person as one cohesive paragraph.
    """).strip()

    context_lines = direct_lines + expanded_lines
    return prompt, context_lines


# =============================================================================
# Main generate function
# =============================================================================

def generate(
    *,
    mode: str = "Enriched",
    plan: Dict[str, Any],
    plan_with_evidence: Dict[str, Any],
    params: Dict[str, Any],
    llm_provider: str,
    llm_model: str,
    ollama_num_ctx: Optional[int] = None,
    max_facts_per_beat: int = 8,
    max_expanded_facts: int = 6,
    context_budget_chars: int = 50000,
    enforce_citation_each_sentence: bool = False,
    citation_style: str = "cqid",
    claims_out: Optional[str] = None,
    story_clean_out: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate a story using CQ direct evidence + graph-expanded supporting evidence.

    Items in plan_with_evidence are expected to have (from evidence_pack_builder):
      rows            – direct SPARQL evidence rows
      expanded_rows   – graph-expansion triples as row-style dicts
      expansion_trace – traceability metadata (seeds, counts, summary)

    Returns (story_markdown, answers_stream).
    answers_stream entries include expansion_traces so downstream evaluation
    can audit what graph expansion contributed to each beat.
    """
    persona = plan.get("persona") or "Narrator"
    beats   = plan.get("beats") or []
    items   = plan_with_evidence.get("items") or []

    # Group items by beat index
    by_beat: Dict[int, List[Dict[str, Any]]] = {}
    for it in items:
        b   = it.get("beat") or {}
        idx = int(b.get("index", 0))
        by_beat.setdefault(idx, []).append(it)

    answers_stream: List[Dict[str, Any]] = []
    story_lines:    List[str] = []
    claims_path = Path(claims_out) if claims_out else None

    for b in beats:
        beat_idx   = int(b.get("index", 0))
        beat_title = b.get("title") or "Untitled"
        beat_items = by_beat.get(beat_idx, [])

        # ---- Collect evidence per CQ ----
        direct_rows_by_cq: Dict[str, List[Dict[str, Any]]] = {}
        all_expanded_rows: List[Dict[str, Any]] = []
        expansion_traces:  List[Dict[str, Any]] = []

        for it in beat_items:
            cqid = it.get("id") or "CQ-UNK"
            for r in (it.get("rows") or []):
                r2 = dict(r)
                r2["__cq_id"] = cqid
                direct_rows_by_cq.setdefault(cqid, []).append(r2)
            all_expanded_rows.extend(it.get("expanded_rows") or [])
            if it.get("expansion_trace"):
                expansion_traces.append({"cq_id": cqid, **it["expansion_trace"]})

        # ---- Build direct context lines (fact-density ranked) ----
        all_direct = [r for rows in direct_rows_by_cq.values() for r in rows]
        all_direct.sort(key=_score_row_for_fact_density, reverse=True)

        direct_lines: List[str] = []
        seen_direct: set = set()
        for row in all_direct[:max_facts_per_beat]:
            line = _row_to_context_line(row)
            if line and line not in seen_direct:
                direct_lines.append(line)
                seen_direct.add(line)

        # ---- Build expanded context lines ----
        expanded_lines: List[str] = []
        seen_expanded: set = set()
        for row in all_expanded_rows[:max_expanded_facts]:
            line = _expanded_row_to_line(row)
            if line and line not in seen_expanded:
                expanded_lines.append(line)
                seen_expanded.add(line)

        # ---- Load persona ----
        try:
            pack        = load_persona_pack(persona, path="config/personas.yaml")
            persona_desc = pack["description"]
        except Exception:
            persona_desc = "General reader; prefers clarity over flourish."

        # ---- Build prompt ----
        prompt, context_lines = build_enriched_prompt(
            persona_description=persona_desc,
            beat_idx=beat_idx,
            beat_title=beat_title,
            direct_lines=direct_lines,
            expanded_lines=expanded_lines,
        )

        print(
            f"[Enriched] Beat {beat_idx + 1} ({beat_title}): "
            f"{len(direct_lines)} direct, {len(expanded_lines)} expanded facts"
        )

        # ---- Log prompt ----
        meta = {
            "persona":       persona,
            "beat_index":    beat_idx,
            "beat_title":    beat_title,
            "citation_style": citation_style,
        }
        rec = make_prompt_record(
            prompt_text=prompt,
            meta=meta,
            model=llm_model,
            temperature=0.5,
            top_p=1.0,
            run_id=run_id,
        )
        log_prompt(rec, "outputs/prompts_log.jsonl")

        # ---- Generate ----
        text = llm_generate(llm_provider, llm_model, prompt, ollama_num_ctx=ollama_num_ctx)

        # Optional per-sentence citation enforcement
        if enforce_citation_each_sentence and citation_style == "cqid":
            fallback_cqid = next(iter(direct_rows_by_cq), None)
            text = _ensure_sentence_citations_cqid(text, fallback_cqid)

        text = _strip_meta_leadins(text)
        text = soften_readability(text, target_words_per_sent=20)

        # ---- Accumulate story ----
        story_lines.append(f"## {beat_title}\n\n{text}\n")

        # ---- Answers record ----
        # Includes expansion_traces for full traceability
        answers_stream.append({
            "beat_index":  beat_idx,
            "beat_title":  beat_title,
            "direct_evidence_by_cq": {
                cqid: [_row_to_context_line(r) for r in rows if _row_to_context_line(r)]
                for cqid, rows in direct_rows_by_cq.items()
            },
            "expanded_evidence": [
                _expanded_row_to_line(r) for r in all_expanded_rows[:max_expanded_facts]
                if _expanded_row_to_line(r)
            ],
            "expansion_traces": expansion_traces,
            "context_lines":    context_lines,
            "text":             text,
        })

        # ---- Claims ----
        if claims_path:
            for sent in _split_sentences(text):
                _append_jsonl(claims_path, {
                    "mode":        mode,
                    "beat_index":  beat_idx,
                    "beat_title":  beat_title,
                    "sentence":    sent,
                    "direct_cq_ids": list(direct_rows_by_cq.keys()),
                })

    story_md = "\n".join(story_lines)

    # Write clean story (no headings, no citations)
    if story_clean_out:
        clean = _clean_story_text_remove_sections_and_citations(story_md)
        Path(story_clean_out).parent.mkdir(parents=True, exist_ok=True)
        Path(story_clean_out).write_text(clean, encoding="utf-8")

    return story_md, answers_stream
