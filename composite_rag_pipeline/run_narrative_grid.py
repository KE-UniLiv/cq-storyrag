#!/usr/bin/env python3
"""
Run N times per (persona × length) for selected patterns (default: KG, Hybrid),
bundle outputs into the evaluator's canonical layout, and run the evaluator
to compute averages across those runs.

CLI ONLY accepts:
  --personas  (space-separated)
  --lengths   (space-separated)
  --patterns  (space-separated; default: KG Hybrid)
  --runs      (int; default: 10)

Everything else is hardcoded to your working command.
"""

from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# =========================
# HARD-CODED CONSTANTS
# =========================
PROJECT_ROOT       = Path(".").resolve()
PIPELINE_PATH      = PROJECT_ROOT / "pipeline_programmatic.py"
PIPELINE_GRAPH     = PROJECT_ROOT / "pipeline_graph.py"
PIPELINE_ENRICHED  = PROJECT_ROOT / "pipeline_enriched.py"
EVAL_CLI           = PROJECT_ROOT / "eval" / "eval_cli.py"

# Inputs (adjust paths if your tree differs)
KG_META            = PROJECT_ROOT / "index" / "KG" / "cq_metadata.json"
HY_META            = PROJECT_ROOT / "index" / "Hybrid" / "cq_metadata.json"
NARRATIVE_PLANS    = PROJECT_ROOT / "data" / "narrative_plans.json"
RDF_PATH           = PROJECT_ROOT / "data" / "liveaid_instances_master.ttl"

# Defaults (can be overridden via CLI)
PATTERNS_DEFAULT   = ["KG", "Hybrid"]
RUNS_PER_GROUP     = 10  # default number of runs per persona×length×pattern

# Fixed generation/planning params
ITEMS_PER_BEAT     = 2
BASE_SEED          = 42
LLM_PROVIDER       = "gemini"
LLM_MODEL          = "gemini-2.5-flash"

USE_EXTERNAL_PLANNER   = True
PLANNER_PATH           = PROJECT_ROOT / "planner" / "planner_dual_random.py"
PLANNER_MATCH_STRATEGY = "intersect"
PERSIST_PARAMS         = True

# Retriever / Generator params (as provided)
RETRIEVER_JSON = json.dumps({
    "event": "ex:LiveAid1985",
    "musicgroup": "ex:Queen",
    "singleartist": "ex:Madonna",
    "bandmember": "ex:BrianMay",
    "venue": "ex:WembleyStadium",
    "venue2": "ex:JFKStadium",
})
GENERATOR_JSON = json.dumps({
    "Event": "Live Aid 1985",
    "MusicGroup": "Queen",
    "SingleArtist": "Madonna",
    "BandMember": "Brian May",
    "Venue": "Wembley Stadium",
    "Venue2": "JFK Stadium",
})

# Output roots
RUN_ROOT_BASE      = PROJECT_ROOT / "runs"
BUNDLE_ROOT_BASE   = PROJECT_ROOT / "eval" / "data"
EVAL_REPORTS_BASE  = PROJECT_ROOT / "eval" / "reports"

# =========================
# Helpers
# =========================
ANS_RE   = re.compile(r"answers_(?P<pat>[\w\-]+)\.jsonl$", re.I)
PLAN_RE  = re.compile(r"plan_with_evidence_(?P<pat>[\w\-]+)\.json$", re.I)
STORY_RE = re.compile(r"story_(?P<pat>[\w\-]+)(?P<clean>_clean)?\.md$", re.I)

def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

# Put near the other small helpers in pipeline_programmatic.py (or in your graph pipeline file)

def _graph_length_profile(length: str) -> dict:
    """
    Returns a profile controlling the size of the Graph story:
      - beats_limit: how many sections to keep from the plan
      - beat_sentences: sentences per section for the generator
    """
    L = (length or "Medium").strip().lower()
    profiles = {
      "Short":  {"max_hops": 1, "beam": 2, "beats": 4, "sentences": 3},
      "Medium": {"max_hops": 2, "beam": 3, "beats": 6, "sentences": 3},  # was 6 sentences → 3
      "Long":   {"max_hops": 3, "beam": 4, "beats": 8, "sentences": 4},
    }
    return profiles.get(L, profiles["medium"])


def _apply_length_profile_to_plan(plan_obj: dict, beats_limit: int) -> dict:
    """
    Trim whatever the graph planner produced to the first N sections.
    We try common keys to be robust: 'beats' > 'sections' > 'outline'.
    """
    plan = copy.deepcopy(plan_obj) if isinstance(plan_obj, dict) else {}
    for key in ("beats", "sections", "outline"):
        if isinstance(plan.get(key), list):
            plan[key] = plan[key][:max(1, int(beats_limit))]
            break
    return plan


def simple_clean_story(text: str) -> str:
    """Create a clean narrative: drop headings/bullets/code fences/URLs/IDs and rewrap."""
    lines = []
    for raw in (text or "").splitlines():
        ln = raw.rstrip()
        if not ln:
            lines.append("")
            continue
        if ln.lstrip().startswith(("#","- ","* ","+ ","```","~~~")):
            continue
        ln = re.sub(r"<https?://[^>]+>", "", ln)
        ln = re.sub(r"https?://\S+", "", ln)
        ln = re.sub(r"\b[a-z]{3,}:[A-Za-z0-9/_#\.\-]+", "", ln)
        ln = re.sub(r"\s{2,}", " ", ln).strip()
        if ln:
            lines.append(ln)
    out, para = [], []
    for ln in lines:
        if ln == "":
            if para:
                out.append(" ".join(para)); para=[]
        else:
            para.append(ln)
    if para: out.append(" ".join(para))
    return "\n\n".join(out).strip() + ("\n" if out else "")

def _norm_pat(p: str) -> str:
    m = {"kg": "KG", "hybrid": "Hybrid", "graph": "Graph", "enriched": "Enriched"}
    return m.get(p.lower().strip(), p)

def build_cmd(pattern: str, persona: str, length: str, seed: int, run_root: Path) -> List[str]:
    """
    Always pass BOTH --kg_meta and --hy_meta because pipeline_programmatic.py requires them,
    regardless of which pattern you're running.
    """
    if pattern.lower() == "graph":
        args: List[str] = [sys.executable, str(PIPELINE_GRAPH)]
    elif pattern.lower() == "enriched":
        args: List[str] = [sys.executable, str(PIPELINE_ENRICHED)]
    else:
        args: List[str] = [sys.executable, str(PIPELINE_PATH)]
        args += ["--pattern", pattern]

    # REQUIRED by the pipeline (for all patterns):
    args += ["--kg_meta", str(KG_META)]
    args += ["--hy_meta", str(HY_META)]

    # Shared required inputs
    args += [
        "--narrative_plans", str(NARRATIVE_PLANS),
        "--rdf", str(RDF_PATH),
        "--persona", persona,
        "--length", length,
        "--items_per_beat", str(ITEMS_PER_BEAT),
        "--seed", str(seed),
        "--retriever_params_json", RETRIEVER_JSON,
        "--generator_params_json", GENERATOR_JSON,
        "--llm_provider", LLM_PROVIDER,
        "--llm_model", LLM_MODEL,
        "--run_root", str(run_root),
    ]

    # Planner flags (hardcoded ON per your setup)
    if USE_EXTERNAL_PLANNER:
        args += [
            "--use_external_planner",
            "--planner_path", str(PLANNER_PATH),
            "--planner_match_strategy", PLANNER_MATCH_STRATEGY
        ]

    if PERSIST_PARAMS:
        args += ["--persist_params"]

    # NOTE: we don't need a pattern flag because your pipeline infers pattern from metas
    # If your pipeline *does* need an explicit pattern, add e.g.:
    # args += ["--pattern", pattern]

    return args


def run_once(cmd: List[str], cwd: Path):
    print("[RUN]", " ".join(cmd))
    cp = subprocess.run(cmd, cwd=str(cwd))
    if cp.returncode != 0:
        raise RuntimeError(f"pipeline_programmatic failed with code {cp.returncode}")

def _latest(paths: List[Path]) -> Optional[Path]:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None

def discover_outputs(dirpath: Path, pattern: str) -> Dict[str, Path]:
    """
    Recursively discover artifacts anywhere under dirpath.
    Picks the most-recent (by mtime) file for each role.
    We match by canonical names that include the pattern.
    """
    pat = pattern  # e.g., "KG", "Hybrid", "Graph"

    # answers & plan – exact filenames per pattern
    ans_cands  = list(dirpath.rglob(f"answers_{pat}.jsonl"))
    plan_cands = list(dirpath.rglob(f"plan_with_evidence_{pat}.json"))

    # story clean (preferred) and raw
    clean_cands = list(dirpath.rglob(f"story_{pat}_clean.md"))
    story_cands = list(dirpath.rglob(f"story_{pat}.md"))

    answers = _latest(ans_cands)
    plan    = _latest(plan_cands)
    story_clean = _latest(clean_cands)
    story       = _latest(story_cands)

    files: Dict[str, Path] = {}
    if answers: files["answers"] = answers
    if plan:    files["plan"]    = plan
    if story:   files["story"]   = story
    if story_clean: files["story_clean"] = story_clean

    # Debug prints help when something is missing
    if not answers or not plan:
        print("[WARN] Missing required artifacts for", pat)
        print("       answers:", answers, "plan:", plan)
        print("       search root:", dirpath)

    return files


def ensure_story_clean(files: Dict[str, Path], pattern: str, target_dir: Path) -> Optional[Path]:
    clean = files.get("story_clean")
    if clean and clean.exists():
        return clean
    raw = files.get("story")
    if not raw or not raw.exists():
        return None
    cleaned = simple_clean_story(read_text(raw))
    out = target_dir / f"story_{pattern}_clean.md"
    write_text(out, cleaned)
    print(f"[CLEAN] wrote {out}")
    files["story_clean"] = out
    return out

def copy_into_bundle(src_files: Dict[str, Path], dst_dir: Path, pattern: str) -> Dict[str, str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    final: Dict[str, str] = {}
    if "answers" in src_files:
        dst = dst_dir / f"answers_{pattern}.jsonl"
        shutil.copy2(src_files["answers"], dst)
        final["answers"] = str(dst)
    if "plan" in src_files:
        dst = dst_dir / f"plan_with_evidence_{pattern}.json"
        shutil.copy2(src_files["plan"], dst)
        final["plan"] = str(dst)
    if "story" in src_files:
        dst = dst_dir / f"story_{pattern}.md"
        shutil.copy2(src_files["story"], dst)
        final["story"] = str(dst)
    if "story_clean" in src_files:
        dst = dst_dir / f"story_{pattern}_clean.md"
        if src_files["story_clean"].resolve() != dst.resolve():
            shutil.copy2(src_files["story_clean"], dst)
        final["story_clean"] = str(dst)
    return final

def run_eval(root: Path, report_dir: Path):
    cmd = [sys.executable, str(EVAL_CLI), "--root", str(root), "--report-dir", str(report_dir)]
    print("[EVAL]", " ".join(cmd))
    cp = subprocess.run(cmd, cwd=str(EVAL_CLI.parent))
    if cp.returncode != 0:
        raise RuntimeError(f"evaluate_rag failed with code {cp.returncode}")

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Run KG/Hybrid (or Graph) N× per persona×length; bundle + eval.")
    ap.add_argument("--personas", nargs="+", required=True, help="Persona names (space-separated)")
    ap.add_argument("--lengths", nargs="+", required=True, help="Story lengths (e.g., Short Medium Long)")
    ap.add_argument(
        "--patterns",
        nargs="+",
        default=PATTERNS_DEFAULT,
        help="Patterns to run (e.g., KG Hybrid [Graph]). Default: KG Hybrid",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=RUNS_PER_GROUP,
        help=f"Number of runs per persona×length×pattern (default: {RUNS_PER_GROUP})",
    )
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    patterns = [_norm_pat(p) for p in args.patterns]

    print(f"patterns : {patterns}")

    for persona in args.personas:
        for length in args.lengths:
            group = f"{persona}-{length}-{ts}"
            # bundles go here; evaluator will average across run-xx under this root
            group_bundle_root = mkdir(BUNDLE_ROOT_BASE / group)
            report_dir = mkdir(EVAL_REPORTS_BASE / group)

            for run_idx in range(1, args.runs + 1):
                per_run_seed = BASE_SEED + run_idx
                run_tag = f"run-{run_idx:02d}"

                for pattern in patterns:


                    raw_out = mkdir(RUN_ROOT_BASE / group / pattern / run_tag)
                    print(f"raw_out : {raw_out}")
                    #/ Users / sowjanyab / code / dissertation / comp70225 - wembrewind / composite_rag_pipeline / runs / Emma - Medium - 20250820 - 181659 / Graph / run - 01
                    cmd = build_cmd(pattern, persona, length, per_run_seed, raw_out)
                    run_once(cmd, cwd=PROJECT_ROOT)

                    discovered = discover_outputs(raw_out, pattern)
                    ensure_story_clean(discovered, pattern, raw_out)

                    dst = group_bundle_root / run_tag / pattern
                    print(f"dst : {dst}")
                    # /Users/sowjanyab/code/dissertation/comp70225-wembrewind/composite_rag_pipeline/eval/data/Emma-Medium-20250820-181659/run-01/Graph
                    final = copy_into_bundle(discovered, dst, pattern)

                    # small manifest for traceability
                    (dst / "manifest.json").write_text(
                        json.dumps({
                            "pattern": pattern,
                            "persona": persona,
                            "length": length,
                            "run": run_idx,
                            "seed": per_run_seed,
                            "files": final,
                        }, indent=2),
                        encoding="utf-8",
                    )

            # evaluate the whole persona×length group (averages across runs)
            run_eval(root=group_bundle_root, report_dir=report_dir)
            print(f"[AVG] persona={persona} length={length} → {report_dir}")

    print("\n[OK] All groups done.")
    print(f"Bundles: {BUNDLE_ROOT_BASE}")
    print(f"Reports: {EVAL_REPORTS_BASE}")

if __name__ == "__main__":
    main()
