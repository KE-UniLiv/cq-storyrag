# CQ Story RAG
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)

## ESWC 2026

This work has been accepted at the **23rd International Conference on the Semantic Web (ESWC 2026)**, Dubrovnik, Croatia.

Paper:  
*Competency Questions as Executable Plans: A Controlled RAG Architecture for Cultural Heritage Storytelling*

```markdown
## Method Overview

The framework repurposes Competency Questions (CQs):

- At design time, to scope and validate Knowledge Graph construction.
- At run time, as executable primitives that orchestrate Retrieval-Augmented Generation.

Narratives are constructed as persona-conditioned sequences of narrative beats, where each beat corresponds to a tagged set of CQs whose answers are retrieved and assembled into an LLM prompt.
```

Graph-centric Retrieval-Augmented Generation (RAG) for cultural heritage storytelling, demonstrated on the Live Aid 1985 Knowledge Graph.

The system constructs and validates a structured Knowledge Graph, retrieves evidence via Competency Questions (CQs), generates persona-conditioned narratives, and evaluates support, coverage, readability, and cohesion.


## Personas and patterns

- **Personas:** `Emma`, `Luca`
- **Lengths:** `Short`, `Medium`, `Long`
- **Retrieval paradigms:** `KG`, `Hybrid`, `Graph`

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional triple store for ad hoc SPARQL
docker compose up -d
```

### Environment variables (optional)

```bash
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
```

---

## One-click KG release

Assumed working directory: `kg/scripts`. This sequence merges TTL modules, validates schema usage, then can check CQ coverage.

```bash
# 1) Merge modules -> master
python merge_all_ttls.py

# 2) Run strict vocab validation (fails if unknown terms exist)
python release_one_click.py --label "Post-schema-lock baseline"

# (Optional) CQ coverage against the working master
python run_cq_coverage.py   --kg ../liveaid_instances_master.ttl   --input ../cqs/cqs_queries_template_filled_in.rq   --out ../coverage_summary
```

**Outputs**

- Released KG: `kg/liveaid_instances_master.ttl`
- Coverage report: `kg/coverage_summary`

---

## Run narratives (grid runner)

Assumed working directory: `composite_rag_pipeline`. This triggers the respective pipelines for each combination.

```bash
# Example: Emma, Medium, all patterns, single run
python run_narrative_grid.py   --personas Emma   --lengths Medium   --patterns KG Hybrid Graph   --runs 1

# Full grid: two personas, all lengths, KG+Hybrid, 3 repeats
python run_narrative_grid.py   --personas Emma Luca   --lengths Short Medium Long   --patterns KG Hybrid   --runs 3

# Graph-only: Luca, Long, two repeats
python run_narrative_grid.py   --personas Luca   --lengths Long   --patterns Graph   --runs 2
```

**Outputs**

- Narratives and prompt logs: `composite_rag_pipeline/outputs`
- Generated stories: `composite_rag_pipeline/generator`
- Metrics and aggregates: `composite_rag_pipeline/eval`

---

## One-click evaluation

Assumed working directory: `composite_rag_pipeline`.

```bash
python one_click_rag_eval.py   --exp-dirs data/Luca-Long-20250909-133910   --rag-type KG   --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30                    --use-tfidf 1 --use-char3 1 --use-topic 1                    --topic-th 0.30 --fusion rrf --rrf-k 60                    --decision vote --vote-k 2 --emit-coverage"   --out-csv data/Luca-Long-20250909-133910/aggregated/core4_KG.csv
```

**Flags**

- `--exp-dirs` one or more run folders
- `--rag-type` `KG` or `Hybrid` or `Graph`
- `--support-extra` matcher thresholds and options for support and coverage
- `--out-csv` Core-4 CSV (support, coverage, readability and cohesion)

---

## Repository structure (top level)

- `kg/` canonical RDF graph of people, performances, venues and broadcasts. Includes enrichment scripts for external sources and release with schema and CQ validation
- `kg_enrichment_pipeline/` cleaning, validation and harmonisation into a query-safe export
- `kg_builder/` modelling helpers for classes, properties and SPO patterns
- `cq_generator/` persona scenarios to atomic facts and competency questions
- `rag_cq_sparql_embeddings/` hybrid retrieval with SPARQL, compact digests and embeddings
- `composite_rag_pipeline/` orchestration and evaluation for KG, Hybrid and Graph runs
- `rag_minimal_kg/` minimal triples-to-narrative demo

---

## Reproducibility

- Use fixed `--seed` where available
- Keep the released KG at `kg/liveaid_instances_master.ttl` for stable runs
- Thresholds in `--support-extra` control evidence matching behaviour for Support and Coverage metrics and should be fixed for reproducible comparisons.

## Troubleshooting

- **Validation fails in KG release:** unknown or out-of-schema terms were detected. Fix the offending triples and rerun the merge
- **Empty retrieval for a CQ:** confirm the CQ is covered by SPARQL templates and the KG has the required entities
- **FAISS or embeddings errors:** reinstall `faiss-cpu` and check Python version compatibility

## How to cite

If you use the LiveAid KG or the methodologies and findings from this research in your work, please cite the original paper. (*Bibliographic details are omitted here to preserve anonymity for review processes.*)

```bibtex
@inproceedings{Barla2026CQDrivenRAG,
  title={Competency Questions as Executable Plans: a Controlled RAG Architecture for Cultural Heritage Storytelling},
  author={Barla, Naga Sowjanya and de Berardinis, Jacopo},
  publisher={Springer},
  booktitle = {Semantic Web -- 23rd International Conference, ESWC 2026, Dubrovnik, Croatia, May 10--14, 2026, Proceedings},
  year={2026},
  note={To appear}
}
