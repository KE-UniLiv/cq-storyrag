#!/usr/bin/env python3
"""
Evidence pack builder for the Enriched RAG pipeline.

Merges CQ SPARQL results (direct evidence) with per-beat bounded graph
expansion (supporting evidence).  Each item in plan_with_evidence is
augmented in place with:

  expanded_rows    – graph-expansion triples as row-style dicts
  expansion_trace  – seed URIs, node/edge counts, and summary (for
                     traceability and evaluation)

The original plan_with_evidence dict is never mutated; a deep copy is
returned so the caller can diff before/after if needed.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from rdflib import Graph

from retriever.graph_expander import expand_from_seeds, extract_seeds_from_rows


def build_evidence_packs(
    plan_with_evidence: Dict[str, Any],
    kg: Graph,
    *,
    max_hops: int = 1,
    max_nodes: int = 50,
    top_k: int = 20,
    edge_types_exclude: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Augment every item in *plan_with_evidence* with graph-expanded evidence.

    Parameters
    ----------
    plan_with_evidence  : output of retriever_local_rdflib.run()
    kg                  : loaded rdflib Graph (shared across all beats)
    max_hops            : hop depth for graph expansion
    max_nodes           : max subgraph nodes per item
    top_k               : max triples kept per item (degree-ranked)
    edge_types_exclude  : predicate local-names to exclude from expansion

    Returns
    -------
    A new dict (deep copy of plan_with_evidence) where each item has two
    additional fields:

    expanded_rows : list[dict]
        Each entry: {"subject", "predicate", "object", "__source": "graph_expansion"}
        These are row-compatible with the generator's fact-packing logic.

    expansion_trace : dict
        {"seeds", "nodes_count", "edges_count", "triples_kept", "summary"}
        Provides full traceability for evaluation and debugging.
    """
    result = copy.deepcopy(plan_with_evidence)

    for item in result.get("items", []):
        rows = item.get("rows") or []
        seeds = extract_seeds_from_rows(rows, kg)

        if not seeds:
            item["expanded_rows"] = []
            item["expansion_trace"] = {
                "seeds": [],
                "nodes_count": 0,
                "edges_count": 0,
                "triples_kept": 0,
                "reason": "no_seeds_found",
            }
            continue

        exp = expand_from_seeds(
            kg,
            seeds,
            max_hops=max_hops,
            max_nodes=max_nodes,
            top_k=top_k,
            edge_types_exclude=edge_types_exclude,
        )

        # Convert triples to row-style dicts so the generator can handle
        # them with the same factlet-building logic as direct KG rows.
        expanded_rows = [
            {
                "subject":   s,
                "predicate": p,
                "object":    o,
                "__source":  "graph_expansion",
            }
            for s, p, o in exp["triples"]
        ]

        item["expanded_rows"] = expanded_rows
        item["expansion_trace"] = {
            "seeds":        exp["seeds"],
            "nodes_count":  exp["nodes_count"],
            "edges_count":  exp["edges_count"],
            "triples_kept": len(expanded_rows),
            "summary":      exp["summary"],
        }

    return result
