#!/usr/bin/env python3
"""
Per-beat bounded graph expansion for the Enriched RAG pipeline.

Extracts seed URIs from CQ SPARQL result rows and performs a bounded
k-hop expansion over the knowledge graph, returning structured results
for traceability.

Key functions
-------------
extract_seeds_from_rows(rows, kg)   → List[URIRef]
expand_from_seeds(kg, seeds, ...)   → Dict[str, Any]
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from rdflib import Graph, URIRef

# Reuse expansion primitives from the existing graph_rag module
from graph_rag.graph_retriever import (
    GraphRetrieverConfig,
    export_triples,
    k_hop_expand,
    summarise_neighborhood,
    to_networkx,
)

# Predicates too generic to add useful per-beat signal
DEFAULT_EXCLUDE_PREDICATES: List[str] = [
    "type",
    "sameAs",
    "subClassOf",
    "subPropertyOf",
    "domain",
    "range",
]

# Match N3-encoded URI values: <http://...>
_ANGLE_URI_RE = re.compile(r"^<(https?://[^>]+)>$")
# Match bare URI values: http://...
_BARE_URI_RE  = re.compile(r"^https?://\S+$")


def extract_seeds_from_rows(
    rows: List[Dict[str, Any]],
    kg: Graph,
) -> List[URIRef]:
    """
    Extract URI-shaped values from SPARQL result rows as seed entities.

    Rows from retriever_local_rdflib contain N3-encoded values:
      "<http://...>"  → URI
      '"string"'      → literal (ignored)

    Only URIs that exist as subjects in the KG are kept, so expansion
    starts from anchored, known nodes.
    """
    seeds: List[URIRef] = []
    seen: set = set()

    for row in (rows or []):
        for k, v in row.items():
            # skip internal metadata fields added by the retriever
            if isinstance(k, str) and k.startswith("__"):
                continue
            if not isinstance(v, str):
                continue

            uri_str: Optional[str] = None
            m = _ANGLE_URI_RE.match(v.strip())
            if m:
                uri_str = m.group(1)
            elif _BARE_URI_RE.match(v.strip()):
                uri_str = v.strip()

            if uri_str and uri_str not in seen:
                ref = URIRef(uri_str)
                # Only seed from nodes that actually exist in the KG
                if (ref, None, None) in kg:
                    seeds.append(ref)
                    seen.add(uri_str)

    return seeds


def expand_from_seeds(
    kg: Graph,
    seeds: List[URIRef],
    *,
    max_hops: int = 1,
    max_nodes: int = 50,
    top_k: int = 20,
    edge_types_exclude: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Bounded k-hop expansion from seed URIs over the knowledge graph.

    Parameters
    ----------
    kg                  : loaded rdflib Graph
    seeds               : seed URIs to expand from
    max_hops            : maximum hop depth (1 or 2 recommended)
    max_nodes           : hard cap on nodes added to the subgraph
    top_k               : how many triples to keep (ranked by node degree)
    edge_types_exclude  : predicate local-names to exclude (defaults to
                          DEFAULT_EXCLUDE_PREDICATES)

    Returns
    -------
    dict with keys:
      seeds         – input seed URIs as strings
      nodes_count   – total nodes in expanded subgraph
      edges_count   – total edges in expanded subgraph
      triples       – top-k (subject, predicate, object) tuples
      provenance    – matching provenance records
      summary       – human-readable neighbourhood summary
    """
    if not seeds:
        return {
            "seeds": [],
            "nodes_count": 0,
            "edges_count": 0,
            "triples": [],
            "provenance": [],
            "summary": "",
        }

    cfg = GraphRetrieverConfig(
        k_hops=max_hops,
        max_nodes=max_nodes,
        # No community pruning — we want the full local neighbourhood per beat,
        # not the largest connected community.
        community="none",
        edge_types_exclude=edge_types_exclude or DEFAULT_EXCLUDE_PREDICATES,
        summarise_subgraph=False,
    )

    node_set, edge_list = k_hop_expand(kg, seeds, max_hops, cfg)
    G = to_networkx(node_set, edge_list, kg)
    triples, prov = export_triples(G)

    # Rank triples by combined degree of subject + object.
    # Most-connected nodes are typically the most contextually important.
    deg = dict(G.degree())
    pairs = list(zip(triples, prov))
    pairs.sort(
        key=lambda tp: deg.get(tp[0][0], 0) + deg.get(tp[0][2], 0),
        reverse=True,
    )
    top_triples = [t for t, _ in pairs[:top_k]]
    top_prov    = [p for _, p in pairs[:top_k]]

    summary = summarise_neighborhood(G, kg, max_lines=min(top_k, 12))

    return {
        "seeds": [str(s) for s in seeds],
        "nodes_count": G.number_of_nodes(),
        "edges_count": G.number_of_edges(),
        "triples": top_triples,
        "provenance": top_prov,
        "summary": summary,
    }
