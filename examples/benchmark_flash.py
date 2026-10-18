# -*- coding: utf-8 -*-
"""
benchmark_flash.py - GX-Engine Flash vs NetworkX performance comparison.

Runs the same heavy graph algorithms with GX-Engine Flash (igraph C core)
and pure NetworkX, and prints a speed-up table.

Usage:
    python examples/benchmark_flash.py [--nodes 3000] [--edges 12000] [--seed 42]
"""

import argparse
import os
import sys
import time

import igraph as ig
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_engine import Graph, Node, Edge
import graph_backends as gb


def build_gx_graph(nx_graph: nx.Graph) -> Graph:
    """Convert a NetworkX graph into a GX Graph (same node ids, weighted edges)."""
    g = Graph(directed=nx_graph.is_directed())
    for nid in nx_graph.nodes():
        g.add_node(Node(id=str(nid), name=f"node-{nid}"))
    for u, v, data in nx_graph.edges(data=True):
        w = float(data.get("weight", 1.0))
        g.add_edge_from_st(g.nodes[str(u)], g.nodes[str(v)], weight=w)
    return g


def random_graph(n_nodes: int, n_edges: int, seed: int = 42) -> nx.Graph:
    """Random connected-ish directed graph with positive weights (igraph generator)."""
    ig_g = ig.Graph.Erdos_Renyi(n=n_nodes, m=n_edges, directed=True)
    ig_g.es["weight"] = [1.0 + (i % 9) * 0.1 for i in range(ig_g.ecount())]

    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    for e in ig_g.es:
        g.add_edge(int(e.source), int(e.target), weight=float(e["weight"]))
    return g


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description="GX-Engine Flash vs NetworkX benchmark")
    parser.add_argument("--nodes", type=int, default=3000)
    parser.add_argument("--edges", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating random directed graph: {args.nodes} nodes, {args.edges} edges ...")
    nx_graph = random_graph(args.nodes, args.edges, args.seed)
    gx_graph = build_gx_graph(nx_graph)
    print(f"Built GX Graph: {len(gx_graph.nodes)} nodes, {len(gx_graph.edges)} edges")
    print(f"Active backend: {gb.active_backend()}  ({gb.available_backends()})")
    print()

    rows = []

    # 1. Single-source Dijkstra (weighted, all targets)
    _, t_flash = timed(gb.dijkstra_full, gx_graph, "0")
    _, t_nx = timed(nx.single_source_dijkstra, nx_graph, 0, weight="weight")
    rows.append(("Single-source Dijkstra", t_flash, t_nx))

    # 2. Betweenness centrality (unweighted for a fair, bounded run)
    _, t_flash = timed(gb.betweenness_centrality, gx_graph, normalized=True)
    _, t_nx = timed(nx.betweenness_centrality, nx_graph, normalized=True, weight=None)
    rows.append(("Betweenness centrality", t_flash, t_nx))

    # 3. Louvain community detection
    _, t_flash = timed(gb.louvain, gx_graph)
    _, t_nx = timed(nx.community.louvain_communities, nx_graph, weight="weight")
    rows.append(("Louvain community", t_flash, t_nx))

    # 4. All-pairs shortest path distances
    def _nx_all_pairs():
        return {u: dict(dists) for u, dists in nx.all_pairs_dijkstra_path_length(nx_graph, weight="weight")}

    _, t_flash = timed(gb.all_pairs_distances, gx_graph)
    _, t_nx = timed(_nx_all_pairs)
    rows.append(("All-pairs distances", t_flash, t_nx))

    # 5. Clustering coefficient
    undirected = nx_graph.to_undirected()
    gx_undirected = build_gx_graph(undirected)
    _, t_flash = timed(gb.clustering_coefficient, gx_undirected)
    _, t_nx = timed(nx.average_clustering, undirected)
    rows.append(("Clustering coefficient", t_flash, t_nx))

    header = f"{'Algorithm':<28}{'GX-Flash (igraph)':>20}{'NetworkX':>14}{'Speed-up':>10}"
    print(header)
    print("-" * len(header))
    for name, tf, tn in rows:
        speedup = tn / tf if tf > 0 else float("inf")
        print(f"{name:<28}{tf * 1000:>17.1f} ms{tn * 1000:>11.1f} ms{speedup:>8.1f}x")
    print()
    print("NOTE: igraph conversion cost is included in every GX-Flash timing,")
    print("      so these numbers are conservative (real workloads can cache the")
    print("      igraph view across calls for even larger gains).")


if __name__ == "__main__":
    main()
