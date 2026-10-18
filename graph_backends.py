# -*- coding: utf-8 -*-
"""
graph_backends.py - High-performance graph algorithm backends for GX-Engine Flash.

GX-Engine Flash exists because NetworkX is pure Python and becomes painfully slow
on large graphs. This module routes every heavy graph algorithm through the
fastest available backend, in strict priority order:

  1. **igraph** (C core, pip install igraph)      - PRIMARY backend.
     Shortest paths, Dijkstra, betweenness, eigenvector centrality, clustering
     coefficient, Louvain / Leiden / label propagation, min-cut, topological
     sort, connected components, all-pairs distances.

  2. **SciPy / scikit-learn**                     - SECONDARY backends.
     Sparse-matrix tasks where igraph does not help (spectral partitioning,
     Fiedler vector, directed eigenvector centrality power iteration).

  3. **NetworkX** (optional)                      - LAST-RESORT fallback.
     Only used when neither igraph nor SciPy is installed, so the engine keeps
     working in constrained environments. NetworkX is NOT a required
     dependency of GX-Engine Flash.

Every public function accepts either a GX ``Graph`` instance or, where noted,
an ``igraph.Graph`` (useful to amortize conversion cost across many calls).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------

try:
    import igraph as ig

    HAS_IGRAPH = True
except ImportError:  # pragma: no cover
    ig = None
    HAS_IGRAPH = False

try:
    import leidenalg

    HAS_LEIDENALG = HAS_IGRAPH  # leidenalg is built on igraph
except ImportError:  # pragma: no cover
    leidenalg = None
    HAS_LEIDENALG = False

try:
    import scipy.sparse
    import scipy.sparse.csgraph

    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:  # pragma: no cover
    nx = None
    HAS_NETWORKX = False


def available_backends() -> Dict[str, bool]:
    """Report which backends are installed in the current environment."""
    return {
        "igraph": HAS_IGRAPH,
        "leidenalg": HAS_LEIDENALG,
        "scipy": HAS_SCIPY,
        "networkx": HAS_NETWORKX,
    }


def active_backend() -> str:
    """Return the name of the backend that will actually be used."""
    if HAS_IGRAPH:
        return "igraph"
    if HAS_SCIPY:
        return "scipy"
    if HAS_NETWORKX:
        return "networkx"
    return "pure-python"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _is_gx_graph(obj: Any) -> bool:
    """
    A GX Graph has dict-like ``nodes``/``edges`` mappings and ``is_directed``
    stored as a plain bool attribute (NetworkX stores it as a method, and its
    nodes/edges are views, so this distinguishes the two reliably).
    """
    return (
        hasattr(obj, "nodes")
        and hasattr(obj, "edges")
        and isinstance(getattr(obj, "is_directed", None), bool)
        and not callable(getattr(obj, "is_directed", None))
    )


def _is_nx_graph(obj: Any) -> bool:
    """NetworkX graphs expose ``is_directed`` as a method and ``adj`` mapping."""
    return (
        not _is_gx_graph(obj)
        and callable(getattr(obj, "is_directed", None))
        and hasattr(obj, "adj")
    )


def _require_igraph() -> None:
    if not HAS_IGRAPH:
        raise ImportError(
            "igraph is required for the high-performance backend. "
            "Install it with: pip install igraph"
        )


def to_igraph(graph: Any) -> "ig.Graph":
    """
    Convert a GX ``Graph`` into an ``igraph.Graph`` (C-backed).

    Node ids are stored in the ``name`` vertex attribute; edge weights are
    stored in the ``weight`` edge attribute (missing weights default to 1.0).
    """
    _require_igraph()
    if isinstance(graph, ig.Graph):
        return graph
    if _is_nx_graph(graph):
        # NetworkX graph passed in (backward compatibility) - convert directly.
        return from_nx(graph)
    if not _is_gx_graph(graph):
        raise TypeError(f"Cannot convert object of type {type(graph)!r} to igraph")

    node_ids = list(graph.nodes.keys())
    idx = {nid: i for i, nid in enumerate(node_ids)}

    sources: List[int] = []
    targets: List[int] = []
    weights: List[float] = []
    for edge in graph.edges.values():
        s = idx.get(edge.source.id)
        t = idx.get(edge.target.id)
        if s is None or t is None:
            continue
        sources.append(s)
        targets.append(t)
        w = edge.weight if getattr(edge, "weight", None) is not None else 1.0
        weights.append(float(w))

    g = ig.Graph(
        directed=bool(graph.is_directed),
        n=len(node_ids),
        edges=list(zip(sources, targets)),
    )
    g.vs["name"] = node_ids
    g.es["weight"] = weights if weights else [1.0] * g.ecount()
    return g


def from_nx(nx_graph: Any) -> "ig.Graph":
    """
    Convert a NetworkX graph into an ``igraph.Graph``.

    Kept for backward compatibility with code that used to pass NetworkX
    graphs into GraphSearch methods.
    """
    _require_igraph()
    directed = nx_graph.is_directed()
    node_ids = [str(n) for n in nx_graph.nodes()]
    idx = {n: i for i, n in enumerate(nx_graph.nodes())}

    sources, targets, weights = [], [], []
    for u, v, data in nx_graph.edges(data=True):
        sources.append(idx[u])
        targets.append(idx[v])
        weights.append(float(data.get("weight", 1.0) or 1.0))

    g = ig.Graph(directed=directed, n=len(node_ids), edges=list(zip(sources, targets)))
    g.vs["name"] = node_ids
    g.es["weight"] = weights if weights else [1.0] * g.ecount()
    return g


def _to_undirected(g: "ig.Graph") -> "ig.Graph":
    """Collapse an igraph to its undirected view (weights summed per pair)."""
    if not g.is_directed():
        return g
    try:
        return g.as_undirected(combined_edges="sum")
    except TypeError:
        # older/newer igraph without the combined_edges keyword
        return g.as_undirected()


def _vertex_index(g: "ig.Graph") -> List[str]:
    names = g.vs["name"] if "name" in g.vs.attributes() else [str(i) for i in range(g.vcount())]
    return list(names)


# ---------------------------------------------------------------------------
# Shortest paths
# ---------------------------------------------------------------------------

def shortest_path_lengths(
    graph: Any,
    source: str,
    weights: bool = True,
    ig_graph: Optional["ig.Graph"] = None,
) -> Dict[str, float]:
    """Single-source shortest path lengths -> {node_id: distance}."""
    if HAS_IGRAPH:
        g = ig_graph if ig_graph is not None else to_igraph(graph)
        names = _vertex_index(g)
        try:
            src = names.index(source)
        except ValueError:
            raise ValueError(f"源节点 {source} 不在图中")
        w = "weight" if (weights and "weight" in g.es.attributes()) else None
        dists = g.shortest_paths(source=src, weights=w)[0]
        return {names[i]: float(d) for i, d in enumerate(dists) if not math.isinf(d)}
    return _nx_shortest_lengths(graph, source, weights)


def all_pairs_distances(
    graph: Any,
    weights: bool = True,
    ig_graph: Optional["ig.Graph"] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    All-pairs shortest path distance matrix (C speed via igraph).

    Returns (matrix, node_ids). Unreachable pairs are np.inf.
    """
    if HAS_IGRAPH:
        g = ig_graph if ig_graph is not None else to_igraph(graph)
        names = _vertex_index(g)
        w = "weight" if (weights and "weight" in g.es.attributes()) else None
        return np.asarray(g.distances(weights=w), dtype=float), names
    return _nx_all_pairs_distances(graph, weights)


def all_shortest_paths(
    graph: Any,
    source: str,
    target: str,
    ig_graph: Optional["ig.Graph"] = None,
) -> List[List[str]]:
    """All shortest paths (unweighted) between two nodes as node-id lists."""
    if HAS_IGRAPH:
        g = ig_graph if ig_graph is not None else to_igraph(graph)
        names = _vertex_index(g)
        try:
            src, dst = names.index(source), names.index(target)
        except ValueError:
            raise ValueError(f"源节点 {source} 或目标节点 {target} 不在图中")
        out = []
        for vs in g.get_all_shortest_paths(src, to=dst, mode="out"):
            out.append([names[i] for i in vs])
        return out
    return _nx_all_shortest_paths(graph, source, target)


def all_simple_paths(
    graph: Any,
    source: str,
    target: str,
    cutoff: Optional[int] = None,
    ig_graph: Optional["ig.Graph"] = None,
) -> List[List[str]]:
    """
    All simple (loop-less) paths between two nodes, up to ``cutoff`` edges.

    ``cutoff=None`` means unlimited - caution, this can explode combinatorially.
    """
    if HAS_IGRAPH:
        g = ig_graph if ig_graph is not None else to_igraph(graph)
        names = _vertex_index(g)
        try:
            src, dst = names.index(source), names.index(target)
        except ValueError:
            raise ValueError(f"源节点 {source} 或目标节点 {target} 不在图中")
        out = []
        if cutoff is None:
            it = g.get_all_simple_paths(src, to=dst, mode="out")
        else:
            # igraph uses maxlen = maximum number of edges on a path
            it = g.get_all_simple_paths(src, to=dst, mode="out", maxlen=cutoff)
        for vs in it:
            out.append([names[i] for i in vs])
        return out
    return _nx_all_simple_paths(graph, source, target, cutoff)


def dijkstra_full(
    graph: Any,
    source: str,
    weights: bool = True,
    ig_graph: Optional["ig.Graph"] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Single-source Dijkstra returning {node_id: {"distance": float, "path": [ids]}}.
    Drop-in replacement for the pure-Python GraphSearch.dijkstra.
    """
    if HAS_IGRAPH:
        g = ig_graph if ig_graph is not None else to_igraph(graph)
        names = _vertex_index(g)
        try:
            src = names.index(source)
        except ValueError:
            raise ValueError(f"源节点 {source} 不在图中")
        w = "weight" if (weights and "weight" in g.es.attributes()) else None

        # Distances + predecessor lists (igraph >= 0.10 supports get_shortest_paths
        # with a list of targets; we ask for all vertices at once via distances +
        # per-target path recovery on demand).
        dists = g.shortest_paths(source=src, weights=w)[0]

        result: Dict[str, Dict[str, Any]] = {}
        # Batch path recovery: one call, all targets.
        paths = g.get_shortest_paths(src, to=list(range(g.vcount())), weights=w, mode="out")
        for i, dist in enumerate(dists):
            if math.isinf(dist):
                continue
            path_vs = paths[i]
            if not path_vs and i != src:
                continue
            result[names[i]] = {
                "distance": float(dist),
                "path": [names[j] for j in path_vs],
            }
        return result
    return _nx_dijkstra_full(graph, source, weights)


# ---------------------------------------------------------------------------
# Centrality & clustering
# ---------------------------------------------------------------------------

def betweenness_centrality(
    graph: Any,
    normalized: bool = True,
    weights: bool = True,
    ig_graph: Optional["ig.Graph"] = None,
) -> Dict[str, float]:
    """
    Betweenness centrality via igraph's Brandes algorithm (C core).

    Weighted when edge weights exist; falls back to unit weights otherwise.
    This replaces the O(V^2 * E) pure-Python implementation.
    """
    if HAS_IGRAPH:
        g = ig_graph if ig_graph is not None else to_igraph(graph)
        names = _vertex_index(g)
        w = "weight" if (weights and "weight" in g.es.attributes()) else None
        scores = g.betweenness(directed=g.is_directed(), weights=w)
        bt = {names[i]: float(s) for i, s in enumerate(scores)}

        if normalized:
            n = g.vcount()
            if n > 2:
                if g.is_directed():
                    scale = 1.0 / ((n - 1) * (n - 2))
                else:
                    scale = 2.0 / ((n - 1) * (n - 2))
                bt = {k: v * scale for k, v in bt.items()}
        return bt
    return _nx_betweenness(graph, normalized)


def eigenvector_centrality(
    graph: Any,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    ig_graph: Optional["ig.Graph"] = None,
) -> Dict[str, float]:
    """
    Eigenvector centrality.

    - Undirected graphs: igraph's ARPACK-based ``evcent`` (C core).
    - Directed graphs: vectorized SciPy power iteration matching the original
      engine's semantics (centrality flows along edges, L1-normalized).
    """
    if HAS_IGRAPH and _is_gx_graph(graph) and not graph.is_directed:
        g = ig_graph if ig_graph is not None else to_igraph(graph)
        names = _vertex_index(g)
        w = "weight" if "weight" in g.es.attributes() else None
        try:
            scores = g.evcent(weights=w, scale=True)
            return {names[i]: float(s) for i, s in enumerate(scores)}
        except Exception:
            pass  # fall through to scipy power iteration

    # SciPy vectorized power iteration (also handles directed graphs).
    if HAS_SCIPY:
        matrix, node_ids = adjacency_matrix(graph)
        n = len(node_ids)
        if n == 0:
            return {}
        vec = np.ones(n) / n
        for _ in range(max_iter):
            new_vec = matrix.T @ vec
            norm = np.sum(new_vec)
            if norm > 0:
                new_vec = new_vec / norm
            if np.max(np.abs(new_vec - vec)) < tol:
                vec = new_vec
                break
            vec = new_vec
        return {node_ids[i]: float(vec[i]) for i in range(n)}

    return _nx_eigenvector(graph, max_iter, tol)


def clustering_coefficient(
    graph: Any,
    node_id: Optional[str] = None,
    ig_graph: Optional["ig.Graph"] = None,
) -> Union[float, Dict[str, float]]:
    """
    Clustering (transitivity) coefficient via igraph.

    For directed graphs the undirected view is used (standard convention).
    ``node_id=None`` returns the average over all vertices.
    """
    if HAS_IGRAPH:
        g = ig_graph if ig_graph is not None else to_igraph(graph)
        gu = _to_undirected(g)
        names = _vertex_index(gu)
        if node_id is None:
            return float(gu.transitivity_avglocal_undirected(mode="zero"))
        try:
            v = names.index(node_id)
        except ValueError:
            raise ValueError(f"节点 {node_id} 不在图中")
        val = gu.transitivity_local_undirected(mode="zero", vertices=v)
        if isinstance(val, (list, tuple)):
            return float(val[0])
        return float(val)
    return _nx_clustering(graph, node_id)


# ---------------------------------------------------------------------------
# Community detection / partitioning
# ---------------------------------------------------------------------------

def _membership_dict(names: List[str], membership) -> Dict[str, int]:
    return {names[i]: int(m) for i, m in enumerate(membership)}


def louvain(
    graph: Any,
    resolution: float = 1.0,
    ig_graph: Optional["ig.Graph"] = None,
) -> Dict[str, int]:
    """
    Louvain community detection on the undirected view, via igraph's C
    implementation (community_multilevel). ``resolution != 1.0`` requires
    igraph >= 1.0 (native ``resolution`` parameter); on older igraph builds
    it falls back to leidenalg's RBConfiguration partition (the
    resolution-generalized Louvain objective), also at C speed.
    """
    if HAS_IGRAPH:
        g = _to_undirected(ig_graph if ig_graph is not None else to_igraph(graph))
        names = _vertex_index(g)
        w = "weight" if "weight" in g.es.attributes() else None
        try:
            communities = g.community_multilevel(
                weights=w, return_levels=False, resolution=resolution
            )
            return _membership_dict(names, communities.membership)
        except TypeError:
            # igraph < 1.0 has no resolution parameter
            if abs(resolution - 1.0) < 1e-9:
                communities = g.community_multilevel(weights=w, return_levels=False)
                return _membership_dict(names, communities.membership)
            if HAS_LEIDENALG:
                part = leidenalg.find_partition(
                    g,
                    leidenalg.RBConfigurationVertexPartition,
                    weights=w,
                    resolution_parameter=resolution,
                )
                return _membership_dict(names, part.membership)
            raise ImportError(
                "leidenalg is required for resolution != 1.0 on igraph < 1.0: "
                "pip install leidenalg"
            )
    return _nx_louvain(graph, resolution)


def leiden(
    graph: Any,
    resolution: float = 1.0,
    ig_graph: Optional["ig.Graph"] = None,
) -> Dict[str, int]:
    """Leiden community detection (leidenalg on igraph, C speed)."""
    if HAS_LEIDENALG:
        g = _to_undirected(ig_graph if ig_graph is not None else to_igraph(graph))
        names = _vertex_index(g)
        w = "weight" if "weight" in g.es.attributes() else None
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=w,
            resolution_parameter=resolution,
        )
        return _membership_dict(names, part.membership)
    # graceful degradation: plain Louvain is a close cousin
    return louvain(graph, resolution)


def label_propagation(graph: Any, ig_graph: Optional["ig.Graph"] = None) -> Dict[str, int]:
    """Label propagation community detection via igraph (undirected view)."""
    if HAS_IGRAPH:
        g = _to_undirected(ig_graph if ig_graph is not None else to_igraph(graph))
        names = _vertex_index(g)
        w = "weight" if "weight" in g.es.attributes() else None
        communities = g.community_label_propagation(weights=w)
        return _membership_dict(names, communities.membership)
    return _nx_label_propagation(graph)


def modularity(graph: Any, partition: Dict[str, List[str]]) -> float:
    """
    Modularity of a partition {community_id: [node_ids]} via igraph (C speed).
    """
    if HAS_IGRAPH:
        g = _to_undirected(to_igraph(graph))
        names = _vertex_index(g)
        node_community: Dict[str, int] = {}
        for cid, nodes in partition.items():
            for nid in nodes:
                node_community[nid] = int(cid)
        membership = [node_community.get(nid, -1) for nid in names]
        if min(membership) < 0:
            raise ValueError("partition 未覆盖图中的全部节点")
        w = "weight" if "weight" in g.es.attributes() else None
        try:
            return float(g.modularity(membership, weights=w))
        except Exception:
            return float(g.modularity(membership))
    return _nx_modularity(graph, partition)


def min_cut_partition(graph: Any, ig_graph: Optional["ig.Graph"] = None) -> Dict[str, List[str]]:
    """
    Two-way min-cut partition, mirroring the original engine's heuristics:

    1. restrict to the largest connected component,
    2. pick the farthest vertex pair as source/target,
    3. run a max-flow min-cut with edge weights as capacities.
    """
    if HAS_IGRAPH:
        g = ig_graph if ig_graph is not None else to_igraph(graph)
        # Largest connected component first
        comps = g.connected_components(mode="weak" if g.is_directed() else "strong")
        if len(comps) > 1:
            largest = max(comps, key=len)
            g = g.induced_subgraph(largest)
        names = _vertex_index(g)
        if g.vcount() <= 2:
            return {"0": list(names)}

        # Symmetric capacities (the original built both directions per edge)
        flow_g = g.copy()
        if not flow_g.is_directed():
            flow_g.to_directed()

        # Farthest pair by unweighted hop distance
        dist_matrix = np.asarray(flow_g.distances(mode="out"), dtype=float)
        np.nan_to_num(dist_matrix, copy=False, posinf=-1.0)
        flat = np.argmax(dist_matrix)
        i, j = np.unravel_index(flat, dist_matrix.shape)
        source, target = int(i), int(j)
        if dist_matrix[i, j] <= 0:
            source, target = 0, flow_g.vcount() - 1

        cap = "weight" if "weight" in flow_g.es.attributes() else None
        try:
            cut = flow_g.st_mincut(source=source, target=target, capacity=cap)
        except Exception as exc:
            print(f"最小割算法失败: {exc}，使用谱聚类作为备选")
            from graph_engine import GraphPartition  # local import to avoid a cycle

            return GraphPartition().spectral_partition(graph, 2)

        return {
            "0": [names[v] for v in cut.partition[0]],
            "1": [names[v] for v in cut.partition[1]],
        }
    return _nx_min_cut(graph)


def largest_connected_component(graph: Any, ig_graph: Optional["ig.Graph"] = None) -> List[str]:
    """Node ids of the largest (weakly) connected component."""
    if HAS_IGRAPH:
        g = ig_graph if ig_graph is not None else to_igraph(graph)
        comps = g.connected_components(mode="weak" if g.is_directed() else "strong")
        largest = max(comps, key=len)
        names = _vertex_index(g)
        return [names[v] for v in largest]
    return _nx_largest_cc(graph)


def is_connected(graph: Any, ig_graph: Optional["ig.Graph"] = None) -> bool:
    """Whether the graph has exactly one connected component."""
    if HAS_IGRAPH:
        g = ig_graph if ig_graph is not None else to_igraph(graph)
        comps = g.connected_components(mode="weak" if g.is_directed() else "strong")
        return len(comps) <= 1
    return _nx_is_connected(graph)


def topological_sort(graph: Any, ig_graph: Optional["ig.Graph"] = None) -> List[str]:
    """Topological order of a DAG via igraph (raises on cycles)."""
    if HAS_IGRAPH:
        g = ig_graph if ig_graph is not None else to_igraph(graph)
        order = g.topological_sorting(mode="out")
        names = _vertex_index(g)
        if len(order) != g.vcount():
            raise ValueError("图中存在环，无法进行拓扑排序")
        return [names[v] for v in order]
    return _nx_topological_sort(graph)


# ---------------------------------------------------------------------------
# Spectral methods (SciPy / scikit-learn secondary backends)
# ---------------------------------------------------------------------------

def adjacency_matrix(
    graph: Any,
    ig_graph: Optional["ig.Graph"] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Dense adjacency matrix (rows follow the returned node id order).

    Returned node ids always match ``list(graph.nodes.keys())`` for GX Graphs.
    """
    if HAS_IGRAPH:
        g = ig_graph if ig_graph is not None else to_igraph(graph)
        names = _vertex_index(g)
        try:
            mat = np.asarray(g.get_adjacency(type="adjacency").data, dtype=float)
        except (ValueError, TypeError):
            mat = np.asarray(g.get_adjacency().data, dtype=float)
        return mat, names
    if HAS_SCIPY:
        return _scipy_adjacency(graph)
    return _nx_adjacency(graph)


def sparse_laplacian(
    graph: Any,
    ig_graph: Optional["ig.Graph"] = None,
):
    """
    Sparse unnormalized Laplacian (D - A) via SciPy, for spectral methods.
    Returns (laplacian_sparse, node_ids).
    """
    if HAS_SCIPY:
        if HAS_IGRAPH:
            g = ig_graph if ig_graph is not None else to_igraph(graph)
            names = _vertex_index(g)
            a = scipy.sparse.csr_matrix(np.asarray(g.get_adjacency().data, dtype=float))
        else:
            a_dense, names = _scipy_adjacency(graph)
            a = scipy.sparse.csr_matrix(a_dense)
        lap = scipy.sparse.csgraph.laplacian(a, normed=False)
        return lap, names
    raise ImportError("scipy is required for spectral methods: pip install scipy")


def fiedler_vector(graph: Any) -> Tuple[np.ndarray, List[str]]:
    """
    Fiedler vector (2nd smallest eigenvector of the Laplacian) for bisection.
    Uses dense eigendecomposition for small graphs and ARPACK for large ones.
    """
    lap, names = sparse_laplacian(graph)
    n = lap.shape[0]
    if n <= 2:
        return np.array([1.0, -1.0][:n]), names
    if n < 500:
        vals, vecs = np.linalg.eigh(np.asarray(lap.todense()))
        return vecs[:, 1], names
    vals, vecs = scipy.sparse.linalg.eigsh(lap.astype(float), k=2, sigma=-1e-5, which="LM")
    order = np.argsort(vals)
    return vecs[:, order[1]], names


# ---------------------------------------------------------------------------
# GraphML I/O without NetworkX
# ---------------------------------------------------------------------------

def save_graphml(graph: Any, filepath: str, ig_graph: Optional["ig.Graph"] = None) -> None:
    """Save a GX Graph as GraphML using igraph's native writer (no NetworkX)."""
    _require_igraph()
    g = ig_graph if ig_graph is not None else to_igraph(graph)
    names = _vertex_index(g)

    node_attrs = {}
    if _is_gx_graph(graph):
        for nid in names:
            node = graph.nodes[nid]
            node_attrs[nid] = {
                "gx_name": node.name if node.name is not None else "",
                "gx_class": node.class_ if node.class_ is not None else "",
                "gx_weight": float(node.weight) if node.weight is not None else 1.0,
            }
    else:
        for nid in names:
            node_attrs[nid] = {}

    g.vs["id"] = names
    g.vs["gx_name"] = [node_attrs[nid]["gx_name"] for nid in names]
    g.vs["gx_class"] = [node_attrs[nid]["gx_class"] for nid in names]
    g.vs["gx_weight"] = [node_attrs[nid]["gx_weight"] for nid in names]

    edge_labels = [""] * g.ecount()
    if _is_gx_graph(graph):
        # Map GX edge list onto igraph edge ids (multi-edges keep order)
        idx = {nid: i for i, nid in enumerate(names)}
        label_by_pair: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for edge in graph.edges.values():
            s, t = idx[edge.source.id], idx[edge.target.id]
            label_by_pair[(s, t)].append(edge.label if edge.label else "")
        for e in g.es:
            pair_labels = label_by_pair.get((e.source, e.target))
            if pair_labels:
                edge_labels[e.index] = pair_labels.pop(0) or ""
    g.es["gx_label"] = edge_labels

    g.write_graphml(filepath)


def load_graphml(filepath: str):
    """Load a GraphML file into a GX Graph using igraph's native reader."""
    _require_igraph()
    import uuid

    from graph_engine import Edge, Graph, Node

    g = ig.Graph.Read_GraphML(filepath)
    directed = g.is_directed()
    graph = Graph(directed=directed)

    names = (
        list(g.vs["id"])
        if "id" in g.vs.attributes()
        else [str(i) for i in range(g.vcount())]
    )

    for i, nid in enumerate(names):
        attrs = g.vs[i].attributes()
        name = attrs.get("gx_name") or nid
        class_ = attrs.get("gx_class") or None
        weight = attrs.get("gx_weight")
        if weight in (None, ""):
            weight = None
        node = Node(id=str(nid), name=name, class_=class_, weight=weight)
        extra = {
            k: v
            for k, v in attrs.items()
            if k not in ("id", "gx_name", "gx_class", "gx_weight") and v not in (None, "")
        }
        if extra:
            node.attributes.update(extra)
        graph.add_node(node)

    for e in g.es:
        s = str(names[e.source])
        t = str(names[e.target])
        attrs = e.attributes()
        weight = attrs.get("weight")
        try:
            weight = float(weight) if weight not in (None, "") else 1.0
        except (TypeError, ValueError):
            weight = 1.0
        edge = Edge(
            id=str(uuid.uuid1()),
            source=graph.nodes[s],
            target=graph.nodes[t],
            weight=weight,
            label=attrs.get("gx_label") or attrs.get("label") or None,
        )
        extra = {
            k: v
            for k, v in attrs.items()
            if k not in ("weight", "gx_label", "label") and v not in (None, "")
        }
        if extra:
            edge.attributes.update(extra)
        graph.add_edge(edge)

    return graph


# ===========================================================================
# NetworkX last-resort fallbacks (private)
# ===========================================================================

def _require_nx():
    if not HAS_NETWORKX:
        raise ImportError(
            "None of the fast backends (igraph/scipy) is installed and NetworkX "
            "is missing too. Install igraph for best performance: pip install igraph"
        )
    return nx


def _to_nx(graph: Any):
    n = _require_nx()
    if not _is_gx_graph(graph):
        return graph
    g = n.DiGraph() if graph.is_directed else n.Graph()
    for nid in graph.nodes:
        g.add_node(nid)
    for edge in graph.edges.values():
        w = edge.weight if getattr(edge, "weight", None) is not None else 1.0
        g.add_edge(edge.source.id, edge.target.id, weight=float(w))
    return g


def _nx_shortest_lengths(graph, source, weights=True):
    g = _to_nx(graph)
    return dict(nx.shortest_path_length(g, source=source, weight="weight" if weights else None))


def _nx_all_pairs_distances(graph, weights=True):
    g = _to_nx(graph)
    node_ids = list(g.nodes())
    sp = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight" if weights else None))
    m = np.full((len(node_ids), len(node_ids)), np.inf)
    idx = {n: i for i, n in enumerate(node_ids)}
    for u, dists in sp.items():
        for v, d in dists.items():
            m[idx[u], idx[v]] = d
    return m, node_ids


def _nx_all_shortest_paths(graph, source, target):
    g = _to_nx(graph)
    try:
        return [list(p) for p in nx.all_shortest_paths(g, source=source, target=target)]
    except nx.NetworkXNoPath:
        return []


def _nx_all_simple_paths(graph, source, target, cutoff=None):
    g = _to_nx(graph)
    try:
        return [
            list(p)
            for p in nx.all_simple_paths(g, source=source, target=target, cutoff=cutoff)
        ]
    except nx.NetworkXNoPath:
        return []


def _nx_dijkstra_full(graph, source, weights=True):
    g = _to_nx(graph)
    w = "weight" if weights else None
    dists = nx.shortest_path_length(g, source=source, weight=w)
    paths = nx.shortest_path(g, source=source, weight=w)
    return {
        t: {"distance": float(dists[t]), "path": paths[t]} for t in dists if t in paths
    }


def _nx_betweenness(graph, normalized=True):
    g = _to_nx(graph)
    return nx.betweenness_centrality(g, normalized=normalized, weight="weight")


def _nx_eigenvector(graph, max_iter, tol):
    g = _to_nx(graph)
    try:
        return nx.eigenvector_centrality(g, max_iter=max_iter, tol=tol, weight="weight")
    except nx.PowerIterationFailedConvergence:
        return nx.eigenvector_centrality(g, max_iter=max_iter, weight=None)


def _nx_clustering(graph, node_id=None):
    g = _to_nx(graph)
    gu = g.to_undirected() if g.is_directed() else g
    if node_id is None:
        return nx.average_clustering(gu)
    return nx.clustering(gu, node_id)


def _nx_louvain(graph, resolution):
    try:
        from community import best_partition
    except ImportError as exc:
        raise ImportError("需要安装python-louvain库: pip install python-louvain") from exc
    g = _to_nx(graph)
    gu = g.to_undirected() if g.is_directed() else g
    return dict(best_partition(gu, resolution=resolution))


def _nx_label_propagation(graph):
    g = _to_nx(graph)
    gu = g.to_undirected() if g.is_directed() else g
    result = {}
    for cid, community in enumerate(nx.community.label_propagation_communities(gu)):
        for nid in community:
            result[nid] = cid
    return result


def _nx_modularity(graph, partition):
    try:
        from networkx.algorithms.community import modularity as nx_mod
    except ImportError as exc:
        raise ImportError("networkx is required for the modularity fallback") from exc
    g = _to_nx(graph)
    gu = g.to_undirected() if g.is_directed() else g
    communities = [set(nodes) for nodes in partition.values()]
    try:
        return float(nx_mod(gu, communities, weight="weight"))
    except Exception:
        return float(nx_mod(gu, communities))


def _nx_min_cut(graph):
    g = _to_nx(graph)
    gu = g.to_undirected() if g.is_directed() else g
    if not nx.is_connected(gu):
        largest_cc = max(nx.connected_components(gu), key=len)
        gu = gu.subgraph(largest_cc).copy()
    if len(gu.nodes()) <= 2:
        return {"0": list(gu.nodes())}
    flow = nx.DiGraph()
    flow.add_nodes_from(gu.nodes())
    for u, v, data in gu.edges(data=True):
        c = float(data.get("weight", 1.0) or 1.0)
        flow.add_edge(u, v, capacity=c)
        flow.add_edge(v, u, capacity=c)
    distances = dict(nx.all_pairs_shortest_path_length(flow))
    max_dist, source, target = 0, None, None
    for u, dists in distances.items():
        for v, d in dists.items():
            if d > max_dist:
                max_dist, source, target = d, u, v
    if source is None or target is None or source == target:
        import random

        nodes = list(flow.nodes())
        if len(nodes) >= 2:
            source, target = random.sample(nodes, 2)
        else:
            return {"0": nodes}
    try:
        cut_value, partition = nx.minimum_cut(flow, source, target)
        return {"0": list(partition[0]), "1": list(partition[1])}
    except Exception as exc:
        print(f"最小割算法失败: {exc}，使用谱聚类作为备选")
        from graph_engine import GraphPartition

        return GraphPartition().spectral_partition(graph, 2)


def _nx_largest_cc(graph):
    g = _to_nx(graph)
    gu = g.to_undirected() if g.is_directed() else g
    return list(max(nx.connected_components(gu), key=len))


def _nx_is_connected(graph):
    g = _to_nx(graph)
    gu = g.to_undirected() if g.is_directed() else g
    return nx.is_connected(gu)


def _nx_topological_sort(graph):
    g = _to_nx(graph)
    return list(nx.topological_sort(g))


def _scipy_adjacency(graph):
    """Build a dense adjacency matrix without igraph/networkx (GX Graph input)."""
    node_ids = list(graph.nodes.keys())
    n = len(node_ids)
    idx = {nid: i for i, nid in enumerate(node_ids)}
    rows, cols, vals = [], [], []
    for edge in graph.edges.values():
        s, t = idx.get(edge.source.id), idx.get(edge.target.id)
        if s is None or t is None:
            continue
        w = edge.weight if getattr(edge, "weight", None) is not None else 1.0
        rows.append(s)
        cols.append(t)
        vals.append(float(w))
    a = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return np.asarray(a.todense(), dtype=float), node_ids


def _nx_adjacency(graph):
    g = _to_nx(graph)
    node_ids = [str(x) for x in g.nodes()]
    return nx.to_numpy_array(g), node_ids
