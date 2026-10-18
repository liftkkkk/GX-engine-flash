# GX-Engine Flash

**English** | [简体中文](README.zh-CN.md)

> The high-performance evolution of [GX-Engine](https://github.com/liftkkkk/gx-engine): the same in-memory graph API, with NetworkX's pure-Python algorithm core replaced by C-backed libraries. Up to **15x faster** on real workloads - and NetworkX is no longer a required dependency.

GX-Engine Flash is a drop-in successor to [GX-Engine](https://github.com/liftkkkk/gx-engine). It keeps the friendly pure-Python graph API (`Graph` / `Node` / `Edge`, search, metrics, partitioning, GraphRAG) and swaps the algorithm engine underneath: every heavy operation is routed through the fastest available backend in strict priority order.

## Why Flash?

NetworkX is pure Python. On large graphs (tens of thousands of nodes) algorithms like betweenness centrality or all-pairs distances become painfully slow. GX-Engine Flash fixes this at the root:

1. **igraph first** - if igraph provides the algorithm (it almost always does), use igraph's C core.
2. **SciPy / scikit-learn second** - for the few sparse-matrix tasks where igraph does not help (spectral methods, directed eigenvector centrality), use vectorized SciPy.
3. **NetworkX last** - only as an optional fallback so the engine still works in constrained environments. NetworkX is *not* required to install or run GX-Engine Flash.

## Measured speed-ups

Random directed graph, 3,000 nodes / 12,000 edges, Windows, Python 3.13
(reproduced with `python examples/benchmark_flash.py`):

| Algorithm | GX-Flash (igraph) | NetworkX | Speed-up |
|---|---|---|---|
| Betweenness centrality | 1.06 s | 16.4 s | **15.4x** |
| All-pairs distances | 1.50 s | 14.5 s | **9.7x** |
| Clustering coefficient | 7.7 ms | 41.1 ms | **5.3x** |
| Louvain community | 86.8 ms | 279.6 ms | **3.2x** |
| Single-source Dijkstra | 9.2 ms | 8.0 ms | ~1x |

Same benchmark at **10,000 nodes / 40,000 edges** - the gap keeps widening:

| Algorithm | GX-Flash (igraph) | NetworkX | Speed-up |
|---|---|---|---|
| Betweenness centrality | 16.5 s | 4.8 min | **17.4x** |
| All-pairs distances | 22.0 s | 4.4 min | **11.9x** |
| Clustering coefficient | 38.1 ms | 146.3 ms | **3.8x** |
| Louvain community | 727 ms | 1.6 s | **2.2x** |

Every GX-Flash timing includes the igraph conversion cost, so these are
conservative numbers. The gap widens as graphs grow - and the old pure-Python
fallbacks (betweenness, hierarchical clustering with O(n^2) NetworkX calls)
are exactly where Flash shines.

## Features

Same surface as GX-Engine, now C-backed where it matters:

- **Graph data structures**: `Node` / `Edge` / `Graph`, directed & undirected, adjacency lists + name index + O(1) edge lookup (unchanged, pure Python - fast enough)
- **Graph search**: Dijkstra (full path + distance output), BFS/DFS, A*, all shortest paths, all simple paths, constrained path search - all via igraph C core
- **Graph metrics**: betweenness centrality (Brandes, C core), eigenvector centrality (ARPACK / SciPy power iteration), clustering coefficient (igraph transitivity), degree distribution
- **Graph partitioning**: Louvain (igraph C implementation, resolution support), Leiden (leidenalg), label propagation, min-cut (max-flow, C core), spectral clustering (SciPy sparse Laplacian + KMeans), balanced partitioning (spectral bisection)
- **I/O**: JSON / pickle / CSV, and **GraphML without NetworkX** (igraph native reader/writer)
- **Visualization**: interactive PyVis HTML / static Matplotlib (optional, uses NetworkX under the hood if installed)
- **Semantic retrieval (GraphRAG)**: `SemanticGraph` with node embeddings, similar-node search and multi-hop context assembly (pluggable embeddings: mock / sentence-transformers / OpenAI)

## Install

```bash
pip install igraph leidenalg numpy scipy scikit-learn   # core deps
# from source
git clone https://github.com/liftkkkk/GX-engine-flash.git
cd GX-engine-flash && pip install -e .

# optional: NetworkX compatibility layer + visualization
pip install -e ".[compat,viz]"
```

## Quick start

The API is identical to GX-Engine - existing code keeps working:

```python
from graph_engine import Graph, Node, Edge, GraphSearch, GraphMetrics, GraphPartition

g = Graph(directed=True)
n1 = Node(id="1", name="Alice", class_="person")
n2 = Node(id="2", name="Bob", class_="person")
n3 = Node(id="3", name="Carol", class_="person")
g.add_node(n1); g.add_node(n2); g.add_node(n3)
g.add_edge_from_st(n1, n2, weight=1.0, label="colleague")
g.add_edge_from_st(n2, n3, weight=2.0, label="manager")

search = GraphSearch()
print(search.get_shortest_paths(g, "1", "3"))     # igraph C core
metrics = GraphMetrics()
print(metrics.betweenness_centrality(g))          # Brandes in C, not O(V^2 E) Python
part = GraphPartition()
print(part.louvain_community_detection(g))        # igraph Louvain
```

### Backend awareness

```python
import graph_backends as gb

gb.available_backends()   # {'igraph': True, 'leidenalg': True, 'scipy': True, 'networkx': True}
gb.active_backend()       # 'igraph' - what will actually be used
```

### GraphML without NetworkX

```python
g.save_to_graphml("graph.graphml")        # igraph native writer
g2 = Graph.load_from_graphml("graph.graphml")  # igraph native reader
```

### GraphRAG semantic retrieval

```python
from graph_rag import SemanticGraph

sg = SemanticGraph()
sg.add_node_with_text("doc1", "AI applications in healthcare")
sg.add_node_with_text("doc2", "Latest advances in deep learning")
print(sg.retrieve_context("AI healthcare", hops=1))
```

### Benchmark it yourself

```bash
python examples/benchmark_flash.py --nodes 3000 --edges 12000
```

## Migration from GX-Engine

- **No code changes needed.** All classes, methods and return formats are preserved.
- NetworkX moves from required dependency to optional: only `GraphVisualizer`, `Graph.to_networkx()` and the last-resort fallbacks need `pip install networkx`.
- `GraphSearch.get_shortest_paths()` / `get_all_simple_paths()` now accept a GX `Graph` directly (recommended); NetworkX graphs are still accepted and auto-converted.
- Community detection no longer requires `python-louvain` - Louvain runs inside igraph.
- The directed clustering coefficient uses the undirected view (igraph convention, standard in the literature).

## Dependencies

Required: `igraph`, `leidenalg`, `numpy`, `scipy`, `scikit-learn`.
Optional: `networkx` (compat layer / last-resort fallback / visualization), `pyvis` / `matplotlib` (visualization), `torch` / `transformers` (real semantic embeddings).

## License

MIT

---

## Buy me a coffee

GX Graph Engine is an open-source project I develop and maintain in my spare time, free forever.

If it helped you power graph storage, retrieval, or your agent's graph substrate, consider buying me a
coffee (9.9 CNY is enough). Every bit of support goes straight into new features and bug fixes.

<p align="center">
  <img src="icon.jpg" alt="Buy me a coffee" width="280">
</p>

<p align="center"><i>Leave a note with your donation about the feature you want most - I prioritize those ;)</i></p>

**Can't donate?** Starring the repo, opening an Issue, or sharing it with someone who needs it helps just as much!
