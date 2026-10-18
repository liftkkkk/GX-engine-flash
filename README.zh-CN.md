# GX-Engine Flash

[English](README.md) | **简体中文**

> [GX-Engine](https://github.com/liftkkkk/gx-engine) 的高性能进化版：同样的内存图 API，底层算法引擎从 NetworkX 纯 Python 换成 C 核心，实测最高 **15 倍加速**，且 NetworkX 不再是必需依赖。

GX-Engine Flash 是 [GX-Engine](https://github.com/liftkkkk/gx-engine) 的平替升级版：完整保留原有 API（`Graph` / `Node` / `Edge`、图搜索、图指标、图分区、GraphRAG），只把底层算法引擎换掉——所有重计算操作都按严格优先级路由到最快的可用后端。

## 为什么要 Flash？

NetworkX 是纯 Python 实现。图规模上万节点后，介数中心性、全对最短距离这类算法慢到没法用。GX-Engine Flash 从根上解决这个问题：

1. **优先 igraph**——igraph 有的算法（几乎全部核心算法），直接用 igraph 的 C 核心。
2. **SciPy / scikit-learn 备用**——少数 igraph 帮不上的稀疏矩阵场景（谱方法、有向图特征向量中心性），用 SciPy 向量化实现。
3. **NetworkX 最后兜底**——仅在极端受限环境下作为可选回退，保证引擎还能跑。NetworkX **不是** GX-Engine Flash 的必需依赖。

## 实测加速比

随机有向图，3,000 节点 / 12,000 边，Windows，Python 3.13（可用
`python examples/benchmark_flash.py` 复现）：

| 算法 | GX-Flash (igraph) | NetworkX | 加速比 |
|---|---|---|---|
| 介数中心性 | 1.06 s | 16.4 s | **15.4x** |
| 全对最短距离 | 1.50 s | 14.5 s | **9.7x** |
| 聚集系数 | 7.7 ms | 41.1 ms | **5.3x** |
| Louvain 社区发现 | 86.8 ms | 279.6 ms | **3.2x** |
| 单源 Dijkstra | 9.2 ms | 8.0 ms | ~1x |

每次 GX-Flash 计时都包含 igraph 转换开销，所以这是保守数字。图越大差距越大——旧版纯 Python 实现（介数中心性、O(n²) 次调 NetworkX 的层次聚类）正是 Flash 的主战场。

## 功能特性

与 GX-Engine 同一套接口，关键路径全部 C 加速：

- **图数据结构**：`Node` / `Edge` / `Graph`，有向与无向，邻接表 + 名称索引 + O(1) 边查找（保持纯 Python，够快）
- **图搜索**：Dijkstra（完整路径 + 距离）、BFS/DFS、A*、全部最短路径、全部简单路径、带约束路径搜索——全部走 igraph C 核心
- **图指标**：介数中心性（Brandes，C 核心）、特征向量中心性（ARPACK / SciPy 幂迭代）、聚集系数（igraph transitivity）、度分布
- **图分区**：Louvain（igraph C 实现，支持 resolution）、Leiden（leidenalg）、标签传播、最小割（最大流，C 核心）、谱聚类（SciPy 稀疏拉普拉斯 + KMeans）、平衡分区（谱二分）
- **I/O**：JSON / pickle / CSV，以及**无需 NetworkX 的 GraphML**（igraph 原生读写）
- **可视化**：交互式 PyVis HTML / 静态 Matplotlib（可选，装了 NetworkX 时使用）
- **语义检索（GraphRAG）**：`SemanticGraph` 支持节点嵌入、相似节点搜索与多跳上下文组装（嵌入可插拔：mock / sentence-transformers / OpenAI）

## 安装

```bash
pip install igraph leidenalg numpy scipy scikit-learn   # 核心依赖
# 从源码安装
git clone https://github.com/liftkkkk/GX-engine-flash.git
cd GX-engine-flash && pip install -e .

# 可选：NetworkX 兼容层 + 可视化
pip install -e ".[compat,viz]"
```

## 快速上手

API 与 GX-Engine 完全一致——现有代码无需改动：

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
print(search.get_shortest_paths(g, "1", "3"))     # igraph C 核心
metrics = GraphMetrics()
print(metrics.betweenness_centrality(g))          # C 的 Brandes，不再是 O(V^2 E) 纯 Python
part = GraphPartition()
print(part.louvain_community_detection(g))        # igraph Louvain
```

### 查看当前后端

```python
import graph_backends as gb

gb.available_backends()   # {'igraph': True, 'leidenalg': True, 'scipy': True, 'networkx': True}
gb.active_backend()       # 'igraph' —— 实际生效的后端
```

### 不装 NetworkX 也能读写 GraphML

```python
g.save_to_graphml("graph.graphml")                 # igraph 原生写入
g2 = Graph.load_from_graphml("graph.graphml")      # igraph 原生读取
```

### GraphRAG 语义检索

```python
from graph_rag import SemanticGraph

sg = SemanticGraph()
sg.add_node_with_text("doc1", "AI applications in healthcare")
sg.add_node_with_text("doc2", "Latest advances in deep learning")
print(sg.retrieve_context("AI healthcare", hops=1))
```

### 自己跑基准

```bash
python examples/benchmark_flash.py --nodes 3000 --edges 12000
```

## 从 GX-Engine 迁移

- **代码零改动。** 所有类、方法和返回格式全部保留。
- NetworkX 从必需依赖降级为可选：只有 `GraphVisualizer`、`Graph.to_networkx()` 和最后兜底路径需要 `pip install networkx`。
- `GraphSearch.get_shortest_paths()` / `get_all_simple_paths()` 现在直接接受 GX `Graph`（推荐）；传 NetworkX 图也能自动转换兼容。
- 社区发现不再依赖 `python-louvain`——Louvain 直接在 igraph 里跑。
- 有向图的聚集系数采用无向视图口径（igraph 惯例，也是文献通行口径）。

## 依赖

必需：`igraph`、`leidenalg`、`numpy`、`scipy`、`scikit-learn`。
可选：`networkx`（兼容层 / 最后兜底 / 可视化）、`pyvis` / `matplotlib`（可视化）、`torch` / `transformers`（真实语义嵌入）。

## 许可证

MIT

---

## 请我喝杯咖啡

GX 图引擎是我在业余时间开发和维护的开源项目，永久免费。

如果它帮你撑起了图存储、检索，或者你 Agent 的图底座，可以考虑请我喝杯咖啡（9.9 元就够）。每一笔支持都会直接变成新功能和 bug 修复。

<p align="center">
  <img src="icon.jpg" alt="Buy me a coffee" width="280">
</p>

<p align="center"><i>打赏时留言你最想要的功能，我会优先做 ;)</i></p>

**不想打赏？** 点个 Star、提个 Issue、或者转发给需要的人，同样是对项目最大的支持！
