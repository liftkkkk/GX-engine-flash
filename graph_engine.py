from __future__ import annotations

from dataclasses import dataclass, field
import uuid
import time
import json
import pickle
import os
from typing import Dict, List, Optional, DefaultDict, Any, Callable, Union, Set, Iterator, Tuple

import graph_backends as gb

# NetworkX 在 GX-Engine Flash 中是可选依赖（仅作兼容接口与可视化使用）
try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    nx = None
    _HAS_NETWORKX = False

from collections import defaultdict, OrderedDict
import numpy as np
from sklearn.cluster import KMeans
from functools import lru_cache

def _nested_dict_factory():
    """用于pickle序列化的辅助函数"""
    return defaultdict(list)

############# 数据结构 ############
@dataclass
class Node:
    """图中的节点类，存储节点的基本信息"""
    id: str
    name: str
    class_: Optional[str] = None  # 替代clazz，避免与关键字冲突
    weight: Optional[float] = None
    data: Optional[str] = None
    path: List = field(default_factory=list)
    # 新增字段以支持动态图和多维度属性
    timestamp: float = field(default_factory=time.time)
    version: int = 1
    attributes: Dict[str, Any] = field(default_factory=dict)
    # 向量嵌入支持
    embedding: Optional[List[float]] = None

    def to_str(self) -> str:
        """返回节点的字符串表示"""
        return f"{self.id}: {self.name} [{self.class_}]"
    
    def convert_dict(self) -> Dict[str, Any]:
        """将节点转换为字典表示"""
        return {
            "id": self.id, 
            "name": self.name, 
            "type": self.class_,
            "weight": self.weight,
            "timestamp": self.timestamp,
            "version": self.version,
            "attributes": self.attributes,
            "embedding": self.embedding
        }
    
    def __eq__(self, other) -> bool:
        """节点相等性比较，基于节点ID"""
        if not isinstance(other, Node):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """节点哈希值，基于节点ID"""
        return hash(self.id)
    
    def update(self, **kwargs) -> None:
        """更新节点属性，并增加版本号"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.version += 1
        self.timestamp = time.time()


@dataclass
class Edge:
    """图中的边类，存储边的连接信息和属性"""
    id: str
    source: Node
    target: Node
    weight: float
    label: Optional[str] = None
    # 新增字段以支持动态图和多维度属性
    timestamp: float = field(default_factory=time.time)
    version: int = 1
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_str(self) -> str:
        """返回边的字符串表示"""
        return (f"{self.source.id}:{self.source.name}-> {self.target.id}:{self.target.name} "
                f"[weight: {self.weight}, label:{self.label}]")
    
    def convert_dict(self) -> Dict[str, Any]:
        """将边转换为字典表示"""
        return {
            "id": self.id,
            "source_id": self.source.id,
            "target_id": self.target.id,
            "weight": self.weight,
            "label": self.label,
            "timestamp": self.timestamp,
            "version": self.version,
            "attributes": self.attributes,
            "embedding": self.embedding
        }
    
    def __eq__(self, other) -> bool:
        """边相等性比较，基于边ID"""
        if not isinstance(other, Edge):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """边哈希值，基于边ID"""
        return hash(self.id)
    
    def update(self, **kwargs) -> None:
        """更新边属性，并增加版本号"""
        for key, value in kwargs.items():
            if key == 'source' or key == 'target':
                continue  # 不允许直接修改源和目标节点
            if hasattr(self, key):
                setattr(self, key, value)
        self.version += 1
        self.timestamp = time.time()


class Graph:
    """图类，包含节点和边的管理及图操作方法"""
    
    def __init__(self, directed: bool = True) -> None:
        self.size: int = 0  # 节点数量
        self.adj: DefaultDict[str, List[Edge]] = defaultdict(list)  # 邻接表，存储边
        self.nodes: Dict[str, Node] = {}  # 节点集合，id到Node的映射
        # 增强索引系统
        self.edges: Dict[str, Edge] = {}  # 边ID到Edge的映射，O(1)访问
        self.node_by_name: Dict[str, List[Node]] = defaultdict(list)  # 按名称索引节点
        self.edges_by_label: Dict[str, List[Edge]] = defaultdict(list)  # 按标签索引边
        # 优化索引：源节点->目标节点->边列表，提供O(1)边查找
        self.edge_lookup: DefaultDict[str, DefaultDict[str, List[Edge]]] = defaultdict(_nested_dict_factory)
        self.is_directed: bool = directed  # 图类型标记，默认为有向图
        # 版本控制，用于缓存验证
        self.version: int = 1
        self.timestamp: float = time.time()

    def get_neighbors(self, node: Node) -> List[Edge]:
        """获取节点的所有邻接边"""
        if node.id not in self.adj:
            return []
        return self.adj[node.id]

    def get_edge(self, source: Node, target: Node, weight: float = 1.0, label: Optional[str] = None) -> Edge:
        """获取从源节点到目标节点的边，不存在则创建"""
        # 优化：使用edge_lookup进行O(1)查找
        if source.id in self.edge_lookup and target.id in self.edge_lookup[source.id]:
            edges = self.edge_lookup[source.id][target.id]
            for edge in edges:
                if label is None or edge.label == label:
                    return edge
        
        # 未找到则创建新边
        edge_id = str(uuid.uuid1())
        new_edge = Edge(edge_id, source, target, weight, label)
        self.add_edge(new_edge)
        return new_edge

    def add_edge(self, edge: Edge) -> None:
        """添加边到图中，并确保边的两个节点已存在于图中"""
        self.adj[edge.source.id].append(edge)
        
        # 确保源节点和目标节点在节点集合中
        self.nodes[edge.source.id] = edge.source
        self.nodes[edge.target.id] = edge.target
        self.size = len(self.nodes)  # 更新节点数量
        
        # 更新边索引
        self.edges[edge.id] = edge
        if edge.label:
            self.edges_by_label[edge.label].append(edge)
        
        # 更新快速查找索引
        self.edge_lookup[edge.source.id][edge.target.id].append(edge)
        
        # 更新图版本
        self.version += 1
        self.timestamp = time.time()
        
        # 如果是无向图，添加反向边
        if not self.is_directed:
            # 检查反向边是否已存在
            reverse_edge_exists = False
            if edge.target.id in self.edge_lookup and edge.source.id in self.edge_lookup[edge.target.id]:
                reverse_edge_exists = True
                
            if not reverse_edge_exists:
                reverse_edge_id = str(uuid.uuid1())
                reverse_edge = Edge(
                    id=reverse_edge_id,
                    source=edge.target,
                    target=edge.source,
                    weight=edge.weight,
                    label=edge.label,
                    attributes=edge.attributes.copy()
                )
                self.adj[edge.target.id].append(reverse_edge)
                self.edges[reverse_edge.id] = reverse_edge
                if reverse_edge.label:
                    self.edges_by_label[reverse_edge.label].append(reverse_edge)
                # 更新反向边索引
                self.edge_lookup[reverse_edge.source.id][reverse_edge.target.id].append(reverse_edge)

    def add_edge_from_st(self, source: Node, target: Node, weight: float = 1.0, label: Optional[str] = None) -> None:
        """通过源节点和目标节点直接添加边"""
        edge_id = str(uuid.uuid1())
        edge = Edge(edge_id, source, target, weight, label)
        self.add_edge(edge)

    def get_node_create(self, node_id: str, name: Optional[str] = None, class_: Optional[str] = None) -> Node:
        """获取节点，若不存在则创建新节点"""
        if node_id in self.nodes:
            return self.nodes[node_id]
        self.add_node(node_id, name, class_)
        return self.nodes[node_id]

    def get_node(self, node_id: str) -> Optional[Node]:
        """仅获取节点，不存在则返回None"""
        return self.nodes.get(node_id)

    def get_all_edges(self) -> Iterator[Edge]:
        """获取图中所有的边"""
        return iter(self.edges.values())
    
    def has_edge(self, node1: str, node2: str) -> bool:
        """检查两个节点之间是否存在边
        
        参数:
            node1: 源节点ID
            node2: 目标节点ID
        
        返回:
            如果存在从node1到node2的边，则返回True，否则返回False
        """
        if node1 not in self.edge_lookup:
            return False
        
        return node2 in self.edge_lookup[node1] and len(self.edge_lookup[node1][node2]) > 0
    
    def add_node_from_str(self, node_id: str, name: Optional[str], class_: Optional[str]) -> None:
        """添加节点到图中，若节点已存在则不操作"""
        if node_id not in self.nodes and name is not None:
            self.nodes[node_id] = Node(node_id, name, class_)
            self.size = len(self.nodes)  # 更新节点数量
            # 更新图版本
            self.version += 1
            self.timestamp = time.time()

    def add_node(self, node: Node) -> None:
        """添加节点到图中，若节点已存在则不操作"""
        if node.id not in self.nodes and node.name is not None:
            self.nodes[node.id] = node  # 直接存储节点对象，保留所有属性
            # 更新名称索引
            if node.name:
                self.node_by_name[node.name].append(node)
            self.size = len(self.nodes)  # 更新节点数量
            
            # 更新图版本
            self.version += 1
            self.timestamp = time.time()

    def delete_node(self, node_id: str) -> None:
        """从图中删除节点及其相关边"""
        if node_id not in self.nodes:
            raise ValueError(f"节点 {node_id} 不存在，无法删除")
        
        # 获取节点对象用于更新名称索引
        node = self.nodes[node_id]
        
        # 删除节点
        del self.nodes[node_id]
        self.size = len(self.nodes)
        
        # 从名称索引中移除
        if node.name and node.name in self.node_by_name:
            self.node_by_name[node.name] = [n for n in self.node_by_name[node.name] if n.id != node_id]
        
        # 删除以该节点为源的边
        edges_to_delete = []
        if node_id in self.adj:
            edges_to_delete.extend(self.adj[node_id])
            del self.adj[node_id]
        
        # 删除以该节点为目标的边
        for source_id in list(self.adj.keys()):
            new_edges = []
            for edge in self.adj[source_id]:
                if edge.target.id == node_id:
                    edges_to_delete.append(edge)
                else:
                    new_edges.append(edge)
            self.adj[source_id] = new_edges
        
        # 更新边索引
        for edge in edges_to_delete:
            if edge.id in self.edges:
                del self.edges[edge.id]
            if edge.label and edge.label in self.edges_by_label:
                self.edges_by_label[edge.label] = [e for e in self.edges_by_label[edge.label] if e.id != edge.id]
            
            # 从快速查找索引中移除
            source_id = edge.source.id
            target_id = edge.target.id
            if source_id in self.edge_lookup and target_id in self.edge_lookup[source_id]:
                self.edge_lookup[source_id][target_id] = [e for e in self.edge_lookup[source_id][target_id] if e.id != edge.id]
                if not self.edge_lookup[source_id][target_id]:
                    del self.edge_lookup[source_id][target_id]
                if not self.edge_lookup[source_id]:
                    del self.edge_lookup[source_id]
                    
        # 更新图版本
        self.version += 1
        self.timestamp = time.time()

    def to_str(self) -> str:
        """返回图的字符串表示"""
        # 节点信息
        nodes_str = " | ".join([node.to_str() for node in self.nodes.values()])
        
        # 边信息
        edges_str_list = []
        for source_id, edges in self.adj.items():
            source_node = self.nodes.get(source_id)
            source_name = source_node.name if source_node else source_id
            edges_str = " | ".join([edge.to_str() for edge in edges])
            edges_str_list.append(f"{source_name} 的边: {edges_str}")
        edges_str = "\n".join(edges_str_list)
        
        return f"==== 节点 ====\n{nodes_str}\n==== 边 ====\n{edges_str}"

    def to_networkx(self):
        """
        将当前图转换为networkx有向图（兼容接口，需要可选的 networkx 依赖）
        """
        if not _HAS_NETWORKX:
            raise ImportError(
                "NetworkX库未安装。GX-Engine Flash 不强制依赖 networkx；"
                "如需此兼容接口请安装: pip install gx-engine-flash[compat]"
            )
        # 添加节点
        nodes = [
            (node.id, {"label": node.name, "weight": node.weight, "class": node.class_})
            for node in self.nodes.values()
        ]
        
        # 添加边
        edges = []
        for edges_list in self.adj.values():
            for edge in edges_list:
                edge_data = {
                    "label": edge.label,
                    "weight": edge.weight
                }
                edges.append((edge.source.id, edge.target.id, edge_data))
        
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph
    
    def to_pyg(self) -> Any:
        """
        将当前图转换为PyTorch Geometric (PyG) Data对象
        
        返回:
            PyTorch Geometric Data对象
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("PyTorch Geometric库未安装，请先安装: pip install torch_geometric")
        
        # 为节点创建连续的索引映射
        node_ids = list(self.nodes.keys())
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        num_nodes = len(node_ids)
        
        # 构建边索引
        edge_index = []
        edge_weights = []
        for edge in self.edges.values():
            source_idx = node_id_to_idx[edge.source.id]
            target_idx = node_id_to_idx[edge.target.id]
            edge_index.append([source_idx, target_idx])
            edge_weights.append(edge.weight)
        
        # 转换为PyTorch张量
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        # 构建节点特征
        # 默认使用节点的weight作为特征，如果没有则使用1.0
        x = []
        for node_id in node_ids:
            node = self.nodes[node_id]
            # 基础特征：weight
            feature = [node.weight if node.weight is not None else 1.0]
            # 添加attributes中的数值特征
            for attr_name, attr_value in node.attributes.items():
                if isinstance(attr_value, (int, float)):
                    feature.append(attr_value)
            x.append(feature)
        
        # 转换为PyTorch张量
        x = torch.tensor(x, dtype=torch.float)
        
        # 创建PyG Data对象
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=num_nodes
        )
        
        # 保存节点ID映射，方便后续转换回原节点ID
        data.node_id_to_idx = node_id_to_idx
        data.node_ids = node_ids
        
        return data
    
    @classmethod
    def from_pyg(cls, data: Any, directed: bool = True) -> 'Graph':
        """
        从PyTorch Geometric (PyG) Data对象创建图
        
        参数:
            data: PyTorch Geometric Data对象
            directed: 是否创建有向图
            
        返回:
            Graph实例
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch库未安装，请先安装: pip install torch")
        
        graph = cls(directed=directed)
        
        # 创建节点
        num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x.shape[0]
        
        # 如果有node_ids属性，使用它作为节点ID，否则使用索引
        if hasattr(data, 'node_ids'):
            node_ids = data.node_ids
        else:
            node_ids = [f"node_{i}" for i in range(num_nodes)]
        
        for i in range(num_nodes):
            node_id = node_ids[i]
            # 获取节点特征
            if hasattr(data, 'x') and data.x is not None:
                # 使用第一个特征作为weight
                weight = data.x[i][0].item() if data.x.shape[1] > 0 else None
            else:
                weight = 1.0
            
            # 创建节点
            node = Node(
                id=node_id,
                name=node_id,
                weight=weight
            )
            graph.add_node(node)
        
        # 添加边
        edge_index = data.edge_index
        num_edges = edge_index.shape[1]
        
        # 获取边权重
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            edge_weights = data.edge_weight
        else:
            edge_weights = torch.ones(num_edges, dtype=torch.float)
        
        for i in range(num_edges):
            source_idx = edge_index[0][i].item()
            target_idx = edge_index[1][i].item()
            weight = edge_weights[i].item()
            
            # 获取源节点和目标节点
            source_node = graph.nodes[node_ids[source_idx]]
            target_node = graph.nodes[node_ids[target_idx]]
            
            # 添加边
            graph.add_edge_from_st(source_node, target_node, weight=weight)
        
        return graph

    def get_adjacency_matrix(self, nx_graph=None) -> Any:
        """获取图的邻接矩阵（igraph 高速转换；显式传入 networkx 图时保持旧行为）"""
        if nx_graph is not None:
            if not _HAS_NETWORKX:
                raise ImportError("NetworkX库未安装，无法处理 networkx 图输入")
            return nx.to_numpy_array(nx_graph)
        matrix, _ = gb.adjacency_matrix(self)
        return matrix

    def convert_nx_path_to_edges(self, paths: List[List[Dict[str, Any]]]) -> List[List[Edge]]:
        """将networkx路径转换为本图的边路径"""
        result = []
        for path in paths:
            edge_path = []
            for i in range(len(path) - 1):
                source = self.get_node(path[i]["id"])
                target = self.get_node(path[i+1]["id"])
                edge = self.get_edge(source, target)
                edge_path.append(edge)
            result.append(edge_path)
        return result

    def sort_paths_by_length(self, paths: List[List[Edge]]) -> List[List[Edge]]:
        """按路径长度排序"""
        return sorted(paths, key=lambda p: len(p))

    def sort_paths_by_function(self, paths: List[List[Edge]], func: Callable[[List[Edge]], float]) -> List[List[Edge]]:
        """按自定义函数排序路径"""
        return sorted(paths, key=lambda p: self.calculate_path_score(p, func))

    def group_paths_by_length(self, paths: List[List[Edge]]) -> Dict[int, List[List[Edge]]]:
        """按路径长度分组"""
        length_groups = defaultdict(list)
        for path in paths:
            length_groups[len(path)].append(path)
        return dict(length_groups)

    @staticmethod
    def calculate_path_score(path: List[Edge], func: Callable[[List[Edge]], float]) -> float:
        """计算路径的评分"""
        return func(path)

    def print_paths(self, paths: List[List[Edge]]) -> None:
        """打印路径信息"""
        for i, path in enumerate(paths, 1):
            edge_strs = [edge.to_str() for edge in path]
            print(f"路径 {i}: {' -> '.join(edge_strs)}")
        print(f"总路径数：{len(paths)}")

    def merge_duplicate_paths(self, paths: List[List[Edge]]) -> List[List[Edge]]:
        """合并重复的路径"""
        seen = set()
        unique_paths = []
        
        for path in paths:
            # 用边ID的字符串作为路径唯一标识
            path_id = "-".join(edge.id for edge in path)
            if path_id not in seen:
                seen.add(path_id)
                unique_paths.append(path)
        
        return unique_paths

    def is_node_satisfy_params(self, node: Node, params: Dict[str, str]) -> bool:
        """判断节点是否满足参数条件"""
        def check_attribute(edges: List[Edge], attr: str, value: str) -> bool:
            """检查节点是否有指定属性和值的边"""
            return any(edge.label == attr and edge.target.name == value for edge in edges)
        
        node_edges = self.get_neighbors(node)
        return all(check_attribute(node_edges, attr, value) for attr, value in params.items())
    
    def subgraph(self, node_ids: List[str]) -> 'Graph':
        """提取子图，包含指定的节点及其之间的边"""
        sub = Graph()
        sub.is_directed = self.is_directed
        
        # 添加指定的节点
        for node_id in node_ids:
            if node_id in self.nodes:
                sub.add_node(self.nodes[node_id])
        
        # 添加节点之间的边
        for node_id in node_ids:
            if node_id in self.adj:
                for edge in self.adj[node_id]:
                    if edge.target.id in node_ids:
                        sub.add_edge(edge)
        
        return sub
    
    def to_undirected(self) -> 'Graph':
        """将有向图转换为无向图"""
        if not self.is_directed:
            return self  # 已经是无向图
        
        undirected = Graph()
        undirected.is_directed = False
        
        # 添加所有节点
        for node in self.nodes.values():
            undirected.add_node(node)
        
        # 添加所有边及其反向边
        added_edges = set()  # 用于避免重复添加边
        
        for edges in self.adj.values():
            for edge in edges:
                # 创建边的唯一标识符
                edge_key = frozenset([edge.source.id, edge.target.id])
                if edge_key not in added_edges:
                    undirected.add_edge(edge)
                    added_edges.add(edge_key)
        
        return undirected
    
    def normalize_weights(self) -> None:
        """归一化边权重到[0, 1]区间"""
        weights = []
        for edges in self.adj.values():
            for edge in edges:
                weights.append(edge.weight)
        
        if weights:
            min_weight = min(weights)
            max_weight = max(weights)
            if max_weight > min_weight:  # 避免除以零
                for edges in self.adj.values():
                    for edge in edges:
                        edge.update(weight=(edge.weight - min_weight) / (max_weight - min_weight))
    
    def add_nodes_batch(self, nodes: List[Node]) -> None:
        """批量添加节点，提高初始化性能"""
        for node in nodes:
            self.add_node(node)
    
    def add_edges_batch(self, edges: List[Edge]) -> None:
        """批量添加边，提高初始化性能"""
        for edge in edges:
            self.add_edge(edge)
    
    def merge(self, other_graph: 'Graph') -> None:
        """合并另一个图到当前图"""
        # 添加所有节点
        for node in other_graph.nodes.values():
            if node.id not in self.nodes:
                self.add_node(node)
        
        # 添加所有边（按 source/target/label 去重，避免重复合并导致边翻倍）
        for edge in other_graph.edges.values():
            # 确保使用当前图中的节点
            source_node = self.nodes[edge.source.id] if edge.source.id in self.nodes else edge.source
            target_node = self.nodes[edge.target.id] if edge.target.id in self.nodes else edge.target

            # 去重：若 (source, target, label) 组合已存在则跳过，防止 union 合并时边被不断翻倍
            existing_edges = self.edge_lookup[source_node.id][target_node.id]
            if any(e.label == edge.label for e in existing_edges):
                continue

            # 创建新边
            new_edge = Edge(
                id=str(uuid.uuid1()),  # 生成新ID避免冲突
                source=source_node,
                target=target_node,
                weight=edge.weight,
                label=edge.label,
                attributes=edge.attributes.copy()
            )
            self.add_edge(new_edge)
    
    def to_dict(self) -> Dict[str, Any]:
        """将图转换为字典表示"""
        graph_dict = {
            "nodes": [],
            "edges": [],
            "is_directed": self.is_directed
        }
        
        # 添加节点信息
        for node in self.nodes.values():
            graph_dict["nodes"].append(node.convert_dict())
        
        # 添加边信息
        for edge in self.edges.values():
            graph_dict["edges"].append(edge.convert_dict())
        
        return graph_dict
    
    @classmethod
    def from_dict(cls, graph_dict: Dict[str, Any]) -> 'Graph':
        """从字典创建图"""
        graph = cls()
        graph.is_directed = graph_dict.get("is_directed", True)
        
        # 首先创建所有节点
        node_map = {}
        for node_data in graph_dict.get("nodes", []):
            node = Node(
                id=node_data["id"],
                name=node_data["name"],
                class_=node_data.get("type"),
                weight=node_data.get("weight"),
                timestamp=node_data.get("timestamp", time.time()),
                version=node_data.get("version", 1),
                attributes=node_data.get("attributes", {})
            )
            graph.add_node(node)
            node_map[node.id] = node
        
        # 然后创建所有边
        for edge_data in graph_dict.get("edges", []):
            if edge_data["source_id"] in node_map and edge_data["target_id"] in node_map:
                edge = Edge(
                    id=edge_data["id"],
                    source=node_map[edge_data["source_id"]],
                    target=node_map[edge_data["target_id"]],
                    weight=edge_data["weight"],
                    label=edge_data.get("label"),
                    timestamp=edge_data.get("timestamp", time.time()),
                    version=edge_data.get("version", 1),
                    attributes=edge_data.get("attributes", {})
                )
                graph.add_edge(edge)
        
        return graph
    
    def save(self, file_path: str) -> None:
        """保存图到文件（支持JSON和pickle格式）"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        elif file_ext == '.pkl' or file_ext == '.pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}，支持 .json 和 .pkl/.pickle")
    
    @classmethod
    def load(cls, file_path: str) -> 'Graph':
        """从文件加载图"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_dict = json.load(f)
            return cls.from_dict(graph_dict)
        elif file_ext == '.pkl' or file_ext == '.pickle':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}，支持 .json 和 .pkl/.pickle")
    
    def save_to_graphml(self, file_path: str) -> None:
        """保存图为GraphML格式（igraph 原生读写，无需 networkx）"""
        gb.save_graphml(self, file_path)

    @classmethod
    def load_from_graphml(cls, file_path: str) -> 'Graph':
        """从GraphML格式加载图（igraph 原生读写，无需 networkx）"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        return gb.load_graphml(file_path)

############ 图指标计算 ############
class GraphMetrics:
    """
    图指标计算工具类，提供各种图的全局和局部统计指标
    包括度分布、中心性、聚集系数等
    """
    
    def degree_distribution(self, graph: 'Graph') -> Dict[int, int]:
        """
        计算图的度分布
        
        参数:
            graph: Graph实例
        
        返回:
            度值到节点数量的映射字典
        """
        degrees = {}
        for node_id, node in graph.nodes.items():
            degree = len(graph.get_neighbors(node))
            degrees[degree] = degrees.get(degree, 0) + 1
        return degrees
    
    def average_degree(self, graph: 'Graph') -> float:
        """
        计算图的平均度
        
        参数:
            graph: Graph实例
        
        返回:
            平均度值
        """
        if not graph.nodes:
            return 0.0
        total_degree = 0
        for node_id, node in graph.nodes.items():
            total_degree += len(graph.get_neighbors(node))
        return total_degree / len(graph.nodes)
    
    def clustering_coefficient(self, graph: 'Graph', node_id: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        计算图的聚集系数（整体或单个节点）

        高性能路径：igraph transitivity（C 核心）。有向图取无向视图（业界通行口径）。

        参数:
            graph: Graph实例
            node_id: 节点ID，None表示计算全局聚集系数

        返回:
            聚集系数值或节点ID到聚集系数的映射
        """
        return gb.clustering_coefficient(graph, node_id)
    
    def betweenness_centrality(self, graph: 'Graph', normalized: bool = True) -> Dict[str, float]:
        """
        计算图中各节点的介数中心性

        高性能路径：igraph Brandes 算法（C 核心，O(VE)），替代旧版纯 Python 双重循环。

        参数:
            graph: Graph实例
            normalized: 是否归一化

        返回:
            节点ID到介数中心性的映射
        """
        return gb.betweenness_centrality(graph, normalized=normalized)
    
    def eigenvector_centrality(self, graph: 'Graph', max_iter: int = 100, tol: float = 1.0e-6) -> Dict[str, float]:
        """
        计算图中各节点的特征向量中心性

        高性能路径：无向图走 igraph ARPACK（C 核心）；有向图走 SciPy 向量化幂迭代，
        均替代旧版纯 Python 逐边循环。

        参数:
            graph: Graph实例
            max_iter: 最大迭代次数
            tol: 收敛阈值

        返回:
            节点ID到特征向量中心性的映射
        """
        return gb.eigenvector_centrality(graph, max_iter=max_iter, tol=tol)

############ 图可视化 ############
class GraphVisualizer:
    """
    图可视化工具类，基于NetworkX和PyVis提供交互式可视化
    支持高亮路径、社区、中心节点等功能
    """
    
    def __init__(self):
        # 尝试导入必要的库
        try:
            import networkx as nx
            self.nx = nx
            self.has_nx = True
        except ImportError:
            self.nx = None
            self.has_nx = False
        
        try:
            from pyvis.network import Network
            self.Network = Network
            self.has_pyvis = True
        except ImportError:
            self.Network = None
            self.has_pyvis = False
    
    def to_networkx_graph(self, graph: 'Graph') -> 'nx.Graph':
        """
        将自定义Graph对象转换为NetworkX图
        
        参数:
            graph: 自定义Graph实例
        
        返回:
            NetworkX图对象
        """
        if not self.has_nx:
            raise ImportError("NetworkX库未安装，请先安装: pip install networkx")
        
        # 根据图的类型创建NetworkX图
        if graph.is_directed:
            nx_graph = self.nx.DiGraph()
        else:
            nx_graph = self.nx.Graph()
        
        # 添加节点
        for node_id, node in graph.nodes.items():
            # 将节点属性添加到NetworkX节点
            node_attrs = {}
            if hasattr(node, 'name') and node.name:
                node_attrs['label'] = node.name
            if hasattr(node, 'attributes') and node.attributes:
                node_attrs.update(node.attributes)
            # 添加节点ID作为标签的一部分
            if 'label' not in node_attrs:
                node_attrs['label'] = node_id
            nx_graph.add_node(node_id, **node_attrs)
        
        # 添加边
        for node_id, node in graph.nodes.items():
            for edge in graph.get_neighbors(node):
                # 边的属性
                edge_attrs = {}
                if hasattr(edge, 'weight') and edge.weight is not None:
                    edge_attrs['weight'] = edge.weight
                    edge_attrs['title'] = f"Weight: {edge.weight}"
                if hasattr(edge, 'label') and edge.label:
                    edge_attrs['label'] = edge.label
                if hasattr(edge, 'attributes') and edge.attributes:
                    edge_attrs.update(edge.attributes)
                
                # 添加边
                nx_graph.add_edge(node_id, edge.target.id, **edge_attrs)
        
        return nx_graph
    
    def visualize(self, graph: 'Graph', filename: str = 'graph_visualization.html', 
                  height: str = '600px', width: str = '800px', 
                  notebook: bool = False, bgcolor: str = '#ffffff', 
                  font_color: str = '#000000', 
                  highlight_paths: Optional[List[List[str]]] = None,
                  highlight_nodes: Optional[List[str]] = None,
                  community_assignments: Optional[Dict[str, int]] = None,
                  **kwargs):
        """
        使用PyVis可视化图
        
        参数:
            graph: 自定义Graph实例
            filename: 输出HTML文件名
            height: 图高度
            width: 图宽度
            notebook: 是否在Jupyter Notebook中显示
            bgcolor: 背景色
            font_color: 字体颜色
            highlight_paths: 要高亮显示的路径列表
            highlight_nodes: 要高亮显示的节点列表
            community_assignments: 节点到社区ID的映射，用于着色
            **kwargs: 传递给PyVis Network的其他参数
        """
        if not self.has_pyvis:
            raise ImportError("PyVis库未安装，请先安装: pip install pyvis")
        
        # 转换为NetworkX图
        nx_graph = self.to_networkx_graph(graph)
        
        try:
            # 创建PyVis网络
            net = self.Network(height=height, width=width, bgcolor=bgcolor, 
                              font_color=font_color, notebook=notebook, **kwargs)
            
            # 从NetworkX图加载数据
            net.from_nx(nx_graph)
            
            # 高亮路径
            if highlight_paths:
                # 为路径上的边设置不同的颜色
                edge_colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff']
                for path_idx, path in enumerate(highlight_paths):
                    color = edge_colors[path_idx % len(edge_colors)]
                    for i in range(len(path) - 1):
                        # 查找并高亮路径上的边
                        for edge in net.edges:
                            if (edge['from'] == path[i] and edge['to'] == path[i+1]) or \
                               (edge['to'] == path[i] and edge['from'] == path[i+1]):
                                edge['color'] = color
                                edge['width'] = 3
            
            # 高亮节点
            if highlight_nodes:
                for node_id in highlight_nodes:
                    for node in net.nodes:
                        if node['id'] == node_id:
                            node['color'] = '#ff0000'
                            node['size'] = 20
            
            # 按社区着色
            if community_assignments:
                # 使用不同的颜色为每个社区
                import matplotlib.pyplot as plt
                colors = plt.cm.tab10.colors
                
                for node_id, community in community_assignments.items():
                    for node in net.nodes:
                        if node['id'] == node_id:
                            color_idx = community % len(colors)
                            r, g, b = colors[color_idx]
                            node['color'] = f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)'
            
            # 保存并返回可视化
            net.show_buttons(filter_=['physics'])
            net.show(filename)
            print(f"可视化已保存到: {filename}")
            return net
        except Exception as e:
            # 处理模板渲染错误的备用方法
            print(f"注意: {e}. 尝试备用方法...")
            # 直接生成HTML内容并写入文件
            
            # 基本的HTML模板
            html_template = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Graph Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        #mynetwork {
            width: {{ width }};
            height: {{ height }};
            background-color: {{ bgcolor }};
        }
    </style>
</head>
<body>
    <div id="mynetwork"></div>
    <script type="text/javascript">
        var nodes = new vis.DataSet({{ nodes_json }});
        var edges = new vis.DataSet({{ edges_json }});
        
        var container = document.getElementById('mynetwork');
        
        var data = {
            nodes: nodes,
            edges: edges
        };
        
        var options = {
            nodes: {
                font: {
                    color: '{{ font_color }}'
                }
            },
            edges: {
                smooth: {
                    forceDirection: 'horizontal',
                    roundness: 0.4
                }
            }
        };
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
'''
            
            # 准备节点和边的数据
            import json
            nodes_data = []
            for node_id, node in graph.nodes.items():
                node_info = {'id': node_id, 'label': node.name if hasattr(node, 'name') and node.name else node_id}
                # 应用高亮和社区着色（如果有）
                if highlight_nodes and node_id in highlight_nodes:
                    node_info['color'] = '#ff0000'
                    node_info['size'] = 20
                elif community_assignments and node_id in community_assignments:
                    import matplotlib.pyplot as plt
                    colors = plt.cm.tab10.colors
                    color_idx = community_assignments[node_id] % len(colors)
                    r, g, b = colors[color_idx]
                    node_info['color'] = f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)'
                nodes_data.append(node_info)
            
            edges_data = []
            for node_id, node in graph.nodes.items():
                for edge in graph.get_neighbors(node):
                    edge_info = {'from': node_id, 'to': edge.target.id}
                    if hasattr(edge, 'weight') and edge.weight is not None:
                        edge_info['weight'] = edge.weight
                    if hasattr(edge, 'label') and edge.label:
                        edge_info['label'] = edge.label
                    # 应用高亮路径（如果有）
                    if highlight_paths:
                        for path in highlight_paths:
                            for i in range(len(path) - 1):
                                if (edge_info['from'] == path[i] and edge_info['to'] == path[i+1]) or \
                                   (edge_info['to'] == path[i] and edge_info['from'] == path[i+1]):
                                    edge_info['color'] = '#ff0000'
                                    edge_info['width'] = 3
                    edges_data.append(edge_info)
            
            # 写入HTML文件
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_template.format(
                    width=width,
                    height=height,
                    bgcolor=bgcolor,
                    font_color=font_color,
                    nodes_json=json.dumps(nodes_data, ensure_ascii=False),
                    edges_json=json.dumps(edges_data, ensure_ascii=False)
                ))
            
            print(f"可视化已通过备用方法保存到: {filename}")
            return None
    
    def simple_plot(self, graph: 'Graph', title: str = 'Graph Visualization', save_path: Optional[str] = None, **kwargs):
        """
        使用NetworkX和Matplotlib创建简单的静态可视化
        
        参数:
            graph: 自定义Graph实例
            title: 图标题
            save_path: 可选的保存路径，如果提供则保存图片
            **kwargs: 其他参数，兼容性保留
        """
        if not self.has_nx:
            raise ImportError("NetworkX库未安装，请先安装: pip install networkx")
        
        try:
            import matplotlib.pyplot as plt
            # 设置中文字体支持
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'SimSun']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        except ImportError:
            raise ImportError("Matplotlib库未安装，请先安装: pip install matplotlib")
        
        # 转换为NetworkX图
        nx_graph = self.to_networkx_graph(graph)
        
        # 绘制图
        plt.figure(figsize=(10, 8))
        
        # 布局选择
        pos = self.nx.spring_layout(nx_graph)
        
        # 绘制节点
        self.nx.draw_networkx_nodes(nx_graph, pos, node_size=300, node_color='lightblue')
        
        # 绘制边
        self.nx.draw_networkx_edges(nx_graph, pos, edge_color='gray')
        
        # 绘制标签 - 使用节点名称（如果有）作为标签
        labels = {}
        for node_id in nx_graph.nodes():
            if 'label' in nx_graph.nodes[node_id] and nx_graph.nodes[node_id]['label']:
                labels[node_id] = nx_graph.nodes[node_id]['label']
            else:
                labels[node_id] = node_id
        self.nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_size=10, font_family='sans-serif')
        
        # 添加边权重标签（如果有）
        edge_labels = {}
        for u, v, data in nx_graph.edges(data=True):
            if 'weight' in data and data['weight'] != 1:
                edge_labels[(u, v)] = data['weight']
        if edge_labels:
            self.nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图已保存到: {save_path}")
            plt.close()
        else:
            plt.show()

############ 图工具函数 ############
class GraphUtils:
    """
    图工具函数类，提供各种图处理的通用辅助功能
    包括节点ID映射、权重归一化、拓扑排序等
    """
    
    def remap_node_ids(self, graph: 'Graph', mapping: Optional[Dict[str, str]] = None) -> Tuple['Graph', Dict[str, str]]:
        """
        重新映射图中节点的ID
        
        参数:
            graph: 原始Graph实例
            mapping: 可选的节点ID映射字典，None表示自动生成新ID
        
        返回:
            新的Graph实例和ID映射字典
        """
        # 如果没有提供映射，自动生成
        if mapping is None:
            mapping = {}
            for i, node_id in enumerate(graph.nodes.keys()):
                mapping[node_id] = f"node_{i}"
        
        # 创建新图
        new_graph = Graph(directed=graph.is_directed)
        
        # 存储旧节点到新节点的映射
        node_mapping = {}
        
        # 添加重映射后的节点
        for old_id, old_node in graph.nodes.items():
            if old_id in mapping:
                new_id = mapping[old_id]
                # 创建新节点，复制原始节点的属性
                # 复制原始节点的名称，如果没有则使用ID作为名称
                node_name = getattr(old_node, 'name', new_id)
                new_node = Node(id=new_id, name=node_name)
                # 复制其他属性
                for attr in ['name', 'timestamp', 'version', 'attributes']:
                    if hasattr(old_node, attr):
                        setattr(new_node, attr, getattr(old_node, attr))
                new_graph.add_node(new_node)
                node_mapping[old_id] = new_node
        
        # 添加重映射后的边
        for old_id, old_node in graph.nodes.items():
            if old_id in mapping:
                for old_edge in graph.get_neighbors(old_node):
                    target_old_id = old_edge.target.id
                    if target_old_id in mapping:
                        # 创建新边
                        # 生成新的边ID
                        target_new_id = mapping[target_old_id]
                        new_edge_id = f"{new_id}_{target_new_id}"
                        # 获取原边的权重，如果没有则默认为1.0
                        weight = getattr(old_edge, 'weight', 1.0)
                        # 创建新边，提供必需的id和weight参数
                        new_edge = Edge(
                            id=new_edge_id,
                            source=node_mapping[old_id],
                            target=node_mapping[target_old_id],
                            weight=weight
                        )
                        # 复制边的属性
                        for attr in ['weight', 'label', 'timestamp', 'version', 'attributes']:
                            if hasattr(old_edge, attr):
                                setattr(new_edge, attr, getattr(old_edge, attr))
                        new_graph.add_edge(new_edge)
        
        return new_graph, mapping
    
    def normalize_weights(self, graph: 'Graph', method: str = 'min_max') -> 'Graph':
        """
        归一化图中所有边的权重
        
        参数:
            graph: 原始Graph实例
            method: 归一化方法，支持'min_max'、'z_score'、'sum'、'log'
        
        返回:
            权重归一化后的新Graph实例
        """
        # 创建新图（深拷贝）
        import copy
        new_graph = Graph(directed=graph.is_directed)
        
        # 复制节点
        node_mapping = {}
        for node_id, node in graph.nodes.items():
            # 深拷贝节点
            new_node = copy.deepcopy(node)
            new_graph.add_node(new_node)
            node_mapping[node_id] = new_node
        
        # 收集所有边的权重
        weights = []
        for node_id, node in graph.nodes.items():
            for edge in graph.get_neighbors(node):
                if hasattr(edge, 'weight') and edge.weight is not None:
                    weights.append(edge.weight)
        
        # 如果没有权重或只有一个权重，直接返回深拷贝的图
        if not weights or len(weights) <= 1:
            # 复制边
            for node_id, node in graph.nodes.items():
                for edge in graph.get_neighbors(node):
                    new_edge = copy.deepcopy(edge)
                    new_edge.source = node_mapping[edge.source.id]
                    new_edge.target = node_mapping[edge.target.id]
                    new_graph.add_edge(new_edge)
            return new_graph
        
        # 计算归一化参数
        min_weight = min(weights)
        max_weight = max(weights)
        mean_weight = sum(weights) / len(weights)
        std_weight = (sum((w - mean_weight) ** 2 for w in weights) / len(weights)) ** 0.5
        sum_weights = sum(weights)
        
        # 复制并归一化边
        for node_id, node in graph.nodes.items():
            for edge in graph.get_neighbors(node):
                new_edge = copy.deepcopy(edge)
                new_edge.source = node_mapping[edge.source.id]
                new_edge.target = node_mapping[edge.target.id]
                
                # 归一化权重
                if hasattr(new_edge, 'weight') and new_edge.weight is not None:
                    w = new_edge.weight
                    if method == 'min_max' and max_weight > min_weight:
                        new_edge.weight = (w - min_weight) / (max_weight - min_weight)
                    elif method == 'z_score' and std_weight > 0:
                        new_edge.weight = (w - mean_weight) / std_weight
                    elif method == 'sum' and sum_weights > 0:
                        new_edge.weight = w / sum_weights
                    elif method == 'log':
                        import math
                        new_edge.weight = math.log(w + 1) if w >= 0 else w
                
                new_graph.add_edge(new_edge)
        
        return new_graph
    
    def topological_sort(self, graph: 'Graph') -> List[str]:
        """
        对有向无环图(DAG)进行拓扑排序
        
        参数:
            graph: 有向图实例
        
        返回:
            节点ID的拓扑排序列表
        
        抛出:
            ValueError: 如果图中存在环
        """
        if not graph.is_directed:
            raise ValueError("拓扑排序只适用于有向图")
        
        # 计算每个节点的入度
        in_degree = {node_id: 0 for node_id in graph.nodes}
        for node_id, node in graph.nodes.items():
            for edge in graph.get_neighbors(node):
                in_degree[edge.target.id] += 1
        
        # 初始化队列，将所有入度为0的节点加入队列
        from collections import deque
        queue = deque()
        for node_id in graph.nodes:
            if in_degree[node_id] == 0:
                queue.append(node_id)
        
        # 执行拓扑排序
        result = []
        while queue:
            current_id = queue.popleft()
            result.append(current_id)
            
            # 减少所有邻居节点的入度
            current_node = graph.nodes[current_id]
            for edge in graph.get_neighbors(current_node):
                neighbor_id = edge.target.id
                in_degree[neighbor_id] -= 1
                # 如果入度变为0，加入队列
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)
        
        # 检查是否存在环
        if len(result) != len(graph.nodes):
            raise ValueError("图中存在环，无法进行拓扑排序")
        
        return result
    
    def transpose_graph(self, graph: 'Graph') -> 'Graph':
        """
        计算有向图的转置（所有边的方向反转）
        
        参数:
            graph: 有向图实例
        
        返回:
            转置后的新图实例
        """
        if not graph.is_directed:
            # 对于无向图，转置等同于原图
            import copy
            new_graph = Graph(directed=False)
            # 复制节点
            node_mapping = {}
            for node_id, node in graph.nodes.items():
                new_node = copy.deepcopy(node)
                new_graph.add_node(new_node)
                node_mapping[node_id] = new_node
            # 复制边
            for node_id, node in graph.nodes.items():
                for edge in graph.get_neighbors(node):
                    new_edge = copy.deepcopy(edge)
                    new_edge.source = node_mapping[edge.source.id]
                    new_edge.target = node_mapping[edge.target.id]
                    new_graph.add_edge(new_edge)
            return new_graph
        
        # 创建转置图
        transposed = Graph(directed=True)
        
        # 复制节点
        node_mapping = {}
        for node_id, node in graph.nodes.items():
            import copy
            new_node = copy.deepcopy(node)
            transposed.add_node(new_node)
            node_mapping[node_id] = new_node
        
        # 添加反转方向的边
        for node_id, node in graph.nodes.items():
            for edge in graph.get_neighbors(node):
                # 创建反向边
                reverse_edge = Edge(
                    source=node_mapping[edge.target.id],
                    target=node_mapping[node_id]
                )
                # 复制边的属性
                for attr in ['weight', 'label', 'timestamp', 'version', 'attributes']:
                    if hasattr(edge, attr):
                        setattr(reverse_edge, attr, getattr(edge, attr))
                transposed.add_edge(reverse_edge)
        
        return transposed
    
    def sample_subgraph(self, graph: 'Graph', sample_size: int, method: str = 'random') -> 'Graph':
        """
        从原图中采样子图
        
        参数:
            graph: 原始图实例
            sample_size: 采样的节点数量
            method: 采样方法，支持'random'(随机)、'bfs'(广度优先)、'dfs'(深度优先)
        
        返回:
            采样得到的子图实例
        """
        if sample_size >= len(graph.nodes):
            # 如果采样大小大于等于原图节点数，返回原图的深拷贝
            import copy
            new_graph = Graph(directed=graph.is_directed)
            # 复制节点
            node_mapping = {}
            for node_id, node in graph.nodes.items():
                new_node = copy.deepcopy(node)
                new_graph.add_node(new_node)
                node_mapping[node_id] = new_node
            # 复制边
            for node_id, node in graph.nodes.items():
                for edge in graph.get_neighbors(node):
                    new_edge = copy.deepcopy(edge)
                    new_edge.source = node_mapping[edge.source.id]
                    new_edge.target = node_mapping[edge.target.id]
                    new_graph.add_edge(new_edge)
            return new_graph
        
        # 选择采样的节点
        sample_nodes = set()
        if method == 'random':
            # 随机采样
            import random
            sample_nodes = set(random.sample(list(graph.nodes.keys()), sample_size))
        elif method == 'bfs' or method == 'dfs':
            # BFS或DFS采样
            import random
            # 随机选择起始节点
            start_node_id = random.choice(list(graph.nodes.keys()))
            sample_nodes.add(start_node_id)
            
            # 使用队列或栈进行遍历
            from collections import deque
            if method == 'bfs':
                traversal_queue = deque([start_node_id])
            else:  # dfs
                traversal_queue = [start_node_id]
            
            while traversal_queue and len(sample_nodes) < sample_size:
                if method == 'bfs':
                    current_id = traversal_queue.popleft()
                else:  # dfs
                    current_id = traversal_queue.pop()
                
                # 获取所有邻居
                current_node = graph.nodes[current_id]
                for edge in graph.get_neighbors(current_node):
                    neighbor_id = edge.target.id
                    if neighbor_id not in sample_nodes:
                        sample_nodes.add(neighbor_id)
                        traversal_queue.append(neighbor_id)
                        if len(sample_nodes) >= sample_size:
                            break
        
        # 创建子图
        subgraph = graph.subgraph(sample_nodes)
        return subgraph
    
    def get_node_similarity(self, graph: 'Graph', node1_id: str, node2_id: str, method: str = 'jaccard') -> float:
        """
        计算两个节点的相似度
        
        参数:
            graph: 图实例
            node1_id: 第一个节点ID
            node2_id: 第二个节点ID
            method: 相似度计算方法，支持'jaccard'(杰卡德系数)、'cosine'(余弦相似度)、'adamic_adar'(Adamic-Adar指数)
        
        返回:
            节点相似度值（0-1之间）
        """
        if node1_id not in graph.nodes or node2_id not in graph.nodes:
            raise ValueError(f"节点 {node1_id} 或 {node2_id} 不在图中")
        
        # 获取两个节点的邻居集合
        node1 = graph.nodes[node1_id]
        node2 = graph.nodes[node2_id]
        
        neighbors1 = {edge.target.id for edge in graph.get_neighbors(node1)}
        neighbors2 = {edge.target.id for edge in graph.get_neighbors(node2)}
        
        # 计算相似度
        if method == 'jaccard':
            # 杰卡德系数: |A ∩ B| / |A ∪ B|
            intersection = len(neighbors1.intersection(neighbors2))
            union = len(neighbors1.union(neighbors2))
            return intersection / union if union > 0 else 0.0
        
        elif method == 'cosine':
            # 余弦相似度: |A ∩ B| / (√|A| * √|B|)
            intersection = len(neighbors1.intersection(neighbors2))
            denominator = (len(neighbors1) ** 0.5) * (len(neighbors2) ** 0.5)
            return intersection / denominator if denominator > 0 else 0.0
        
        elif method == 'adamic_adar':
            # Adamic-Adar指数: Σ(1 / log(|N(u)|)) 对于所有u ∈ N(node1) ∩ N(node2)
            # N(u)表示u的邻居集合
            common_neighbors = neighbors1.intersection(neighbors2)
            score = 0.0
            import math
            
            for common_node_id in common_neighbors:
                common_node = graph.nodes[common_node_id]
                common_neighbors_count = len([edge for edge in graph.get_neighbors(common_node)])
                if common_neighbors_count > 1:  # 避免log(1)=0
                    score += 1 / math.log(common_neighbors_count)
            
            # 归一化到0-1范围
            max_possible = len(common_neighbors) / math.log(2)  # 假设最小邻居数为2
            return score / max_possible if max_possible > 0 else 0.0
        
        else:
            raise ValueError(f"不支持的相似度计算方法: {method}")

############ 图输入输出 ############
class GraphIO:
    """
    图输入输出工具类，支持多种格式的图数据导入导出
    包括JSON、GraphML、CSV、pickle等格式
    """
    
    def __init__(self):
        # 尝试导入必要的库
        try:
            import networkx as nx
            self.nx = nx
            self.has_nx = True
        except ImportError:
            self.nx = None
            self.has_nx = False
    
    def save_to_json(self, graph: 'Graph', filepath: str):
        """
        将图保存为JSON格式
        
        参数:
            graph: Graph实例
            filepath: 输出文件路径
        """
        graph_dict = graph.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_dict, f, ensure_ascii=False, indent=2)
        
        print(f"图已保存到JSON文件: {filepath}")
    
    def load_from_json(self, filepath: str, directed: bool = False) -> 'Graph':
        """
        从JSON文件加载图
        
        参数:
            filepath: 输入文件路径
            directed: 是否创建有向图（注意：会被JSON中的is_directed覆盖，除非JSON中不存在）
        
        返回:
            Graph实例
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_dict = json.load(f)
        
        # 从字典创建图
        graph = Graph.from_dict(graph_dict)
        
        # 如果JSON中没有指定is_directed，则使用参数值
        if "is_directed" not in graph_dict:
            graph.is_directed = directed
        
        print(f"从JSON文件加载图: {filepath}")
        return graph
    
    def save_to_pickle(self, graph: 'Graph', filepath: str):
        """
        将图保存为pickle格式（保留所有Python对象属性）
        
        参数:
            graph: Graph实例
            filepath: 输出文件路径
        """
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
        
        print(f"图已保存到pickle文件: {filepath}")
    
    def load_from_pickle(self, filepath: str) -> 'Graph':
        """
        从pickle文件加载图
        
        参数:
            filepath: 输入文件路径
        
        返回:
            Graph实例
        """
        with open(filepath, 'rb') as f:
            graph = pickle.load(f)
        
        print(f"从pickle文件加载图: {filepath}")
        return graph
    
    def save_to_graphml(self, graph: 'Graph', filepath: str):
        """
        将图保存为GraphML格式

        高性能路径：igraph 原生写入（无需 networkx）。

        参数:
            graph: Graph实例
            filepath: 输出文件路径
        """
        gb.save_graphml(graph, filepath)
        print(f"图已保存到GraphML文件: {filepath}")

    def load_from_graphml(self, filepath: str) -> 'Graph':
        """
        从GraphML文件加载图

        高性能路径：igraph 原生读取（无需 networkx）。

        参数:
            filepath: 输入文件路径

        返回:
            Graph实例
        """
        try:
            graph = gb.load_graphml(filepath)
        except Exception as e:
            raise ValueError(f"无法从GraphML文件加载图: {str(e)}")

        print(f"从GraphML文件加载图: {filepath}")
        return graph
    
    def save_to_csv(self, graph: 'Graph', nodes_file: str, edges_file: str):
        """
        将图保存为CSV格式（节点表和边表）
        
        参数:
            graph: Graph实例
            nodes_file: 节点CSV文件路径
            edges_file: 边CSV文件路径
        """
        import csv
        
        # 保存节点表
        with open(nodes_file, 'w', newline='', encoding='utf-8') as f:
            # 获取所有可能的节点属性键
            all_attrs = set()
            for node in graph.nodes.values():
                if hasattr(node, 'attributes') and node.attributes:
                    all_attrs.update(node.attributes.keys())
            
            # 创建CSV表头
            fieldnames = ['id', 'name'] + sorted(all_attrs)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # 写入节点数据
            for node in graph.nodes.values():
                row = {'id': node.id}
                if hasattr(node, 'name') and node.name:
                    row['name'] = node.name
                if hasattr(node, 'attributes') and node.attributes:
                    row.update(node.attributes)
                writer.writerow(row)
        
        # 保存边表
        with open(edges_file, 'w', newline='', encoding='utf-8') as f:
            # 获取所有可能的边属性键
            all_attrs = set()
            for node in graph.nodes.values():
                for edge in graph.get_neighbors(node):
                    if hasattr(edge, 'attributes') and edge.attributes:
                        all_attrs.update(edge.attributes.keys())
            
            # 创建CSV表头
            fieldnames = ['source', 'target', 'weight', 'label'] + sorted(all_attrs)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # 写入边数据
            written_edges = set()
            for node in graph.nodes.values():
                for edge in graph.get_neighbors(node):
                    # 避免重复写入无向图的边
                    edge_key = (edge.source.id, edge.target.id)
                    if not graph.is_directed and (edge.target.id, edge.source.id) in written_edges:
                        continue
                    written_edges.add(edge_key)
                    
                    row = {
                        'source': edge.source.id,
                        'target': edge.target.id
                    }
                    if hasattr(edge, 'weight') and edge.weight is not None:
                        row['weight'] = edge.weight
                    if hasattr(edge, 'label') and edge.label:
                        row['label'] = edge.label
                    if hasattr(edge, 'attributes') and edge.attributes:
                        row.update(edge.attributes)
                    writer.writerow(row)
        
        print(f"图已保存到CSV文件: {nodes_file} 和 {edges_file}")
    
    def load_from_csv(self, nodes_file: str, edges_file: str, directed: bool = False) -> 'Graph':
        """
        从CSV文件（节点表和边表）加载图
        
        参数:
            nodes_file: 节点CSV文件路径
            edges_file: 边CSV文件路径
            directed: 是否创建有向图
        
        返回:
            Graph实例
        """
        import csv
        
        # 创建图实例
        graph = Graph(directed=directed)
        
        # 加载节点
        if os.path.exists(nodes_file):
            with open(nodes_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 创建节点
                    node_id = row.pop('id')
                    # 获取节点名称，如果没有则使用ID作为名称
                    node_name = row.pop('name') if 'name' in row and row['name'] else node_id
                    # 创建节点时直接提供name参数
                    node = Node(id=node_id, name=node_name)
                    # 设置其他属性
                    if row:
                        node.attributes = row
                    # 添加到图
                    graph.add_node(node)
        
        # 加载边
        if os.path.exists(edges_file):
            with open(edges_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 获取源节点和目标节点
                    source_id = row.pop('source')
                    target_id = row.pop('target')
                    
                    # 检查节点是否存在
                    if source_id not in graph.nodes or target_id not in graph.nodes:
                        continue
                    
                    # 生成边ID
                    edge_id = f"{source_id}_{target_id}"
                    
                    # 设置权重
                    weight = 1.0
                    if 'weight' in row and row['weight']:
                        try:
                            weight = float(row['weight'])
                        except:
                            weight = 1.0
                        row.pop('weight')
                    
                    # 创建边，提供必需的id和weight参数
                    edge = Edge(id=edge_id, source=graph.nodes[source_id], target=graph.nodes[target_id], weight=weight)
                    
                    # 设置标签
                    if 'label' in row and row['label']:
                        edge.label = row['label']
                        row.pop('label')
                    
                    # 设置其他属性
                    if row:
                        edge.attributes = row
                    
                    # 添加到图
                    graph.add_edge(edge)
        
        print(f"从CSV文件加载图: {nodes_file} 和 {edges_file}")
        return graph

############ 算法 ############
class GraphSearch:
    """图搜索算法工具类，提供BFS、路径查询、环检测等图算法功能"""
    
    # 类级别的缓存，存储格式: {(graph_id, version, start_id): result_dict}
    _dijkstra_cache = {}

    def __init__(self):
        pass
    
    def cached_dijkstra(self, graph: 'Graph', start_node_id: str, end_node_id: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        带缓存的Dijkstra算法，基于图的版本号进行缓存验证
        
        参数:
            graph: Graph实例
            start_node_id: 起始节点ID
            end_node_id: 目标节点ID，如果为None则返回到所有节点的路径
            
        返回:
            如果指定end_node_id，返回该节点的路径信息；否则返回所有节点的路径信息字典
        """
        # 生成缓存键：(图对象内存地址, 图版本号, 起始节点ID)
        cache_key = (id(graph), graph.version, start_node_id)
        
        # 检查缓存
        if cache_key in self._dijkstra_cache:
            all_results = self._dijkstra_cache[cache_key]
            if end_node_id:
                return all_results.get(end_node_id)
            return all_results
            
        # 清理该图的旧版本缓存
        keys_to_remove = [k for k in self._dijkstra_cache.keys() if k[0] == id(graph) and k[1] < graph.version]
        for k in keys_to_remove:
            del self._dijkstra_cache[k]
            
        # 简单的缓存容量控制（FIFO）
        if len(self._dijkstra_cache) > 100:
            # 移除最早添加的项
            first_key = next(iter(self._dijkstra_cache))
            del self._dijkstra_cache[first_key]
            
        # 计算并存入缓存
        all_results = self.dijkstra(graph, start_node_id)
        self._dijkstra_cache[cache_key] = all_results
        
        if end_node_id:
            return all_results.get(end_node_id)
        return all_results

    def bfs_traversal(self, graph: 'Graph', source: Union[Node, str]) -> Dict[str, Any]:
        """
        广度优先搜索(BFS)遍历图，返回遍历结果
        
        参数:
            graph: 待遍历的图（Graph实例）
            source: 起始节点（Node对象或节点ID）
        
        返回:
            包含遍历信息的字典，结构为:
            {
                "sequence": 遍历节点的顺序（Node对象列表）,
                "distances": 各节点到源节点的距离（节点ID到距离的映射）,
                "predecessors": 各节点的前驱节点ID（节点ID到前驱ID的映射）
            }
        """
        # 统一处理源节点为ID
        source_id = source.id if isinstance(source, Node) else source
        if source_id not in graph.nodes:
            raise ValueError(f"源节点 {source_id} 不在图中")

        # 初始化BFS状态（每次调用重置，避免状态污染）
        visited = {source_id: True}
        queue = [source_id]
        distances = {source_id: 0}
        predecessors = {}
        traversal_sequence = [graph.nodes[source_id]]

        while queue:
            current_id = queue.pop(0)
            current_node = graph.nodes[current_id]

            # 遍历当前节点的所有邻接边
            for edge in graph.get_neighbors(current_node):
                neighbor_id = edge.target.id
                if neighbor_id not in visited:
                    visited[neighbor_id] = True
                    queue.append(neighbor_id)
                    distances[neighbor_id] = distances[current_id] + 1
                    predecessors[neighbor_id] = current_id
                    traversal_sequence.append(edge.target)

        return {
            "sequence": traversal_sequence,
            "distances": distances,
            "predecessors": predecessors
        }

    def dfs_traversal(self, graph: 'Graph', source: Union[Node, str]) -> Dict[str, Any]:
        """
        深度优先搜索(DFS)遍历图（使用栈的非递归实现），返回遍历结果
        
        参数:
            graph: 待遍历的图（Graph实例）
            source: 起始节点（Node对象或节点ID）
        
        返回:
            包含遍历信息的字典，结构为:
            {
                "sequence": 遍历节点的顺序（Node对象列表）,
                "distances": 各节点到源节点的距离（节点ID到距离的映射）,
                "predecessors": 各节点的前驱节点ID（节点ID到前驱ID的映射）
            }
        """
        # 统一处理源节点为ID
        source_id = source.id if isinstance(source, Node) else source
        if source_id not in graph.nodes:
            raise ValueError(f"源节点 {source_id} 不在图中")

        # 初始化DFS状态
        visited = {source_id: True}
        stack = [source_id]
        distances = {source_id: 0}
        predecessors = {}
        traversal_sequence = [graph.nodes[source_id]]

        while stack:
            current_id = stack.pop()
            current_node = graph.nodes[current_id]

            # 遍历当前节点的所有邻接边（逆序压栈以保证顺序正确）
            for edge in reversed(graph.get_neighbors(current_node)):
                neighbor_id = edge.target.id
                if neighbor_id not in visited:
                    visited[neighbor_id] = True
                    stack.append(neighbor_id)
                    distances[neighbor_id] = distances[current_id] + 1
                    predecessors[neighbor_id] = current_id
                    traversal_sequence.append(edge.target)

        return {
            "sequence": traversal_sequence,
            "distances": distances,
            "predecessors": predecessors
        }

    def get_shortest_paths(self, graph, source: str, target: str) -> List[List[str]]:
        """
        获取两节点间所有最短路径（节点ID序列）

        高性能路径：igraph get_all_shortest_paths（C 核心）。
        graph 可以是 GX Graph（推荐）或 networkx 图（自动兼容转换）。

        参数:
            graph: 图实例
            source: 源节点ID
            target: 目标节点ID

        返回:
            最短路径列表，每个路径为节点ID组成的列表
        """
        try:
            return gb.all_shortest_paths(graph, source, target)
        except ValueError:
            raise
        except Exception as exc:
            if "路径" in str(exc) or "No path" in str(exc):
                return []  # 无路径时返回空列表
            raise

    def get_all_simple_paths(self, graph, source: str, target: str, max_depth: int = 3) -> List[List[str]]:
        """
        获取两节点间所有简单路径（无重复节点），限制最大深度

        高性能路径：igraph get_all_simple_paths（C 核心）。
        graph 可以是 GX Graph（推荐）或 networkx 图（自动兼容转换）。

        参数:
            graph: 图实例
            source: 源节点ID
            target: 目标节点ID
            max_depth: 最大路径长度（节点数），默认3

        返回:
            简单路径列表，每个路径为节点ID组成的列表
        """
        if max_depth < 1:
            raise ValueError("最大深度必须为正整数")

        try:
            return gb.all_simple_paths(graph, source, target, cutoff=max_depth - 1)
        except ValueError:
            raise
        except Exception as exc:
            if "路径" in str(exc) or "No path" in str(exc):
                return []  # 无路径时返回空列表
            raise

    def has_cycle_in_path(self, path: List[Union[Node, Dict[str, Any]]], max_check_depth: Optional[int] = None) -> bool:
        """
        检测路径中是否存在环（重复节点）
        
        参数:
            path: 路径列表，元素为Node对象或含"id"键的字典
            max_check_depth: 最大检查深度（前N个节点），None表示检查全部
        
        返回:
            若存在环（重复节点ID）则返回True，否则False
        """
        if not path:
            return False  # 空路径无环

        # 提取节点ID列表
        node_ids = []
        for item in path:
            if isinstance(item, Node):
                node_ids.append(item.id)
            elif isinstance(item, dict) and "id" in item:
                node_ids.append(item["id"])
            else:
                raise TypeError("路径元素必须是Node对象或含'id'键的字典")

        # 限制检查深度
        check_depth = max_check_depth if max_check_depth is not None else len(node_ids)
        check_ids = node_ids[:check_depth]

        # 检查是否有重复ID（存在环）
        seen = set()
        for node_id in check_ids:
            if node_id in seen:
                return True
            seen.add(node_id)
        return False

    def compute_core_neighbor_density(self, adj_matrix: np.ndarray, node_id: int) -> float:
        """
        计算核心节点的邻居密度（基于邻接矩阵）
        
        参数:
            adj_matrix: 图的邻接矩阵（numpy数组）
            node_id: 核心节点的索引（对应邻接矩阵的行/列）
        
        返回:
            邻居密度值（越大表示邻居间连接越紧密）
        """
        if not isinstance(adj_matrix, np.ndarray):
            raise TypeError("邻接矩阵必须是numpy数组")
        if node_id < 0 or node_id >= adj_matrix.shape[0]:
            raise IndexError(f"节点索引 {node_id} 超出邻接矩阵范围")

        # 获取核心节点的邻居索引（邻接矩阵中值为1的列）
        neighbor_indices = np.where(adj_matrix[node_id] == 1)[0]
        if len(neighbor_indices) == 0:
            return 0.0  # 无邻居时密度为0

        # 计算邻居间的共同连接数
        total_common = 0
        for i in neighbor_indices:
            # 邻居i与其他邻居的共同连接（逻辑与操作后求和）
            common = np.sum(np.logical_and(adj_matrix[i], adj_matrix[node_id]))
            total_common += common

        # 密度 = 总共同连接数 / 邻居数量
        return total_common / len(neighbor_indices)
    
    def dijkstra(self, graph: 'Graph', start: Union[Node, str]) -> Dict[str, Dict[str, Any]]:
        """
        使用Dijkstra算法计算从起始节点到所有其他节点的最短路径
        
        参数:
            graph: 图实例
            start: 起始节点（Node对象或节点ID）
        
        返回:
            字典，键为目标节点ID，值为包含距离和路径的字典
        """
        start_id = start.id if isinstance(start, Node) else start
        if start_id not in graph.nodes:
            raise ValueError(f"起始节点 {start_id} 不在图中")
        
        # 初始化距离字典，所有距离设为无穷大
        distances = {node_id: float('infinity') for node_id in graph.nodes}
        distances[start_id] = 0
        
        # 初始化前驱节点字典，用于重构路径
        predecessors = {node_id: None for node_id in graph.nodes}
        
        # 使用优先队列存储待访问的节点（距离，节点ID）
        import heapq
        priority_queue = [(0, start_id)]
        visited = set()
        
        while priority_queue:
            current_distance, current_id = heapq.heappop(priority_queue)
            
            # 如果已经访问过该节点，跳过
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # 如果当前距离大于已知距离，跳过
            if current_distance > distances[current_id]:
                continue
            
            # 遍历当前节点的所有邻接边
            current_node = graph.nodes[current_id]
            for edge in graph.get_neighbors(current_node):
                neighbor_id = edge.target.id
                weight = edge.weight
                
                # 计算通过当前边到达邻居节点的新距离
                distance = current_distance + weight
                
                # 如果找到更短的路径，更新
                if distance < distances[neighbor_id]:
                    distances[neighbor_id] = distance
                    predecessors[neighbor_id] = current_id
                    heapq.heappush(priority_queue, (distance, neighbor_id))
        
        # 构建结果
        result = {}
        for node_id in graph.nodes:
            # 重构路径
            path = []
            current = node_id
            while current:
                path.append(current)
                current = predecessors[current]
            path.reverse()  # 反转路径，从起始节点开始
            
            result[node_id] = {
                "distance": distances[node_id],
                "path": path
            }
        
        return result
    
    def a_star_search(self, graph: 'Graph', start: Union[Node, str], goal: Union[Node, str], 
                      heuristic: Optional[Callable[[str, str], float]] = None) -> Dict[str, Any]:
        """
        使用A*搜索算法寻找从起始节点到目标节点的最短路径
        
        参数:
            graph: 图实例
            start: 起始节点（Node对象或节点ID）
            goal: 目标节点（Node对象或节点ID）
            heuristic: 启发式函数，接收两个节点ID，返回预估距离。如果为None，使用0（退化为Dijkstra算法）
        
        返回:
            包含路径和距离的字典
        """
        start_id = start.id if isinstance(start, Node) else start
        goal_id = goal.id if isinstance(goal, Node) else goal
        
        if start_id not in graph.nodes:
            raise ValueError(f"起始节点 {start_id} 不在图中")
        if goal_id not in graph.nodes:
            raise ValueError(f"目标节点 {goal_id} 不在图中")
        
        # 默认启发式函数返回0（退化为Dijkstra算法）
        if heuristic is None:
            def heuristic(n1, n2):
                return 0
        
        # 初始化距离字典
        g_score = {node_id: float('infinity') for node_id in graph.nodes}
        g_score[start_id] = 0
        
        # f_score = g_score + 启发式距离
        f_score = {node_id: float('infinity') for node_id in graph.nodes}
        f_score[start_id] = heuristic(start_id, goal_id)
        
        # 前驱节点字典
        predecessors = {node_id: None for node_id in graph.nodes}
        
        # 优先队列（f_score, 节点ID）
        import heapq
        priority_queue = [(f_score[start_id], start_id)]
        visited = set()
        
        while priority_queue:
            _, current_id = heapq.heappop(priority_queue)
            
            # 如果到达目标节点，重构路径并返回
            if current_id == goal_id:
                path = []
                current = current_id
                while current:
                    path.append(current)
                    current = predecessors[current]
                path.reverse()  # 反转路径，从起始节点开始
                return {
                    "path": path,
                    "distance": g_score[goal_id]
                }
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # 遍历当前节点的所有邻接边
            current_node = graph.nodes[current_id]
            for edge in graph.get_neighbors(current_node):
                neighbor_id = edge.target.id
                weight = edge.weight
                
                # 计算通过当前边到达邻居节点的临时g_score
                tentative_g_score = g_score[current_id] + weight
                
                # 如果找到更好的路径，更新
                if tentative_g_score < g_score[neighbor_id]:
                    predecessors[neighbor_id] = current_id
                    g_score[neighbor_id] = tentative_g_score
                    f_score[neighbor_id] = tentative_g_score + heuristic(neighbor_id, goal_id)
                    
                    if neighbor_id not in visited:
                        heapq.heappush(priority_queue, (f_score[neighbor_id], neighbor_id))
        
        # 没有找到路径
        return {
            "path": [],
            "distance": float('infinity')
        }
    
    def multi_target_search(self, graph: 'Graph', start: Union[Node, str], 
                           targets: List[Union[Node, str]]) -> Dict[str, Dict[str, Any]]:
        """
        多目标路径搜索，计算从起始节点到多个目标节点的最短路径
        
        参数:
            graph: 图实例
            start: 起始节点（Node对象或节点ID）
            targets: 目标节点列表
        
        返回:
            字典，键为目标节点ID，值为包含路径和距离的字典
        """
        # 使用Dijkstra算法一次计算到所有节点的最短路径
        all_paths = self.dijkstra(graph, start)
        
        # 过滤出目标节点的路径
        result = {}
        for target in targets:
            target_id = target.id if isinstance(target, Node) else target
            if target_id in all_paths:
                result[target_id] = all_paths[target_id]
        
        return result
    
    def constrained_path_search(self, graph: 'Graph', start: Union[Node, str], 
                              end: Union[Node, str], 
                              max_weight: Optional[float] = None, 
                              min_weight: Optional[float] = None,
                              max_depth: Optional[int] = None) -> List[List[str]]:
        """
        带约束条件的路径搜索
        
        参数:
            graph: 图实例
            start: 起始节点
            end: 目标节点
            max_weight: 最大路径总权重
            min_weight: 最小路径总权重
            max_depth: 最大路径深度（边数）
        
        返回:
            满足约束条件的路径列表
        """
        start_id = start.id if isinstance(start, Node) else start
        end_id = end.id if isinstance(end, Node) else end
        
        if start_id not in graph.nodes or end_id not in graph.nodes:
            return []
        
        # 高性能路径搜索：igraph 后端（networkx 图自动兼容转换）
        ig_graph = gb.to_igraph(graph)

        # 构建边权重映射（同方向多条边取最后一条，与旧版 get_edge_data 语义一致）
        weight_map = {}
        if hasattr(graph, "edges") and hasattr(graph.edges, "values"):
            for edge in graph.edges.values():
                w = edge.weight if getattr(edge, "weight", None) is not None else 1.0
                weight_map[(edge.source.id, edge.target.id)] = float(w)
        else:
            for u, v, data in graph.edges(data=True):
                weight_map[(str(u), str(v))] = float(data.get("weight", 1.0) or 1.0)

        cutoff = max_depth - 1 if max_depth else None
        all_paths = gb.all_simple_paths(graph, start_id, end_id, cutoff=cutoff, ig_graph=ig_graph)

        # 应用约束条件过滤路径
        result_paths = []
        for path in all_paths:
            # 计算路径总权重
            total_weight = 0.0
            for i in range(len(path) - 1):
                total_weight += weight_map.get((path[i], path[i + 1]), 1.0)

            # 检查约束条件
            weight_check = True
            if max_weight is not None and total_weight > max_weight:
                weight_check = False
            if min_weight is not None and total_weight < min_weight:
                weight_check = False

            if weight_check:
                result_paths.append(path)

        return result_paths

class GraphPartition:
    """图分区算法工具类，提供图的社区发现、分区优化等功能"""
    
    def __init__(self):
        pass
        
    def spectral_partition(self, graph: 'Graph', n_clusters: int = 2) -> Dict[str, List[str]]:
        """
        使用谱聚类方法对图进行分区
        
        参数:
            graph: 待分区的图（Graph实例）
            n_clusters: 分区数量，默认为2
            
        返回:
            分区结果字典，键为分区ID，值为该分区包含的节点ID列表
        """
        if n_clusters < 2:
            raise ValueError("分区数量必须大于等于2")

        # 邻接矩阵走 igraph/scipy 高速转换（替代 networkx）
        adj_matrix, node_ids = gb.adjacency_matrix(graph)

        # 计算拉普拉斯矩阵
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        laplacian = degree_matrix - adj_matrix

        # 计算拉普拉斯矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        # 使用前n_clusters个特征向量进行聚类
        features = eigenvectors[:, :n_clusters]

        # 使用k-means聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)

        # 整理分区结果
        partition_result = defaultdict(list)

        for i, cluster_id in enumerate(kmeans.labels_):
            partition_result[str(cluster_id)].append(node_ids[i])

        return dict(partition_result)
    
    def louvain_partition(self, graph: 'Graph') -> Dict[str, List[str]]:
        """
        使用Louvain算法进行社区发现
        
        参数:
            graph: 待分区的图（Graph实例）
            
        返回:
            分区结果字典，键为分区ID，值为该分区包含的节点ID列表
        """
        # 高性能路径：igraph Louvain（C 核心），替代 python-louvain
        membership = gb.louvain(graph)

        # 整理分区结果
        partition_result = defaultdict(list)
        for node_id, community_id in membership.items():
            partition_result[str(community_id)].append(node_id)

        return dict(partition_result)

    def modularity(self, graph: 'Graph', partition: Dict[str, List[str]]) -> float:
        """
        计算给定分区的模块度（Modularity）

        高性能路径：igraph modularity（C 核心）。

        参数:
            graph: 图实例
            partition: 分区结果，键为分区ID，值为节点ID列表

        返回:
            模块度值，范围通常在[-0.5, 1]之间，越高表示分区质量越好
        """
        return gb.modularity(graph, partition)

    def min_cut_partition(self, graph: 'Graph') -> Dict[str, List[str]]:
        """
        使用最小割算法将图分为两个分区

        高性能路径：igraph 最大流最小割（C 核心），保留旧版"最大连通分量 +
        最远点对作为源汇"的启发式。

        参数:
            graph: 待分区的图（Graph实例）

        返回:
            分区结果字典，键为分区ID（'0'和'1'），值为该分区包含的节点ID列表
        """
        return gb.min_cut_partition(graph)
    
    def hierarchical_clustering(self, graph: 'Graph', n_clusters: int = 2) -> Dict[str, int]:
        """
        层次聚类算法实现社区检测
        
        参数:
            graph: 图实例
            n_clusters: 目标聚类数量
        
        返回:
            字典，键为节点ID，值为聚类编号
        """
        import numpy as np
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        # 构建节点列表和邻接矩阵
        nodes = list(graph.nodes.keys())
        n = len(nodes)

        if n <= n_clusters:
            # 如果节点数量小于等于聚类数量，每个节点单独成类
            return {node_id: i for i, node_id in enumerate(nodes)}

        # 全对最短路径距离矩阵一次算完（igraph C 实现），替代旧版 O(n^2) 次 networkx 调用
        dist_matrix, _ = gb.all_pairs_distances(graph, weights=True)
        if np.any(np.isfinite(dist_matrix)):
            finite_max = np.nanmax(dist_matrix[np.isfinite(dist_matrix)])
        else:
            finite_max = 0.0
        # 不可达点对用大有限值代替（旧版直接用 inf，会让 pdist 产生 nan）
        distances = np.where(np.isfinite(dist_matrix), dist_matrix, finite_max * 2.0 + 1.0)

        # 使用层次聚类
        distance_vector = pdist(distances)
        linkage_matrix = linkage(distance_vector, method='ward')

        # 获取聚类结果
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # 转换为字典格式
        result = {nodes[i]: cluster_labels[i] - 1 for i in range(n)}
        return result
    
    def label_propagation(self, graph: 'Graph', max_iterations: int = 100) -> Dict[str, int]:
        """
        标签传播算法实现社区检测

        高性能路径：igraph community_label_propagation（C 核心，无向视图）。

        参数:
            graph: 图实例
            max_iterations: 最大迭代次数（igraph 内部自动收敛，保留参数兼容旧接口）

        返回:
            字典，键为节点ID，值为社区编号
        """
        return gb.label_propagation(graph)

    def leiden_community_detection(self, graph: 'Graph', resolution: float = 1.0) -> Dict[str, int]:
        """
        Leiden社区检测算法（Louvain算法的改进版）

        高性能路径：leidenalg（构建在 igraph C 核心之上）。

        参数:
            graph: 图实例
            resolution: 分辨率参数，控制社区大小

        返回:
            字典，键为节点ID，值为社区编号
        """
        try:
            return gb.leiden(graph, resolution)
        except ImportError:
            # 如果缺少依赖，使用Louvain算法作为备选
            print("警告: 未安装leidenalg库，使用Louvain算法代替")
            return self.louvain_community_detection(graph)
    
    def spectral_bipartitioning(self, graph: 'Graph') -> Tuple[List[str], List[str]]:
        """
        谱二分法，将图分成两个部分

        高性能路径：SciPy 稀疏拉普拉斯矩阵 + Fiedler 向量（大图走 ARPACK），
        替代 networkx 全量稠密计算。

        参数:
            graph: 图实例

        返回:
            两个部分的节点ID列表
        """
        import numpy as np

        fiedler_vector, node_ids = gb.fiedler_vector(graph)

        # 根据Fiedler向量的符号将节点分为两组
        group1 = [node_ids[i] for i in range(len(node_ids)) if fiedler_vector[i] >= 0]
        group2 = [node_ids[i] for i in range(len(node_ids)) if fiedler_vector[i] < 0]

        return group1, group2
    
    def balanced_partitioning(self, graph: 'Graph', num_partitions: int, 
                             balance_factor: float = 0.1) -> Dict[str, int]:
        """
        平衡分区算法，确保各分区大小相对均衡
        
        参数:
            graph: 图实例
            num_partitions: 分区数量
            balance_factor: 平衡因子，控制分区大小差异
        
        返回:
            字典，键为节点ID，值为分区编号
        """
        # 使用多层次k-way分区
        try:
            # 二分：Fiedler 向量谱二分（igraph/scipy 加速），替代 networkx 的 Kernighan-Lin
            group1, group2 = self.spectral_bipartitioning(graph)
            partitions = (group1, group2)
            
            # 如果需要更多分区，递归进行二分
            if num_partitions > 2:
                result = {}
                for i, partition in enumerate(partitions):
                    # 为子图创建新的图实例
                    subgraph = graph.subgraph(partition)
                    # 递归分区
                    sub_partitions = self.balanced_partitioning(
                        subgraph, 
                        num_partitions // 2,
                        balance_factor
                    )
                    # 合并结果
                    for node_id, sub_id in sub_partitions.items():
                        result[node_id] = i * (num_partitions // 2) + sub_id
                return result
            else:
                # 两分区情况
                result = {}
                for i, partition in enumerate(partitions):
                    for node_id in partition:
                        result[node_id] = i
                return result
        except:
            # 如果METIS不可用，使用Louvain算法并调整
            communities = self.louvain_community_detection(graph)
            
            # 按社区大小排序
            community_sizes = {}
            for node_id, comm_id in communities.items():
                if comm_id not in community_sizes:
                    community_sizes[comm_id] = 0
                community_sizes[comm_id] += 1
            
            # 按大小排序社区
            sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
            
            # 分配社区到分区，确保均衡
            partition_assignment = {}
            partition_sizes = [0] * num_partitions
            
            for comm_id, size in sorted_communities:
                # 找到最小的分区
                min_partition = partition_sizes.index(min(partition_sizes))
                partition_assignment[comm_id] = min_partition
                partition_sizes[min_partition] += size
            
            # 转换为节点-分区映射
            result = {}
            for node_id, comm_id in communities.items():
                result[node_id] = partition_assignment[comm_id]
            
            return result
    
    def louvain_community_detection(self, graph: 'Graph', resolution: float = 1.0) -> Dict[str, int]:
        """
        使用Louvain算法进行社区检测，返回节点到社区的映射

        高性能路径：igraph Louvain（C 核心，无向视图），替代 python-louvain。

        参数:
            graph: 图实例
            resolution: 分辨率参数，控制社区大小

        返回:
            字典，键为节点ID，值为社区编号
        """
        return gb.louvain(graph, resolution)
