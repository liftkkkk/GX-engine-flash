import networkx as nx
import random
import uuid
import os
from graph_engine import Graph, Node, Edge

def convert_nx_to_gx(nx_graph: nx.Graph, directed: bool = True) -> Graph:
    """
    将 NetworkX 图转换为项目的 Graph 对象
    """
    g = Graph(directed=directed)
    
    # 转换节点
    for node_id, data in nx_graph.nodes(data=True):
        # 确保 node_id 是字符串
        str_id = str(node_id)
        
        # 提取常用属性
        name = data.get('name', str_id)
        class_ = data.get('class', data.get('type', None))
        weight = data.get('weight', 1.0)
        
        # 创建节点
        node = Node(id=str_id, name=name, class_=class_, weight=weight)
        
        # 保存其他属性
        for k, v in data.items():
            if k not in ['name', 'class', 'type', 'weight']:
                node.attributes[k] = v
                
        g.add_node(node)
        
    # 转换边
    for u, v, data in nx_graph.edges(data=True):
        source = g.get_node(str(u))
        target = g.get_node(str(v))
        
        if source and target:
            # 提取常用属性
            weight = data.get('weight', 1.0)
            label = data.get('label', None)
            
            # 创建边
            edge_id = str(uuid.uuid4())
            edge = Edge(id=edge_id, source=source, target=target, weight=weight, label=label)
            
            # 保存其他属性
            for k, v in data.items():
                if k not in ['weight', 'label']:
                    edge.attributes[k] = v
            
            g.add_edge(edge)
            
    return g

def generate_random_graph(n=20, p=0.2, seed=None):
    """
    生成经典随机图 (Erdős-Rényi Model)
    n: 节点数
    p: 连边概率
    """
    print(f"生成 Erdős-Rényi 随机图 (n={n}, p={p})...")
    nx_g = nx.gnp_random_graph(n, p, seed=seed)
    return convert_nx_to_gx(nx_g, directed=False)

def generate_small_world_graph(n=20, k=4, p=0.3, seed=None):
    """
    生成小世界网络 (Watts–Strogatz Model)
    n: 节点数
    k: 每个节点的初始邻居数
    p: 随机重连边的概率
    """
    print(f"生成小世界网络 (n={n}, k={k}, p={p})...")
    try:
        nx_g = nx.watts_strogatz_graph(n, k, p, seed=seed)
        return convert_nx_to_gx(nx_g, directed=False)
    except Exception as e:
        print(f"生成失败: {e}")
        return Graph()

def generate_scale_free_graph(n=50, m=2, seed=None):
    """
    生成无标度网络 (Barabási–Albert Model)
    n: 节点数
    m: 每个新节点加入时随机连接的已有节点数
    """
    print(f"生成无标度网络 (n={n}, m={m})...")
    nx_g = nx.barabasi_albert_graph(n, m, seed=seed)
    return convert_nx_to_gx(nx_g, directed=False)

def generate_random_geometric_graph(n=50, radius=0.125, seed=None):
    """
    生成几何随机图
    n: 节点数
    radius: 连接半径
    """
    print(f"生成几何随机图 (n={n}, r={radius})...")
    nx_g = nx.random_geometric_graph(n, radius, seed=seed)
    # 几何图包含位置信息(pos)，可以保存到属性中
    return convert_nx_to_gx(nx_g, directed=False)

if __name__ == "__main__":
    # 创建输出目录
    output_dir = "generated_graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. 生成 Erdős-Rényi 随机图
    g1 = generate_random_graph(n=20, p=0.2, seed=42)
    print(f"  -> 节点数: {g1.size}")
    g1.save(os.path.join(output_dir, "random_erdos_renyi.json"))
    
    # 2. 生成小世界网络
    g2 = generate_small_world_graph(n=30, k=4, p=0.1, seed=42)
    print(f"  -> 节点数: {g2.size}")
    g2.save(os.path.join(output_dir, "random_small_world.json"))
    
    # 3. 生成无标度网络
    g3 = generate_scale_free_graph(n=50, m=2, seed=42)
    print(f"  -> 节点数: {g3.size}")
    g3.save(os.path.join(output_dir, "random_scale_free.json"))
    
    # 4. 生成几何随机图
    g4 = generate_random_geometric_graph(n=50, radius=0.2, seed=42)
    print(f"  -> 节点数: {g4.size}")
    g4.save(os.path.join(output_dir, "random_geometric.json"))

    print(f"\n所有测试图已生成并保存至 {output_dir} 目录。")
