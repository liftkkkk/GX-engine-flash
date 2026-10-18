from graph_engine import *
import random, string
def generate_random_graph(num_nodes: int = 5, num_edges: int = 8) -> Graph:
    """生成随机图"""
    graph = Graph()
    node_classes = ['A', 'B', 'C', 'D', None]
    edge_labels = ['friend', 'work', 'family', 'follow', None]
    
    # 创建节点
    for i in range(num_nodes):
        node_id = f"n{i}"
        name = ''.join(random.choices(string.ascii_uppercase, k=4))
        class_ = random.choice(node_classes)
        weight = random.random() if random.random() > 0.3 else None
        level = random.randint(1, 5) if random.random() > 0.4 else None
        
        node = Node(
            id=node_id,
            name=name,
            class_=class_,
            weight=weight,
            data=level
        )
        graph.add_node(node)
    
    # 创建边
    existing_edges = set()
    nodes = list(graph.nodes.values())
    
    for i in range(num_edges):
        while True:
            source, target = random.sample(nodes, 2)
            # 避免重复边
            if (source.id, target.id) not in existing_edges:
                existing_edges.add((source.id, target.id))
                break
        
        edge_id = f"e_{source.id}_{target.id}"
        weight = round(random.uniform(0.1, 5.0), 2)
        label = random.choice(edge_labels)
        
        edge = Edge(
            id=edge_id,
            source=source,
            target=target,
            weight=weight,
            label=label
        )
        graph.add_edge(edge)
    
    return graph

def test_graph_partition():
    """测试 GraphPartition 类的所有方法"""
    print("\n===== 测试 GraphPartition 类 =====")
    
    # 创建一个简单的测试图
    graph = Graph()
    
    # 添加节点
    node1 = Node(id="n1", name="Node1")
    node2 = Node(id="n2", name="Node2")
    node3 = Node(id="n3", name="Node3")
    node4 = Node(id="n4", name="Node4")
    node5 = Node(id="n5", name="Node5")
    
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_node(node4)
    graph.add_node(node5)
    
    # 添加边，创建两个明显的社区: {n1,n2,n3} 和 {n3,n4,n5}
    graph.add_edge(Edge(id="e1", source=node1, target=node2, weight=1.0))
    graph.add_edge(Edge(id="e2", source=node2, target=node3, weight=1.0))
    graph.add_edge(Edge(id="e3", source=node1, target=node3, weight=1.0))
    graph.add_edge(Edge(id="e4", source=node3, target=node4, weight=0.5))
    graph.add_edge(Edge(id="e5", source=node4, target=node5, weight=1.0))
    graph.add_edge(Edge(id="e6", source=node3, target=node5, weight=0.5))
    
    print("测试图结构:")
    print(graph.to_str())
    
    # 创建 GraphPartition 实例
    partitioner = GraphPartition()
    
    # 测试谱聚类分区
    print("\n1. 测试谱聚类分区 (spectral_partition):")
    try:
        spectral_result = partitioner.spectral_partition(graph, n_clusters=2)
        print(f"谱聚类分区结果: {spectral_result}")
        
        # 计算模块度
        mod = partitioner.modularity(graph, spectral_result)
        print(f"分区模块度: {mod:.4f}")
    except Exception as e:
        print(f"谱聚类分区测试失败: {e}")
    
    # 测试Louvain分区（如果安装了python-louvain库）
    print("\n2. 测试Louvain分区 (louvain_partition):")
    try:
        louvain_result = partitioner.louvain_partition(graph)
        print(f"Louvain分区结果: {louvain_result}")
        
        # 计算模块度
        mod = partitioner.modularity(graph, louvain_result)
        print(f"分区模块度: {mod:.4f}")
    except ImportError:
        print("跳过Louvain分区测试: python-louvain库未安装")
    except Exception as e:
        print(f"Louvain分区测试失败: {e}")
    
    # 测试最小割分区
    print("\n3. 测试最小割分区 (min_cut_partition):")
    try:
        min_cut_result = partitioner.min_cut_partition(graph)
        print(f"最小割分区结果: {min_cut_result}")
        
        # 计算模块度
        mod = partitioner.modularity(graph, min_cut_result)
        print(f"分区模块度: {mod:.4f}")
    except Exception as e:
        print(f"最小割分区测试失败: {e}")
    
    # 测试更大的随机图
    print("\n4. 测试更大的随机图分区:")
    random_graph = generate_random_graph(num_nodes=10, num_edges=20)
    try:
        random_spectral = partitioner.spectral_partition(random_graph, n_clusters=3)
        print(f"随机图谱聚类分区结果 (3个分区): {random_spectral}")
        mod = partitioner.modularity(random_graph, random_spectral)
        print(f"分区模块度: {mod:.4f}")
    except Exception as e:
        print(f"随机图分区测试失败: {e}")

# 测试随机图生成和图分区
if __name__ == "__main__":
    random.seed(42)  # 设置随机种子确保可重复结果
    
    # 测试随机图生成
    print("===== 测试随机图生成 =====")
    graph = generate_random_graph()
    print(graph.to_str())
    
    # 测试图分区
    test_graph_partition()