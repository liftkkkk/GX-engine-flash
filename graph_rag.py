import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from graph_engine import Graph, Node, Edge

class MockEmbedding:
    """
    一个用于演示的Mock Embedding生成器。
    在实际生产环境中，这里应该替换为 OpenAI, HuggingFace 或其它 Embedding API。
    """
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.cache = {}

    def embed(self, text: str) -> List[float]:
        """
        生成确定性的随机向量，相同的文本生成相同的向量。
        """
        if text in self.cache:
            return self.cache[text]
        
        # 使用简单的哈希生成种子，保证确定性
        seed = hash(text) % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.random(self.dim)
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        result = vector.tolist()
        self.cache[text] = result
        return result

class SemanticGraph(Graph):
    """
    支持向量检索的语义图引擎。
    结合了图结构（拓扑）和向量空间（语义）。
    """
    def __init__(self, embedding_model=None, directed: bool = True):
        super().__init__(directed)
        self.embedding_model = embedding_model if embedding_model else MockEmbedding()
        
    def add_node_with_text(self, node_id: str, text_content: str, **kwargs) -> Node:
        """
        添加节点，并自动计算文本内容的Embedding。
        """
        embedding = self.embedding_model.embed(text_content)
        node = Node(
            id=node_id, 
            name=text_content[:20], # 使用前20个字符作为名称
            data=text_content,
            embedding=embedding,
            attributes=kwargs
        )
        self.add_node(node)
        return node

    def search_similar_nodes(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Tuple[Node, float]]:
        """
        语义搜索：查找与查询文本语义最相似的节点。
        """
        query_vector = np.array(self.embedding_model.embed(query))
        results = []

        for node in self.nodes.values():
            if node.embedding:
                node_vector = np.array(node.embedding)
                # 计算余弦相似度
                similarity = np.dot(query_vector, node_vector)
                if similarity >= threshold:
                    results.append((node, float(similarity)))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def retrieve_context(self, query: str, max_nodes: int = 5, hops: int = 1) -> str:
        """
        GraphRAG 核心功能：
        1. 找到语义相关的入口节点
        2. 遍历其邻居（1跳或多跳）
        3. 组装成文本上下文返回给LLM
        """
        similar_nodes = self.search_similar_nodes(query, top_k=max_nodes)
        
        context_parts = []
        visited = set()

        for node, score in similar_nodes:
            if node.id in visited:
                continue
            
            # 添加入口节点信息
            context_parts.append(f"相关节点 (相似度 {score:.2f}): {node.data}")
            visited.add(node.id)

            # 获取邻居信息 (1跳)
            if hops > 0:
                neighbors = self.get_neighbors(node)
                if neighbors:
                    context_parts.append(f"  - 关联信息:")
                    for edge in neighbors:
                        target = edge.target
                        if target.id not in visited:
                            rel_text = edge.label if edge.label else "关联"
                            context_parts.append(f"    --[{rel_text}]--> {target.data}")
                            visited.add(target.id)
            
            context_parts.append("") # 空行分隔

        return "\n".join(context_parts)

# 简单的测试代码
if __name__ == "__main__":
    sg = SemanticGraph()
    
    # 模拟构建知识图谱
    n1 = sg.add_node_with_text("1", "Python是一种广泛使用的高级编程语言。", category="Language")
    n2 = sg.add_node_with_text("2", "Guido van Rossum在1989年创造了Python。", category="Person")
    n3 = sg.add_node_with_text("3", "Python支持面向对象、命令式、函数式编程。", category="Features")
    n4 = sg.add_node_with_text("4", "Java是一种静态类型的面向对象语言。", category="Language")
    
    sg.add_edge_from_st(n1, n2, label="创造者")
    sg.add_edge_from_st(n1, n3, label="特性")
    
    # 语义搜索测试
    query = "谁发明了Python?"
    print(f"查询: {query}")
    
    # 1. 纯向量搜索
    print("\n--- 向量搜索结果 ---")
    results = sg.search_similar_nodes(query)
    for node, score in results:
        print(f"[{score:.4f}] {node.data}")
        
    # 2. GraphRAG 上下文检索
    print("\n--- GraphRAG 上下文检索 ---")
    context = sg.retrieve_context(query)
    print(context)
