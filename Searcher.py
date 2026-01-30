import os
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import time
from collections import deque


def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 1.0  # 如果任意向量为零向量，则返回最大距离
    return 1 - dot_product / (norm_vec1 * norm_vec2)


def manhattan_distance(vec1, vec2):
    """计算曼哈顿距离"""
    return np.sum(np.abs(vec1 - vec2))


class Searcher:
    def __init__(self, data_path, semantic_descriptions=None):
        """
        初始化搜索器
        :param data_path: 数据路径
        :param semantic_descriptions: 动作的语义描述列表，用于语义锚点聚类
        """
        # 加载动作向量数据
        vectors_path = os.path.join(data_path, 'ALL-Action-750.npy')
        self.vectors = np.load(vectors_path).astype(np.float32)
        self.vectors = np.nan_to_num(self.vectors) / 100
        
        # 加载动作时长数据
        durations_path = os.path.join(data_path, 'action-length-750.npy')
        action_durations = np.load(durations_path).astype(np.float32)
        self.action_durations = np.nan_to_num(action_durations)
        
        # 如果存在预构建的图则加载，否则构建新图
        graph_path = os.path.join(data_path, "atomic_motion_graph.pkl")
        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                self.G_loaded = pickle.load(f)
        else:
            # 如果没有预构建的图且提供了语义描述，则构建基于新逻辑的图
            if semantic_descriptions is not None:
                self.G_loaded = self.construct_atomic_motion_graph(self.vectors, semantic_descriptions)
            else:
                # 如果没有语义描述，构建一个基础图
                self.G_loaded = self._build_basic_graph(self.vectors)
                
        self.data_path = data_path
        self.action_dimension = self.vectors.shape[1]
        self.action_count = self.vectors.shape[0]
        
        # 预计算向量范数以加速距离计算
        self._precompute_vector_norms()
        
        # 如果有语义描述，计算语义锚点
        if semantic_descriptions is not None:
            self.semantic_anchors = self._compute_semantic_anchors(semantic_descriptions)
        else:
            self.semantic_anchors = None

    def _precompute_vector_norms(self):
        """预计算向量的L2范数以加速距离计算"""
        self.vector_norms = np.linalg.norm(self.vectors, axis=1)

    def _compute_semantic_anchors(self, semantic_descriptions, n_clusters=20):
        """
        计算语义锚点
        """
        # 使用TF-IDF向量化文本描述
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(semantic_descriptions)
        
        # 使用K-means聚类创建语义锚点
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # 计算每个簇的中心作为语义锚点
        anchors = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.any(cluster_mask):
                cluster_vectors = tfidf_matrix[cluster_mask].toarray()
                anchor = np.mean(cluster_vectors, axis=0)
                anchors.append(anchor)
            else:
                anchors.append(np.zeros(tfidf_matrix.shape[1]))
        
        return np.array(anchors)

    def construct_atomic_motion_graph(self, action_features, semantic_descriptions, n_clusters=20, k_kinematic=10):
        """
        根据论文中的两层层次化结构构建原子动作图
        Level 1: Semantic Anchors (基于语义描述的聚类)
        Level 2: Kinematic Topology (基于运动学过渡成本的连接)
        
        :param action_features: 动作特征
        :param semantic_descriptions: 语义描述列表
        :param n_clusters: 语义聚类的数量
        :param k_kinematic: 运动学连接的邻居数
        :return: 构建的图
        """
        num_actions = len(action_features)
        G = nx.DiGraph()  # 使用有向图表示运动学连接的方向性
        
        # 添加所有节点
        for i in range(num_actions):
            G.add_node(i, feature=action_features[i])
        
        # Level 1: 语义锚点聚类
        print("正在构建语义锚点层...")
        semantic_clusters = self._build_semantic_anchors(semantic_descriptions, n_clusters)
        
        # 为每个节点分配语义簇标签
        for i, cluster_id in enumerate(semantic_clusters):
            G.nodes[i]['semantic_cluster'] = cluster_id
        
        # Level 2: 运动学拓扑连接
        print("正在构建运动学拓扑层...")
        self._build_kinematic_topology(G, action_features, k_kinematic)
        
        # 保存图到文件
        graph_path = os.path.join(self.data_path, "atomic_motion_graph.pkl")
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        
        print(f"原子动作图构建完成，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
        return G

    def _build_semantic_anchors(self, semantic_descriptions, n_clusters):
        """
        构建语义锚点：使用文本描述进行聚类
        """
        # 使用TF-IDF向量化文本描述
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(semantic_descriptions)
        
        # 使用K-means聚类创建语义锚点
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        semantic_clusters = kmeans.fit_predict(tfidf_matrix)
        
        return semantic_clusters

    def _build_kinematic_topology(self, G, action_features, k_kinematic):
        """
        构建运动学拓扑：基于物理过渡成本连接节点
        """
        num_actions = len(action_features)
        
        # 为每个动作找到k个kinematically closest的邻居
        nbrs = NearestNeighbors(n_neighbors=min(k_kinematic+1, num_actions), 
                               metric='euclidean', algorithm='auto')
        nbrs.fit(action_features)
        distances, indices = nbrs.kneighbors(action_features)
        
        for i in range(num_actions):
            # 对于每个动作，连接到k个运动学上最接近的邻居（排除自己）
            for j in range(1, min(k_kinematic+1, len(indices[i]))):  # 从1开始跳过自己
                neighbor_idx = indices[i][j]
                
                # 计算过渡成本（基于关节位置和速度的不连续性）
                transition_cost = self._calculate_transition_cost(
                    action_features[i], action_features[neighbor_idx]
                )
                
                # 只添加低于阈值的连接以确保平滑过渡
                if transition_cost < np.percentile(distances[i][1:], 80):  # 只连接到相对较近的邻居
                    G.add_edge(i, neighbor_idx, weight=transition_cost, transition_cost=transition_cost)

    def _calculate_transition_cost(self, action1, action2):
        """
        计算两个动作之间的过渡成本
        基于动作末尾和开头的关节位置/速度不连续性
        """
        # 简化的过渡成本计算：基于动作特征的欧几里得距离
        # 在实际应用中，这可以扩展为更复杂的运动学模型
        cost = np.linalg.norm(action1[-1] - action2[0])  # 动作1的结尾到动作2的开头
        return cost

    def _build_basic_graph(self, action_features, k=10):
        """
        构建基础图（如果未提供语义描述）
        """
        print("正在构建基础图...")
        G = nx.DiGraph()
        num_actions = len(action_features)
        
        # 添加所有节点
        for i in range(num_actions):
            G.add_node(i, feature=action_features[i])
        
        # 使用KNN连接建立基本拓扑
        nbrs = NearestNeighbors(n_neighbors=min(k+1, num_actions), metric='euclidean', algorithm='auto')
        nbrs.fit(action_features)
        distances, indices = nbrs.kneighbors(action_features)
        
        for i in range(num_actions):
            for j in range(1, min(k+1, len(indices[i]))):  # 从1开始跳过自己
                neighbor_idx = indices[i][j]
                distance = distances[i][j]
                G.add_edge(i, neighbor_idx, weight=distance)

        graph_path = os.path.join(self.data_path, "atomic_motion_graph.pkl")
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        return G

    def search_vector_index(self, input_vector, wave_length):
        """
        使用分层图约束搜索算法搜索最匹配的向量索引
        :param input_vector: 输入向量
        :param wave_length: 波长（用于动作持续时间约束）
        :return: 匹配的索引和对应的动作时长
        """
        index = self.hierarchical_graph_search(input_vector, self.G_loaded, self.vectors, wave_length)
        return index, self.action_durations[index]

    def hierarchical_graph_search_with_previous(self, prev_primitive, current_vector, G, vectors, action_duration, 
                                               lambda_sem=1.0, lambda_phy=1.0):
        """
        分层图约束搜索算法（带前置动作）
        Stage 1: Semantic Pruning (Global Scope)
        Stage 2: Topological Refinement (Local Scope)
        
        :param prev_primitive: 前一个原子动作
        :param current_vector: 当前潜在动作查询
        :param G: 原子动作图
        :param vectors: 所有动作向量
        :param action_duration: 期望的动作持续时间
        :param lambda_sem: 语义成本权重
        :param lambda_phy: 物理成本权重
        :return: 最佳匹配的动作索引
        """
        # 检查是否存在语义锚点
        if self.semantic_anchors is not None:
            # Stage 1: Semantic Pruning (Global Scope)
            best_anchor_idx = self._identify_target_semantic_anchor(current_vector)
            candidate_primitives = self._prune_search_space(best_anchor_idx)
        else:
            # 如果没有语义锚点，使用所有节点作为候选
            candidate_primitives = list(range(len(vectors)))
        
        # Stage 2: Topological Refinement (Local Scope)
        optimal_primitive = self._refine_topologically_with_prev(
            prev_primitive, current_vector, G, vectors, candidate_primitives, 
            action_duration, lambda_sem, lambda_phy
        )
        
        return optimal_primitive

    def _identify_target_semantic_anchor(self, query_vector):
        """
        识别目标语义意图
        k^* <- argmax_k CosineSim(q_t, c_k)
        """
        if self.semantic_anchors is None:
            return 0  # 默认返回第一个
        
        similarities = []
        for anchor in self.semantic_anchors:
            # 将查询向量和锚点向量标准化
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
            anchor_norm = anchor / (np.linalg.norm(anchor) + 1e-8)
            similarity = np.dot(query_norm, anchor_norm)
            similarities.append(similarity)
        
        return np.argmax(similarities)

    def _prune_search_space(self, anchor_idx):
        """
        修剪搜索空间
        V_candidate <- { v ∈ V | Cluster(v) = k^* }
        """
        if 'semantic_cluster' not in self.G_loaded.nodes[0]:
            # 如果没有语义簇信息，返回所有节点
            return list(self.G_loaded.nodes())
        
        candidate_primitives = []
        for node_id in self.G_loaded.nodes():
            if self.G_loaded.nodes[node_id].get('semantic_cluster') == anchor_idx:
                candidate_primitives.append(node_id)
        
        return candidate_primitives if candidate_primitives else list(self.G_loaded.nodes())

    def _refine_topologically_with_prev(self, prev_primitive, current_vector, G, vectors, 
                                        candidate_primitives, action_duration, lambda_sem, lambda_phy):
        """
        使用前置动作进行拓扑细化
        """
        min_cost = float('inf')
        optimal_primitive = None
        
        for primitive_idx in candidate_primitives:
            # 检查动作持续时间是否满足条件
            if self.action_durations[primitive_idx] <= action_duration:
                continue
                
            # 计算语义对齐成本
            semantic_cost = self._calculate_semantic_alignment_cost(current_vector, vectors[primitive_idx])
            
            # 检查是否有有效的物理过渡
            if prev_primitive is not None and G.has_edge(prev_primitive, primitive_idx):
                physical_cost = G[prev_primitive][primitive_idx].get('transition_cost', float('inf'))
            else:
                # 如果没有有效过渡，给予无限大的惩罚
                physical_cost = float('inf')
            
            # 计算总成本
            total_cost = lambda_sem * semantic_cost + lambda_phy * physical_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                optimal_primitive = primitive_idx
        
        # 如果没有找到满足持续时间条件的候选动作，返回默认值
        return optimal_primitive if optimal_primitive is not None else 0

    def _calculate_semantic_alignment_cost(self, query_vector, primitive_vector):
        """
        计算语义对齐成本
        E_sem <- ||q_t - Enc(m_j)||_2^2
        """
        diff = query_vector - primitive_vector
        return np.sum(diff ** 2)

    def hierarchical_graph_search(self, current_vector, G, vectors, action_duration, 
                                  lambda_sem=1.0, lambda_phy=1.0):
        """
        分层图约束搜索算法（无前置动作）
        Stage 1: Semantic Pruning (Global Scope)
        Stage 2: Topological Refinement (Local Scope)
        
        :param current_vector: 当前潜在动作查询
        :param G: 原子动作图
        :param vectors: 所有动作向量
        :param action_duration: 期望的动作持续时间
        :param lambda_sem: 语义成本权重
        :param lambda_phy: 物理成本权重
        :return: 最佳匹配的动作索引
        """
        # 检查是否存在语义锚点
        if self.semantic_anchors is not None:
            # Stage 1: Semantic Pruning (Global Scope)
            best_anchor_idx = self._identify_target_semantic_anchor(current_vector)
            candidate_primitives = self._prune_search_space(best_anchor_idx)
        else:
            # 如果没有语义锚点，使用所有节点作为候选
            candidate_primitives = list(range(len(vectors)))
        
        # Stage 2: Topological Refinement (Local Scope)
        optimal_primitive = self._refine_topologically_without_prev(
            current_vector, G, vectors, candidate_primitives, 
            action_duration, lambda_sem, lambda_phy
        )
        
        return optimal_primitive

    def _refine_topologically_without_prev(self, current_vector, G, vectors, 
                                           candidate_primitives, action_duration, 
                                           lambda_sem, lambda_phy):
        """
        不使用前置动作进行拓扑细化
        """
        min_cost = float('inf')
        optimal_primitive = None
        
        for primitive_idx in candidate_primitives:
            # 检查动作持续时间是否满足条件
            if self.action_durations[primitive_idx] <= action_duration:
                continue
                
            # 计算语义对齐成本
            semantic_cost = self._calculate_semantic_alignment_cost(current_vector, vectors[primitive_idx])
            
            # 对于无前置动作的情况，物理成本设为0
            physical_cost = 0
            
            # 计算总成本
            total_cost = lambda_sem * semantic_cost + lambda_phy * physical_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                optimal_primitive = primitive_idx
        
        # 如果没有找到满足持续时间条件的候选动作，返回默认值
        return optimal_primitive if optimal_primitive is not None else 0


if __name__ == "__main__":
    # 示例：如何使用带有语义描述的新图构建
    # 首先需要提供动作的语义描述
    # semantic_descs = ["idle movement", "hand gesture", "nodding", ...] # 需要根据实际情况提供
    # searcher = Searcher("./Data/", semantic_descriptions=semantic_descs)
    
    # 或者使用现有数据初始化（会尝试加载预构建的图或使用基础图）
    searcher = Searcher("./Data/")
    
    print(f"加载了 {searcher.action_count} 个动作向量")
    print(f"动作维度: {searcher.action_dimension}")
    
    # 测试搜索功能
    for i in range(min(5, len(searcher.vectors))):  # 测试前5个动作
        current_vector = searcher.vectors[i]
        best_action, duration = searcher.search_vector_index(current_vector, 7)
        print(f"输入动作 {i}: 最佳匹配动作 ID: {best_action}, 时长: {duration}")