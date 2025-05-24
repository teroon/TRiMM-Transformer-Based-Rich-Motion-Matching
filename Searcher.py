import os
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import scipy
import time


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return 1-dot_product / (norm_vec1 * norm_vec2)

def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

def calculate_fgd(generated_features, real_features):
    """
    Calculate the Frechet Gesture Distance (FGD)
    :param generated_features: Feature vectors of the generated gestures
    :param real_features: Feature vectors of the real gestures
    :return: FGD value
    """
    # 直接计算均值
    mu_g = np.mean(generated_features, axis=0)
    mu_r = np.mean(real_features, axis=0)
    # 计算协方差矩阵
    sigma_g = np.cov(generated_features, rowvar=False)
    sigma_r = np.cov(real_features, rowvar=False)
    
    # 计算 L2 范数的平方
    diff = mu_g - mu_r
    l2_norm_squared = np.sum(diff ** 2)
    
    # 计算协方差矩阵的乘积
    cov_product = np.dot(sigma_g, sigma_r)
    
    # 计算矩阵的平方根
    try:
        sqrt_cov_product = scipy.linalg.sqrtm(cov_product)
        if np.iscomplexobj(sqrt_cov_product):
            sqrt_cov_product = sqrt_cov_product.real
    except np.linalg.LinAlgError:
        # 处理奇异矩阵的情况
        eye = np.eye(sigma_g.shape[0]) * 1e-6
        sqrt_cov_product = scipy.linalg.sqrtm((sigma_g + eye).dot(sigma_r + eye))
        if np.iscomplexobj(sqrt_cov_product):
            sqrt_cov_product = sqrt_cov_product.real
    
    # 计算矩阵的迹
    trace_term = np.trace(sigma_g + sigma_r - 2 * sqrt_cov_product)
    
    # 计算 FGD
    fgd = abs(l2_norm_squared + trace_term)
    return fgd

class Searcher:
    def __init__(
            self,
            data_path
    ):
        self.vectors=np.load(data_path+'ALL-Action-750.npy').astype(np.float32)
        self.vectors=np.nan_to_num(self.vectors)/100
        action_durations =np.load(data_path+'action-length-750.npy').astype(np.float32)
        self.action_durations =np.nan_to_num(action_durations )
        with open(data_path+"knn_graph.pkl", "rb") as f:
            self.G_loaded = pickle.load(f)
        self.data_path = data_path
        self.action_dimension = self.vectors.shape[1]
        self.action_count = self.vectors.shape[0]


    def search_vector_index(
            self,
            input_vector,
            wave_length
    ):
        #start_time = time.time()
        index = self.constrained_knn_search( input_vector, self.G_loaded, self.vectors,wave_length)
        #end_time = time.time()  
        #print ("search time:",end_time - start_time)
        # 将 numpy 数组转换为 torch.Tensor
        #input_tensor = input_vector
        #selected_vector_tensor = self.vectors[index]

        return index,self.action_durations[index]
    
    def constrained_knn_search_pre(self,prev_vector, current_vector, G, vectors,action_duration, top_k=10):
        """
        在 K-NN 图中检索最符合当前特征向量的动作，并考虑上一次的动作

        :param prev_vector: 上一次的特征向量
        :param current_vector: 当前的特征向量
        :param G: K-NN 图
        :param vectors: 所有动作特征向量
        :param top_k: 选取上一次动作的最近邻数量，默认为 5
        :return: 最佳匹配的动作索引
        """
        # 找到上一次向量在 vectors 中的最近邻节点
        prev_nearest_idx = np.argmin(np.linalg.norm(vectors - prev_vector, axis=1))
        # 找到上一次动作的最近邻
        prev_neighbors = sorted(G[prev_nearest_idx], key=lambda x: np.linalg.norm(vectors[x] - prev_vector))[:top_k]
        # 初始化已访问节点集合
        visited = set()
        # 初始化待搜索节点队列
        queue = list(prev_neighbors)
        # 初始化符合条件的节点列表及其距离
        valid_nodes = []
        valid_distances = []

        while queue:
            # 取出队列中的第一个节点
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            # 检查动作持续时间是否满足条件
            if self.action_durations[node] > action_duration:
                # 计算与当前向量的距离
                distance = np.linalg.norm(vectors[node] - current_vector)
                valid_nodes.append(node)
                valid_distances.append(distance)

            # 将当前节点的邻居加入队列
            neighbors = list(G[node])
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

        if valid_nodes:
            # 找到距离最小的节点
            best_index = np.argmin(valid_distances)
            return valid_nodes[best_index]
        else:
            # 如果没有找到符合条件的节点，返回 None
            return None
        
    def constrained_knn_search(self,current_vector, G, vectors, action_duration, top_k=10):
        """
        在 K-NN 图中检索最符合当前特征向量的动作，不考虑上一次的动作

        :param current_vector: 当前的特征向量
        :param G: K-NN 图
        :param vectors: 所有动作特征向量
        :param action_duration: 期望的动作持续时间
        :param top_k: 选取当前动作的最近邻数量，默认为 10
        :return: 最佳匹配的动作索引
        """
        # 找到当前向量在 vectors 中的最近邻节点
        current_nearest_idx = np.argmin(np.linalg.norm(vectors - current_vector, axis=1))
        if self.action_durations[current_nearest_idx] > action_duration:
            return current_nearest_idx
        # 找到当前动作的最近邻
        current_neighbors = sorted(G[current_nearest_idx], key=lambda x: np.linalg.norm(vectors[x] - current_vector))[:top_k]

        # 初始化已访问节点集合
        visited = set()
        # 初始化待搜索节点队列
        queue = list(current_neighbors)
        # 初始化符合条件的节点列表及其距离
        valid_nodes = []
        valid_distances = []

        while queue:
            # 取出队列中的第一个节点
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            # 检查动作持续时间是否满足条件
            if self.action_durations[node] > action_duration:
                # 计算与当前向量的距离
                distance = np.linalg.norm(vectors[node] - current_vector)
                valid_nodes.append(node)
                valid_distances.append(distance)

            # 将当前节点的邻居加入队列
            neighbors = list(G[node])
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

        if valid_nodes:
            # 找到距离最小的节点
            best_index = np.argmin(valid_distances)
            return valid_nodes[best_index]
        else:
            # 如果没有找到符合条件的节点，返回 None
            return 0
        
    def ContructGraph(self, action_features):
        k = 10  # 设定最近邻数量
        G = nx.Graph()
        num_actions = len(action_features)

        for i in range(num_actions):
            fgd_distances = []
            for j in range(num_actions):
                if i != j:
                    # 计算 FGD 距离
                    fgd = calculate_fgd(action_features[i].reshape(75, -1), action_features[j].reshape(75, -1))
                    fgd_distances.append((j, fgd))

            # 按 FGD 距离排序，取前 k 个最近邻
            fgd_distances.sort(key=lambda x: x[1])
            top_k_neighbors = fgd_distances[:k]

            for neighbor_idx, fgd_distance in top_k_neighbors:
                G.add_edge(i, neighbor_idx, weight=fgd_distance)

        with open(self.data_path + "knn_graph.pkl", "wb") as f:
            pickle.dump(G, f)
        return G


if __name__ == "__main__":
    searcher = Searcher(
            "./Data/",
        )
    #searcher.ContructGraph(searcher.vectors)
    fgd=0
    fgd_alt=0
    for i in range(100):
        current_vector = searcher.vectors[i]
        best_action = searcher.constrained_knn_search( current_vector, searcher.G_loaded, searcher.vectors,7)
        print(f"最佳匹配动作 ID: {best_action},leng{searcher.action_durations[best_action]}")
        fgd+=calculate_fgd(current_vector.reshape(75,-1),searcher.vectors[best_action].reshape(75,-1))
        fgd_alt+=calculate_fgd(current_vector.reshape(75,-1),searcher.vectors[0].reshape(75,-1))
        print(fgd)
        print(fgd_alt)
