import os
import random
import numpy as np
import bvh
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA  # 用于降维以可视化
from scipy.interpolate import interp1d

def extract_features(bvh_file_path):
    joints =['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'HeadEnd', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase']
    with open(bvh_file_path, 'r') as f:
        mocap = bvh.Bvh(f.read())
    # Get channel information for all joints in advance
    joint_channels = {joint: mocap.joint_channels(joint) for joint in joints}

    # Get raw data for all frames
    all_frames = np.array(mocap.frames, dtype=np.float32)

    # Manually calculate the starting index of each joint's channels
    channel_indices = []
    joint_order = mocap.get_joints_names()
    index = 0
    for joint in joint_order:
        if joint in joints:
            num_channels = 3
            channel_indices.extend(range(index, index + num_channels))
        index += len(mocap.joint_channels(joint))

    # Extract the required channel data
    features = all_frames[:, channel_indices]

    # 处理数据，使用 np.float16 类型
    frames_array = np.array(features, dtype=np.float64).T
    frames_array[np.isnan(frames_array)] = 0
    frames_array[np.isinf(frames_array)] = 1

    # 检查样本数量
    n_samples = frames_array.shape[0]
    if n_samples < 2:
        # 样本数量不足，直接返回原始数据展平后的结果
        features = frames_array.flatten()
    else:
        # 初始化PCA对象，设置主成分个数为1
        pca = PCA(n_components=1)
        # 使用PCA对象对数据进行拟合和转换，得到降维后的数据
        reduced_frames = pca.fit_transform(frames_array)
        features = reduced_frames.flatten()

    # 统一特征向量长度为 num_channels
    if len(features) != 1000:
        x = np.linspace(0, 1, len(features))
        x_new = np.linspace(0, 1, 1000)
        f = interp1d(x, features)
        features = f(x_new)

    # 再次检查是否有 NaN 值
    if np.isnan(features).any():
        # 处理 NaN 值，这里简单地将其替换为 0
        features = np.nan_to_num(features)

    return features

def extract_features_all(bvh_file_path):
    with open(bvh_file_path, 'r') as f:
        mocap = bvh.Bvh(f.read())
    frames = mocap.frames
    # 处理数据，使用 np.float16 类型
    frames_array = np.array(frames, dtype=np.float64)
    frames_array[np.isnan(frames_array)] = 0
    frames_array[np.isinf(frames_array)] = 1

    # 检查样本数量
    n_samples = frames_array.shape[0]
    if n_samples < 2:
        # 样本数量不足，直接返回原始数据展平后的结果
        features = frames_array.flatten()
    else:
        # 初始化PCA对象，设置主成分个数为1
        pca = PCA(n_components=1)
        # 使用PCA对象对数据进行拟合和转换，得到降维后的数据
        reduced_frames = pca.fit_transform(frames_array)
        features = reduced_frames.flatten()

    # 统一特征向量长度为 num_channels
    if len(features) != 1000:
        x = np.linspace(0, 1, len(features))
        x_new = np.linspace(0, 1, 1000)
        f = interp1d(x, features)
        features = f(x_new)

    features[np.isnan(features)] = 0
    features[np.isinf(features)] = 1
    
    return features

def extract_features_tsne(bvh_file_path):
    """
    Extract feature vectors from a BVH file.
    :param bvh_file_path: Path to the BVH file.
    :return: Feature vector.
    """
    with open(bvh_file_path, 'r') as f:
        mocap = bvh.Bvh(f.read())
    frames = mocap.frames
    num_channels = len(frames[0])

    # 处理数据溢出问题，使用更高精度的数据类型
    frames_array = np.array(frames, dtype=np.float64)

    # 处理 NaN 和 inf 值
    frames_array = np.nan_to_num(frames_array)

    # 检查样本数量
    n_samples = frames_array.shape[0]
    if n_samples < 2:
        # 样本数量不足，直接返回原始数据展平后的结果
        features = frames_array.flatten()
    else:
        # 动态调整 perplexity 参数
        perplexity = min(30, n_samples - 1)
        # 使用TSNE将frames_array降维到一维
        tsne = TSNE(n_components=1, perplexity=perplexity)
        reduced_frames = tsne.fit_transform(frames_array)
        features = reduced_frames.flatten()

    # 统一特征向量长度为 num_channels
    if len(features) != num_channels:
        x = np.linspace(0, 1, len(features))
        x_new = np.linspace(0, 1, num_channels)
        f = interp1d(x, features)
        features = f(x_new)

    # 再次检查是否有 NaN 值
    if np.isnan(features).any():
        # 处理 NaN 值，这里简单地将其替换为 0
        features = np.nan_to_num(features)

    return features


def calculate_diversity(folder_path, subset_size):
    """
    计算多样性指标 Diversity
    :param folder_path: 包含所有动作样本 BVH 文件的文件夹路径
    :param subset_size: 子集大小
    :return: Diversity 值
    """
    # 从文件夹中读取所有 BVH 文件
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.bvh')]

    if len(all_files) < 2 * subset_size:
        raise ValueError("样本数量不足，无法抽取两个指定大小的子集")

    # 随机抽取两个大小相同的子集
    subset1 = random.sample(all_files, subset_size)
    remaining_files = [file for file in all_files if file not in subset1]
    subset2 = random.sample(remaining_files, subset_size)

    # 提取特征向量
    features1 = [extract_features(sample) for sample in subset1]
    features2 = [extract_features(sample) for sample in subset2]

    # 检查特征向量长度是否一致
    expected_length = len(features1[0])
    for feature in features1 + features2:
        if len(feature) != expected_length:
            raise ValueError("提取的特征向量长度不一致")

    # 计算对应特征向量的欧氏距离并累加
    total_distance = 0
    for v1, v2 in zip(features1, features2):
        distance = np.linalg.norm(v1 - v2)
        total_distance += distance

    # 计算 Diversity 值
    diversity = total_distance / subset_size
    return diversity

def calculate_diversity_all(folder_path, subset_size):
    """
    计算多样性指标 Diversity
    :param folder_path: 包含所有动作样本 BVH 文件的文件夹路径
    :param subset_size: 子集大小
    :return: Diversity 值
    """
    # 从文件夹中读取所有 BVH 文件
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.bvh')]

    if len(all_files) < 2 * subset_size:
        raise ValueError("样本数量不足，无法抽取两个指定大小的子集")

    # 随机抽取两个大小相同的子集
    subset1 = random.sample(all_files, subset_size)
    remaining_files = [file for file in all_files if file not in subset1]
    subset2 = random.sample(remaining_files, subset_size)

    # 提取特征向量
    features1 = [extract_features_all(sample) for sample in subset1]
    features2 = [extract_features_all(sample) for sample in subset2]

    # 检查特征向量长度是否一致
    expected_length = len(features1[0])
    for feature in features1 + features2:
        if len(feature) != expected_length:
            raise ValueError("提取的特征向量长度不一致")

    # 计算对应特征向量的欧氏距离并累加
    total_distance = 0
    for v1, v2 in zip(features1, features2):
        distance = np.linalg.norm(v1 - v2)
        total_distance += distance

    # 计算 Diversity 值
    diversity = total_distance / subset_size
    return diversity

def get_subfolders(directory):
    """
    获取指定目录下的所有子文件夹路径。
    
    :param directory: 要搜索的目录
    :return: 子文件夹路径列表
    """
    subfolders = []
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            subfolders.append(subfolder_path)
    return subfolders
# 示例使用
if __name__ == "__main__":
    # 修改为文件夹路径列表
  
    folder_paths=get_subfolders("tested-zeggs")+(get_subfolders("tested-beat"))
    subset_size = 10  # 子集大小
    
    for folder_path in folder_paths:
        try:
            diversity = calculate_diversity_all(folder_path, subset_size)
            print(f"文件夹 {folder_path} 的 all Diversity 值: {diversity}")
        except ValueError as e:
            print(f"处理文件夹 {folder_path} 时出错: {e}")

    for folder_path in folder_paths:
        try:
            diversity = calculate_diversity(folder_path, subset_size)
            print(f"文件夹 {folder_path} 的 Diversity 值: {diversity}")
        except ValueError as e:
            print(f"处理文件夹 {folder_path} 时出错: {e}")