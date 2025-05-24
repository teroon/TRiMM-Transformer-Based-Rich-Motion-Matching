import numpy as np
from scipy.linalg import sqrtm
import bvh
import os
from sklearn.decomposition import PCA  # 用于降维以可视化
from sklearn.manifold import TSNE  # 用于降维以可视化
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor

def get_filenames(directory):
    """
    获取指定目录下所有文件的文件名（不包括扩展名）。
    
    :param directory: 要搜索的目录
    :return: 文件名列表（不含扩展名）
    """
    filenames = []  # 用于存储文件名
    for filename in os.listdir(directory):  # 遍历目录中的文件和子目录
        full_path = os.path.join(directory, filename)  # 获取完整路径
        if filename.endswith('.bvh'):  # 检查是否为文件
            name = full_path # 去掉扩展名
            filenames.append(name)
    
    sorted_filenames = sorted(filenames)  # 使用 sorted() 函数按字典序排序[^20^][^27^]
    
    return sorted_filenames

def load_bvh_features_raw(file_path, selected_joints=None):
    """
    Load feature vectors from a BVH file
    :param file_path: Path to the BVH file
    :param selected_joints: List of joint names to include in the feature vectors. If None, all joints are included.
    :return: Array of feature vectors
    """
    with open(file_path, 'r') as f:
        mocap = bvh.Bvh(f.read())
    if selected_joints is None:
        joints = mocap.get_joints_names()
    else:
        joints = selected_joints
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



        # Perform interpolation

    return np.nan_to_num(features)

def load_bvh_features_2d(file_path, selected_joints=None):
    """
    Load feature vectors from a BVH file
    :param file_path: Path to the BVH file
    :param selected_joints: List of joint names to include in the feature vectors. If None, all joints are included.
    :return: Array of feature vectors
    """
    with open(file_path, 'r') as f:
        mocap = bvh.Bvh(f.read())
    if selected_joints is None:
        joints = mocap.get_joints_names()
    else:
        joints = selected_joints
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
    # 使用 t-SNE 进行降维到二维
    try:
        tsne = TSNE(n_components=2, random_state=42)
        features = tsne.fit_transform(np.nan_to_num(features))
    except ValueError as e:
        features.reshape(75, -1)

    return np.nan_to_num(features)

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
        sqrt_cov_product = sqrtm(cov_product)
        if np.iscomplexobj(sqrt_cov_product):
            sqrt_cov_product = sqrt_cov_product.real
    except np.linalg.LinAlgError:
        # 处理奇异矩阵的情况，增加正则化项
        eps = 1e-4  # Increase the regularization term
        eye = np.eye(sigma_g.shape[0]) * eps
        try:
            sqrt_cov_product = sqrtm((sigma_g + eye).dot(sigma_r + eye))
            if np.iscomplexobj(sqrt_cov_product):
                sqrt_cov_product = sqrt_cov_product.real
        except np.linalg.LinAlgError:
            # 若仍然失败，使用特征值分解计算矩阵平方根
            eigenvalues, eigenvectors = np.linalg.eigh(cov_product)
            # Check for numerical instability
            if np.any(eigenvalues < -1e-8):
                print("Matrix is numerically not positive semi-definite.")
            # 将负特征值设为 0，确保矩阵半正定
            eigenvalues = np.maximum(eigenvalues, 0)
            sqrt_eigenvalues = np.sqrt(eigenvalues)
            sqrt_cov_product = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T


    
    # 计算矩阵的迹
    trace_term = np.trace(sigma_g + sigma_r - 2 * np.nan_to_num(sqrt_cov_product))
    
    # 计算 FGD
    fgd = abs(l2_norm_squared + trace_term)
    return fgd


def get_subfolders(directory):
    """
    获取指定目录下的所有次一级子文件夹路径。
    
    :param directory: 要搜索的目录
    :return: 次一级子文件夹路径列表
    """
    subfolders = []
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        if os.path.isdir(entry_path):
            subfolders.append(entry_path)
    return subfolders


# 读取文件夹中的所有 BVH 文件
generated_folders = get_subfolders("tested-beat")
real_folder = "beat_data/Beat_1"
columns_to_read =['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'HeadEnd', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase']

for generated_folder in generated_folders:
    generated_bvhs = get_filenames(generated_folder)
    real_bvhs = get_filenames(real_folder)

    total_fgd = 0
    total_fgd_raw=0

    for i in range(len(real_bvhs)):
        generated_bvh_file = generated_bvhs[i]
        real_bvh_file = real_bvhs[i]
        if  os.path.exists(generated_bvh_file):
            fgd=0#calculate_fgd(load_bvh_features_2d(generated_bvh_file,columns_to_read),load_bvh_features_2d(real_bvh_file,columns_to_read))
            fgd_raw=calculate_fgd(load_bvh_features_raw(generated_bvh_file,columns_to_read),load_bvh_features_raw(real_bvh_file,columns_to_read))
            total_fgd+=fgd
            total_fgd_raw+=fgd_raw
            print(f"{generated_bvh_file}: {fgd}")
            print(f"{generated_bvh_file}: {fgd_raw}")
            with open("fgd_beat.txt", "a+", encoding="utf-8") as file:
                file.write(f"{generated_bvh_file}: {fgd}"+'\n')
                file.write(f"{generated_bvh_file}: {fgd_raw}"+'\n')
    print(f"Total FGD value: {total_fgd}")
    print(f"Total FGD raw value: {total_fgd_raw}")
    with open("fgd_beat.txt", "a+", encoding="utf-8") as file:
        file.write(f"Total FGD value: {total_fgd}"+'\n')
        file.write(f"Total FGD raw value: {total_fgd_raw}"+'\n')
