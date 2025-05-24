import numpy as np
import torch
import asyncio
import websockets
import json
import ModelDefine
import Searcher
import Wav2VecInferencePytorch
import examples.extract_features_cls as BertExtraction
import time
import os
import wave
import glob
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA  # 用于降维以可视化
from scipy.linalg import sqrtm
import bvh

class Inference:
    def __init__(self):
        self.text_sequence = []
        self.audio_sequence = []
        self.action_index = 0
        self.action_indices = 0
        self.action_lengths = 5

        self.wav2vec_model = Wav2VecInferencePytorch.Wav2VecInference(
            "Data/wav2vec_960",
            "cuda"
        )
        self.wav2vec_model.load_model()

        self.bert_model, self.bert_tokenize = BertExtraction.LoadModel()

        # 假设你的模型类名为MultimodalModel
        self.model = ModelDefine.MultiModalGPT()
        self.model.load_state_dict(torch.load('Data/transformer-5-8-4.pth', map_location="cuda", weights_only=True))

        self.wav2vec_model.model.to("cuda")
        self.bert_model.to("cuda")
        self.model.to("cuda")

        self.searcher = Searcher.Searcher(
            "./Data/"
        )

    def extract_text_audio_feature_by_file(
            self,
            audio_path,
            text,
            window_size=8
    ):
        if len(self.text_sequence) >= window_size:
            if window_size==1:
                self.text_sequence = []
                self.audio_sequence = []
            else:
                self.text_sequence = self.text_sequence[-window_size+1:]
                self.audio_sequence = self.audio_sequence [-window_size+1:]

        data_reduced = self.wav2vec_model.read_single_wav_by_file(audio_path)
        text_feature = BertExtraction.Extract(text, self.bert_model, self.bert_tokenize,"cuda")

        self.text_sequence.append(text_feature)
        self.audio_sequence.append(data_reduced)

    def extract_text_audio_feature_by_data(
            self,
            audio,
            text,
            window_size=8
    ):
        if len(self.text_sequence) >= window_size:
            if window_size==1:
                self.text_sequence = []
                self.audio_sequence = []
            else:
                self.text_sequence = self.text_sequence[-window_size+1:]
                self.audio_sequence = self.audio_sequence [-window_size+1:]

        data_reduced = self.wav2vec_model.read_single_wav_by_data(audio)
        text_feature = BertExtraction.Extract(text, self.bert_model, self.bert_tokenize)

        self.text_sequence.append(text_feature)
        self.audio_sequence.append(data_reduced)

    def inference_action_feature(
            self,wav_length
    ):
        audio_array = np.expand_dims(np.asarray(self.audio_sequence), axis=0)
        text_array = np.expand_dims(np.asarray(self.text_sequence), axis=0)

        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio_array).float().to("cuda").permute(0, 1, 2)
            text_tensor = torch.from_numpy(text_array).float().to("cuda").permute(0, 1, 2)

        # 加载保存的权重

        self.model.eval()

        # 现在你可以使用加载了权重的模型进行推理
        # 例如，获取模型的输出
        with torch.no_grad():
            action_features = self.model(text_tensor , audio_tensor).cpu().numpy() 
            action_features = action_features.squeeze(0)
            self.action_index, self.action_lengths = self.searcher.search_vector_index(action_features,wav_length)

    async def inference_by_file(
            self,
            audio_path,
            text
    ):
        self.extract_text_audio_feature_by_file(
            audio_path,
            text
        )
        wav_length = get_wav_duration(audio_path)
        self.inference_action_feature(wav_length)
        # 初始化一个变量来存储找到的索引

        return self.action_index, wav_length

    async def inference_by_data(
            self,
            audio,
            text
    ):
        self.extract_text_audio_feature_by_data(
            audio,
            text
        )
        wav_length = 3
        self.inference_action_feature(wav_length)
        # 初始化一个变量来存储找到的索引

        return self.action_index, wav_length

def get_wav_duration(wav_path):
    with wave.open(wav_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration

def read_txt_file(file_path):
    """
    读取指定的 .txt 文件并返回其内容为字符串。

    :param file_path: .txt 文件的路径
    :return: 文件内容的字符串
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: 文件未找到，请检查路径是否正确：{file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return None

def find_files_with_string(directory, target_string):
    """
    在指定目录下查找所有包含目标字符串的文件路径，并分别返回扩展名为 .txt 和 .wav 的文件路径列表。
    不尝试打开文件，仅根据文件名和扩展名进行分类。

    :param directory: 要搜索的目录
    :param target_string: 目标字符串
    :return: 包含目标字符串的 .txt 文件路径列表和 .wav 文件路径列表
    """
    txt_files = []  # 用于存储匹配的 .txt 文件路径
    wav_files = []  # 用于存储匹配的 .wav 文件路径

    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件名是否包含目标字符串
            if target_string in file:
                file_path = os.path.join(root, file)  # 获取文件的完整路径
                _, extension = os.path.splitext(file)  # 分离文件名和扩展名
                if extension.lower() == ".txt":
                    txt_files.append(file_path)
                elif extension.lower() == ".wav":
                    wav_files.append(file_path)
    sorted_txt_files = sorted(txt_files) 
    sorted_wav_files = sorted(wav_files) 
    return sorted_txt_files, sorted_wav_files


async def main():
    inference_instance = Inference()

    async with websockets.connect(
        "ws://127.0.0.1:8086",
        ping_interval=30,  # 每20秒发送一次心跳
        ping_timeout=1000    # 心跳超时时间为10秒
    ) as websocket:
        dataset="beat"
        txt_files = glob.glob(os.path.join("Data/"+dataset, '*.txt'))
        wav_files = glob.glob(os.path.join("Data/"+dataset, '*.wav'))
        txt_set = set()
        for i in range (len(txt_files)):
            txt_set.add(os.path.basename(txt_files[i]).rsplit('_', 1)[0])
        filenames = list(txt_set)
        filenames.sort()
        for filename in filenames:
            txt_files, wav_files = find_files_with_string('Data/'+dataset, filename)
            await websocket.send(json.dumps({"recording": True}))
            response = await websocket.recv()
            print(f"收到回复: {response}")
            for i in range (len(txt_files)):
                audio_path = wav_files[i]
                text = txt_files[i]
                filename_send= filename

                action_index, wav_length = await inference_instance.inference_by_file(audio_path, text)
                result = {"filename" : filename_send ,"action_index": int(action_index), "wav_length": wav_length}
                await websocket.send(json.dumps(result))
                response = await websocket.recv()
                print(f"收到回复: {response}")
                if wav_length>1:
                    time.sleep(wav_length)
                else :
                    time.sleep(1)
            await websocket.send(json.dumps({"recording": False, "save_recording": True}))
            response = await websocket.recv()
            print(f"收到回复: {response}")
            time.sleep(3)

        dataset="zeggs"
        txt_files = glob.glob(os.path.join("Data/"+dataset, '*.txt'))
        wav_files = glob.glob(os.path.join("Data/"+dataset, '*.wav'))
        txt_set = set()
        for i in range (len(txt_files)):
            txt_set.add(os.path.basename(txt_files[i]).rsplit('_', 1)[0])
        filenames = list(txt_set)
        filenames.sort()
        for filename in filenames:
            txt_files, wav_files = find_files_with_string('Data/'+dataset, filename)
            await websocket.send(json.dumps({"recording": True}))
            response = await websocket.recv()
            print(f"收到回复: {response}")
            for i in range (len(txt_files)):
                audio_path = wav_files[i]
                text = txt_files[i]
                filename_send= filename

                action_index, wav_length = await inference_instance.inference_by_file(audio_path, text)
                result = {"filename" : filename_send ,"action_index": int(action_index), "wav_length": wav_length}
                await websocket.send(json.dumps(result))
                response = await websocket.recv()
                print(f"收到回复: {response}")
                if wav_length>1:
                    time.sleep(wav_length)
                else :
                    time.sleep(1)
            await websocket.send(json.dumps({"recording": False, "save_recording": True}))
            response = await websocket.recv()
            print(f"收到回复: {response}")
            time.sleep(3)



    
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

    num_channels = features.shape[1]

    # Calculate the minimum and maximum values
    min_val = np.min(features)
    max_val = np.max(features)

    # Normalize the array
    features = (features - min_val) / (max_val - min_val)

    # Initialize the output feature array
    output_features = np.zeros((num_channels, 1000))

    # Define the x-axis range of the original data
    x = np.linspace(0, 1, len(features))

    # Define the x-axis range after interpolation
    x_new = np.linspace(0, 1, 1000)

    # Interpolate each channel
    for i in range(num_channels):
        # Check if the current channel contains invalid data
        if np.any(np.isnan(features[:, i])) or np.any(np.isinf(features[:, i])):
            #raise ValueError(f"Invalid data in channel {i}: contains NaN or Inf")
            features=np.nan_to_num(features)

        # Create an interpolation function, allowing extrapolation to avoid boundary issues
        f = interp1d(x, features[:, i], kind='linear', bounds_error=False, fill_value="extrapolate")

        # Perform interpolation
        output_features[i] = f(x_new)

    return np.nan_to_num(output_features)

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
    features = all_frames[:, channel_indices].T
    features[np.isnan(features)] = 0
    features[np.isinf(features)] = 1
    pca = PCA(n_components=10)
    
    # 使用PCA对象对数据进行拟合和转换，得到降维后的数据
    X_reduced = pca.fit_transform(features)

    return np.nan_to_num(X_reduced)

def load_bvh_features(file_path, selected_joints):
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
    features = all_frames[:, channel_indices].T
    features[np.isnan(features)] = 0
    features[np.isinf(features)] = 1
    pca = PCA(n_components=10)
    df_length_div_120 = features.shape[1] / 120
    # 使用PCA对象对数据进行拟合和转换，得到降维后的数据
    X_reduced = pca.fit_transform(features)
    features = X_reduced.flatten()
    return np.nan_to_num(features),df_length_div_120

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
        # 处理奇异矩阵的情况
        eye = np.eye(sigma_g.shape[0]) * 1e-6
        sqrt_cov_product = sqrtm((sigma_g + eye).dot(sigma_r + eye))
        if np.iscomplexobj(sqrt_cov_product):
            sqrt_cov_product = sqrt_cov_product.real
    
    # 计算矩阵的迹
    trace_term = np.trace(sigma_g + sigma_r - 2 * sqrt_cov_product)
    
    # 计算 FGD
    fgd = abs(l2_norm_squared + trace_term)
    return fgd

if __name__ == "__main__":
    asyncio.run(main())