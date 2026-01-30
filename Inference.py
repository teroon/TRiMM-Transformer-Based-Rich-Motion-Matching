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
        self.model.load_state_dict(torch.load('Data/transformer-5-8-4.pth', map_location="cuda"))

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
            self, wav_length, prev_action_feature=None
    ):
        audio_array = np.expand_dims(np.asarray(self.audio_sequence), axis=0)
        text_array = np.expand_dims(np.asarray(self.text_sequence), axis=0)

        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio_array).float().to("cuda").permute(0, 1, 2)
            text_tensor = torch.from_numpy(text_array).float().to("cuda").permute(0, 1, 2)

        self.model.eval()

        with torch.no_grad():
            if prev_action_feature is not None:
                # Convert prev_action_feature to tensor and move to GPU
                prev_action_tensor = torch.from_numpy(prev_action_feature).float().to("cuda").unsqueeze(0)
                action_features, latent_query = self.model(text_tensor, audio_tensor, prev_action_tensor)
            else:
                action_features, latent_query = self.model(text_tensor, audio_tensor)
            action_features = action_features.cpu().numpy() 
            action_features = action_features.squeeze(0)
            self.action_index, self.action_lengths = self.searcher.search_vector_index(action_features, wav_length)

    async def inference_by_file(
            self,
            audio_path,
            text,
            prev_action_bvh_path=None
    ):
        self.extract_text_audio_feature_by_file(
            audio_path,
            text
        )
        wav_length = get_wav_duration(audio_path)
        
        # Load previous action feature from BVH file if provided
        prev_action_feature = None
        if prev_action_bvh_path is not None:
            prev_action_feature, _ = load_bvh_features(prev_action_bvh_path, None)
        
        self.inference_action_feature(wav_length, prev_action_feature)
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
    # 获取Bvh2LiveLink\recorded_frames目录中最新的BVH文件，即上一个输出的动作，作为下一个动作的反馈输入。
    bvh_dir = "Bvh2LiveLink/recorded_frames"
    bvh_files = glob.glob(os.path.join(bvh_dir, "*.bvh"))
    prev_action_bvh_path = None
    if bvh_files:
        # 按修改时间排序，获取最新的文件
        prev_action_bvh_path = max(bvh_files, key=os.path.getmtime)
    else:
        print(f"Warning: No BVH files found in {bvh_dir}")

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
                text = read_txt_file(txt_files[i])
                filename_send= filename

                action_index, wav_length = await inference_instance.inference_by_file(audio_path, text, prev_action_bvh_path)
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


def load_bvh_features(file_path, selected_joints):
    """
    Load feature vectors from a BVH file
    We extract its terminal kinematic features (root velocity, joint rotations, foot contacts) to form the state vector $\mathbf{s}_{t-1}$
    
    :param file_path: Path to the BVH file
    :param selected_joints: List of joint names to include in the feature vectors. If None, all joints are included.
    :return: Array of feature vectors and frame count normalized by 120
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
            num_channels = 3  # Assuming 3 channels per joint (rotation/position)
            channel_indices.extend(range(index, index + num_channels))
        index += len(mocap.joint_channels(joint))

    # Extract the required channel data
    features = all_frames[:, channel_indices].T
    features[np.isnan(features)] = 0
    features[np.isinf(features)] = 1
    
    # Apply PCA for dimensionality reduction to form the state vector s_{t-1}
    pca = PCA(n_components=10)
    # Use PCA to fit and transform the data to get reduced dimensional representation
    X_reduced = pca.fit_transform(features)
    features = X_reduced.flatten()
    
    return np.nan_to_num(features)


if __name__ == "__main__":
    asyncio.run(main())