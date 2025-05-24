import torch

import wave
import numpy as np
from sklearn.decomposition import PCA
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)

class Wav2VecInference:
    def __init__(
            self,
            model_path,
            device
    ):
        self.model = None
        self.feature_extractor = None
        self.device = device
        self.model_path = model_path

    def load_model(
            self
    ):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_path)
        self.model = Wav2Vec2Model.from_pretrained(self.model_path)

        self.model = self.model.half()

        self.model.eval()

    def wav2vec(
            self,
            wav
    ):
        input_values = self.feature_extractor(wav, return_tensors="pt").input_values
        input_values = input_values.half()
        input_values = input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            output_tensor = outputs.extract_features.data.cpu().numpy()
            # print(outputTensor[0].shape)
        return output_tensor[0]

    def read_single_wav_by_file(
            self,
            file_path
    ):
        wav_file_path = file_path
        with wave.open(wav_file_path, 'rb') as wav_file:
            # 获取音频参数
            n_channels, sample_width, framerate, n_frames, comptype, compname = wav_file.getparams()

            # 读取整个音频文件的数据
            frames = wav_file.readframes(n_frames)

            # 将帧转换为音频样本
            audio_samples = np.frombuffer(frames, dtype=np.int16)
            pca = PCA(n_components=4)

            feature_vector = self.wav2vec(audio_samples)
            data_reduced = pca.fit_transform(np.nan_to_num(feature_vector.T, nan=0))

        return data_reduced.flatten()
    '''
    def read_single_wav_by_file(self,file_path):
        wav_file_path = file_path
        with wave.open(wav_file_path, 'rb') as wav_file:
            # 获取音频参数
            n_channels, sample_width, framerate, n_frames, comptype, compname = wav_file.getparams()

            # 读取整个音频文件的数据
            frames = wav_file.readframes(n_frames)

            # 将帧转换为音频样本
            audio_samples = np.frombuffer(frames, dtype=np.int16)

            feature_vector = self.wav2vec(audio_samples)
            # 转换为 PyTorch 张量并添加批次和通道维度
            feature_tensor = torch.from_numpy(feature_vector).unsqueeze(0).unsqueeze(0).float()
            # 平均池化操作，池化窗口大小设为 4
            pooled_tensor = torch.nn.functional.avg_pool2d(feature_tensor, kernel_size=(4, 1))
            # 转换回 NumPy 数组并去除多余维度
            pooled_features = pooled_tensor.squeeze(0).squeeze(0).numpy()

        return pooled_features.flatten()
    '''
    def read_single_wav_by_data(
            self,
            wav_file
    ):

        # 获取音频参数
        n_channels, sample_width, framerate, n_frames, comptype, compname = wav_file.getparams()

        # 读取整个音频文件的数据
        frames = wav_file.readframes(n_frames)

        # 将帧转换为音频样本
        audio_samples = np.frombuffer(frames, dtype=np.int16)
        pca = PCA(n_components=4)
        # 应用滑动窗口
        feature_vector = self.wav2vec(audio_samples)
        data_reduced = pca.fit_transform(np.nan_to_num(feature_vector.T, nan=0))

        return data_reduced.flatten()
