import Wav2VecInferencePytorch as InferencePytorch
import wave
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os  
import glob

def read_csv_with_sliding_window(file_path, window_size=60, step_size=30):
    """
    读取CSV文件并应用滑动窗口。
    
    参数:
    file_path (str): CSV文件的路径。
    window_size (int): 窗口大小，单位为行数。
    step_size (int): 窗口滑动的步长，单位为行数。
    
    返回:
    list: 包含每个窗口CSV数据的列表。
    """
    # 读取CSV文件
    action_windows = []
    emotion_Windows=[]
    for root, dirs, files in os.walk(file_path):  
        for file in files:  
            # 示例用法  
            csv_file_path = file_path+ file # 替换为你的CSV文件路径  
            df = pd.read_csv(csv_file_path)
            
            # 应用滑动窗口
            num_windows = (len(df) - window_size) // step_size + 1

            for i in range(num_windows):
                start_row = i * step_size
                end_row = start_row + window_size
                action_vec =df.iloc[start_row:end_row,0:10].max() 
                emotion_vec=df.iloc[start_row:end_row,10:22].max() 
                action_windows.append(action_vec)
                emotion_Windows.append(emotion_vec)
                #print(action_vec.shape)
                #print(emotion_vec.shape)
    return action_windows,emotion_Windows

def read_wav_with_sliding_window(directory, window_size=1, step_size=1):
    """
    读取WAV文件并应用滑动窗口。
    
    参数:
    file_path (str): WAV文件的路径。
    window_size (float): 窗口大小，单位为秒。
    step_size (float): 窗口滑动的步长，单位为秒。
    
    返回:
    list: 包含每个窗口音频数据的列表。
    """                
    audio_windows = []
    deduced_audio_windows = []
    wav_files = glob.glob(os.path.join(directory, '*.wav'))
    
            # 示例用法  
    for wav_file_path in wav_files : # 替换为你的CSV文件路径  
        with wave.open(wav_file_path, 'rb') as wav_file:
            # 获取音频参数
            n_channels, sample_width, framerate, n_frames, comptype, compname = wav_file.getparams()
            
            # 将窗口大小和步长转换为帧数
            window_frames = int(window_size * framerate)
            step_frames = int(step_size * framerate)
            
            # 读取整个音频文件的数据
            frames = wav_file.readframes(n_frames)
            
            # 将帧转换为音频样本
            audio_samples = np.frombuffer(frames, dtype=np.int16)
            pca = PCA(n_components=1)
            # 应用滑动窗口
            num_windows = (n_frames - window_frames) // step_frames + 1

            for i in range(num_windows):
                start_frame = i * step_frames
                end_frame = start_frame + window_frames
                window_samples = audio_samples[start_frame:end_frame]
                FeatureVector=InferencePytorch.wav2vev(window_samples)
                data_reduced = pca.fit_transform(np.nan_to_num(FeatureVector, nan=0))
                deduced_audio_windows.append(data_reduced[:,0]  )
                audio_windows.append(FeatureVector)
    return audio_windows,deduced_audio_windows

#read_wav_with_sliding_window('zeggs/audio/001_Neutral_0_x_1_0.wav')

#read_csv_with_sliding_window('zeggs/zeggs_output1/001_Neutral_0_x_1_0_pos.csv')

def read_single_csv(file_path, window_size=60, step_size=30):
    """
    读取CSV文件并应用滑动窗口。
    
    参数:
    file_path (str): CSV文件的路径。
    window_size (int): 窗口大小，单位为行数。
    step_size (int): 窗口滑动的步长，单位为行数。
    
    返回:
    list: 包含每个窗口CSV数据的列表。
    """
    # 读取CSV文件
    action_windows = []
    emotion_Windows=[]
            # 示例用法  
    csv_file_path = file_path # 替换为你的CSV文件路径  
    df = pd.read_csv(csv_file_path)
    
    # 应用滑动窗口
    num_windows = (len(df) - window_size) // step_size + 1

    for i in range(num_windows):
        start_row = i * step_size
        end_row = start_row + window_size
        action_vec =df.iloc[start_row:end_row,0:10].max() 
        emotion_vec=df.iloc[start_row:end_row,10:22].max() 
        action_windows.append(action_vec)
        emotion_Windows.append(emotion_vec)
        #print(action_vec.shape)
        #print(emotion_vec.shape)
    return action_windows,emotion_Windows

def read_single_wav(file_path, time,window_size=1, step_size=1):
    """
    读取WAV文件并应用滑动窗口。
    
    参数:
    file_path (str): WAV文件的路径。
    window_size (float): 窗口大小，单位为秒。
    step_size (float): 窗口滑动的步长，单位为秒。
    
    返回:
    list: 包含每个窗口音频数据的列表。
    """                

    wav_file_path = file_path # 替换为你的CSV文件路径  
    with wave.open(wav_file_path, 'rb') as wav_file:
        # 获取音频参数
        n_channels, sample_width, framerate, n_frames, comptype, compname = wav_file.getparams()
        
        # 将窗口大小和步长转换为帧数
        window_frames = int(window_size * framerate)
        step_frames = int(step_size * framerate)
        
        # 读取整个音频文件的数据
        frames = wav_file.readframes(n_frames)
        
        # 将帧转换为音频样本
        audio_samples = np.frombuffer(frames, dtype=np.int16)
        pca = PCA(n_components=1)
        # 应用滑动窗口
        num_windows = (n_frames - window_frames) // step_frames + 1

        start_frame = time * step_frames
        end_frame = start_frame + window_frames
        window_samples = audio_samples[start_frame:end_frame]
        FeatureVector=InferencePytorch.wav2vev(audio_samples)
        data_reduced = pca.fit_transform(np.nan_to_num(FeatureVector.T, nan=0))
        print(data_reduced.shape)

    return FeatureVector,data_reduced

def read_folder_wav(directory):
    FeatureVectorArray=[]
    data_reducedArray=[]
    wav_files = glob.glob(os.path.join(directory, '*.wav'))
    
            # 示例用法  
    for wav_file_path in wav_files : # 替换为你的CSV文件路径  
        with wave.open(wav_file_path, 'rb') as wav_file:
            # 获取音频参数
            n_channels, sample_width, framerate, n_frames, comptype, compname = wav_file.getparams()
            
            # 读取整个音频文件的数据
            frames = wav_file.readframes(n_frames)
            
            # 将帧转换为音频样本
            audio_samples = np.frombuffer(frames, dtype=np.int16)
            pca = PCA(n_components=1)
            # 应用滑动窗口
            FeatureVector=InferencePytorch.wav2vev(audio_samples)
            data_reduced = pca.fit_transform(np.nan_to_num(FeatureVector.T, nan=0))
            print(data_reduced.shape)
            #FeatureVectorArray.append(FeatureVector)
            data_reducedArray.append(data_reduced)

    return data_reducedArray