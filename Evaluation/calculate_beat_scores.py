from absl import app
from absl import flags
from absl import logging
import librosa
import os
from librosa import beat
import torch
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import scipy.signal as scisignal
import glob
import csv
from datetime import datetime
from scipy.interpolate import interp1d
import bvh

# from aist_plusplus.loader import AISTDataset

"""
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'anno_dir', '/mnt/data/aist_plusplus_final/',
    'Path to the AIST++ annotation files.')
flags.DEFINE_string(
    'audio_dir', '/mnt/data/AIST/music/',
    'Path to the AIST wav files.')
flags.DEFINE_string(
    'audio_cache_dir', './data/aist_audio_feats/',
    'Path to cache dictionary for audio features.')
flags.DEFINE_enum(
    'split', 'testval', ['train', 'testval'],
    'Whether do training set or testval set.')
flags.DEFINE_string(
    'result_files', '/mnt/data/aist_paper_results/*.pkl',
    'The path pattern of the result files.')
flags.DEFINE_bool(
    'legacy', True,
    'Whether the result files are the legacy version.')
"""

"""
class SRGR(object):
    def __init__(self, threshold=0.1, joints=47):
        self.threshold = threshold
        self.pose_dimes = joints
        self.counter = 0
        self.sum = 0

    def run(self, results, targets, semantic):
        results = results.reshape(-1, self.pose_dimes, 3)
        targets = targets.reshape(-1, self.pose_dimes, 3)
        semantic = semantic.reshape(-1)
        diff = np.sum(abs(results - targets), 2)
        success = np.where(diff < self.threshold, 1.0, 0.0)
        for i in range(success.shape[0]):
            # srgr == 0.165 when all success, scale range to [0, 1]
            success[i, :] *= semantic[i] * (1 / 0.165)
        rate = np.sum(success) / (success.shape[0] * success.shape[1])
        self.counter += success.shape[0]
        self.sum += (rate * success.shape[0])
        return rate

    def avg(self):
        return self.sum / self.counter
"""


def run_srgr(results, targets, dim):
    counter = 0
    sum = 0
    results = results.reshape(-1, dim, 3)
    targets = targets.reshape(-1, dim, 3)
    targets = targets[:results.shape[0]]
    # print(results.shape)
    # print(targets.shape)
    # raise ValueError
    # semantic = semantic.reshape(-1)
    diff = np.sum(abs(results - targets), 2)
    # print("min_diff = ", np.min(diff))
    success = np.where(diff < 10, 1.0, 0.0)
    for i in range(success.shape[0]):
        # srgr == 0.165 when all success, scale range to [0, 1]
        success[i, :] *= 1 * (1 / 0.165)

        # print(success[i, :])
        # print(success[i, :] * 1 * (1 / 0.165))
    rate = np.sum(success) / (success.shape[0] * success.shape[1])

    counter += success.shape[0]
    sum += (rate * success.shape[0])
    print("rate = ", rate)
    return rate


def cache_audio_features(filename):
    FPS = 20
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    EPS = 1e-6

    def _get_tempo(audio_name):
        """Get tempo (BPM) for a music by parsing music name."""
        if audio_name[0:3] in ['mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB']:
            return 120
        elif audio_name[0:3] == 'mHO':
            return 120
        else:
            return 120

    audio_names = [filename]

    for audio_name in audio_names:
        save_path = os.path.join(f"{audio_name}.npy")
        if os.path.exists(save_path):
            continue
        data, _ = librosa.load(audio_name, sr=SR)
        envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
        mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
        chroma = librosa.feature.chroma_cens(
            y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T  # (seq_len, 12)

        peak_idxs = librosa.onset.onset_detect(
            onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0  # (seq_len,)

        tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
            start_bpm=_get_tempo(audio_name), tightness=100)
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0  # (seq_len,)

        audio_feature = np.concatenate([
            envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]
        ], axis=-1)
        # np.save(save_path, audio_feature)
        return audio_feature


def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


def recover_to_axis_angles(motion):
    batch_size, seq_len, dim = motion.shape
    assert dim == 225
    transl = motion[:, :, 6:9]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, 9:], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl


def recover_motion_to_keypoints(motion, smpl_model):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()[:, :24, :]  # (seq_len, 24, 3)
    return keypoints3d


def read_npy_to_joints(data):
    #data = np.load(filename)
    # print(data.shape)
    output = []
    for frame in data:
        # print(frame)
        frame_out = []
        i = 0
        xyz = []
        for num in frame:
            if i < 3:
                i += 1
                xyz.append(num)
                if len(frame_out) == 55 and len(xyz) == 3:
                    frame_out.append(xyz)
            else:
                # print(xyz)
                frame_out.append(xyz)
                i = 1
                xyz = [num]
        output.append(frame_out)
    output_np = np.array(output)
    # print(output_np.shape)
    return output_np


def motion_peak_onehot(joints):
    """Calculate motion beats.
    Kwargs:
        joints: [nframes, njoints, 3]
    Returns:
        - peak_onhot: motion beats.
    """
    # Calculate velocity.
    velocity = np.zeros_like(joints, dtype=np.float32)
    velocity[1:] = joints[1:] - joints[:-1]
    velocity_norms = np.linalg.norm(velocity, axis=2)
    envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

    # Find local minima in velocity -- beats
    peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=10)  # 10 for 60FPS
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1

    # # Second-derivative of the velocity shows the energy of the beats
    # peak_energy = np.gradient(np.gradient(envelope)) # (seq_len,)
    # # optimize peaks
    # peak_onehot[peak_energy<0.001] = 0
    return peak_onehot


def alignment_score(music_beats, motion_beats, sigma=3):
    """Calculate alignment score between music and motion."""
    if motion_beats.sum() == 0:
        return 0.0
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]
    score_all = []

    for motion_beat_idx in motion_beat_idxs:
        try:
            dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
            ind = np.argmin(dists)
            score = np.exp(- dists[ind] ** 2 / 2 / sigma ** 2)
            score_all.append(score)
        except ValueError as err:
            pass
    return sum(score_all) / len(score_all)

def load_bvh_features(file_path,Audiolenth):
    """
    Load feature vectors from a BVH file
    :param file_path: Path to the BVH file
    :param selected_joints: List of joint names to include in the feature vectors. If None, all joints are included.
    :return: Array of feature vectors
    """
    with open(file_path, 'r') as f:
        mocap = bvh.Bvh(f.read())
        joints = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'HeadEnd', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase']
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
    output_features = np.zeros((num_channels, Audiolenth))

    # Define the x-axis range of the original data
    x = np.linspace(0, 1, len(features))

    # Define the x-axis range after interpolation
    x_new = np.linspace(0, 1, Audiolenth)

    # Interpolate each channel
    for i in range(num_channels):
        # Check if the current channel contains invalid data
        if np.any(np.isnan(features[:, i])) or np.any(np.isinf(features[:, i])):
            #raise ValueError(f"Invalid data in channel {i}: contains NaN or Inf")
            np.nan_to_num(features[:, i], copy=False)

        # Create an interpolation function, allowing extrapolation to avoid boundary issues
        f = interp1d(x, features[:, i], kind='linear', bounds_error=False, fill_value="extrapolate")

        # Perform interpolation
        output_features[i] = f(x_new)

    return output_features.T

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

def get_filename_without_extension(absolute_path):
    filename = os.path.basename(absolute_path)  # 获取文件名
    filename_without_extension = os.path.splitext(filename)[0]  # 去掉后缀名
    return filename_without_extension

if __name__ == '__main__':

    generated_folders = get_subfolders("tested-beat")
    for generated_folder in generated_folders:
        audio_folder = 'beat_data/audio'   
        npy_folder = generated_folder  # npy文件夹路径
        csv_folder = 'beat_data/beatalign' 
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)

        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        csv_filename = f'results_{os.path.basename(generated_folder)}.csv'  # 生成带时间戳的CSV文件名

        all_audio_files = glob.glob(os.path.join(audio_folder, '*wav'))
        all_npy_files = glob.glob(os.path.join(npy_folder, '*.bvh'))

        results = []
        total_score = []
        for audio_file, npy_file in zip(all_audio_files, all_npy_files):
                        audio_feature = cache_audio_features(audio_file)
                        keypoints3d = read_npy_to_joints(load_bvh_features(npy_file,audio_feature.shape[0]))
                        motion_beats = motion_peak_onehot(keypoints3d)

                        try:
                            audio_beats = audio_feature[:keypoints3d.shape[0], -1]
                            beat_score = alignment_score(audio_beats, motion_beats, sigma=3)
                            results.append([os.path.basename(audio_file), os.path.basename(npy_file), beat_score])
                            total_score.append(beat_score)                       
                        except ValueError as e:
                            pass
                        except ZeroDivisionError as e:
                            pass
        
        average_score = np.mean(np.array(total_score))

        with open(os.path.join(csv_folder, csv_filename), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Audio File', 'NPY File', 'Alignment Score'])
            writer.writerows(results)
            writer.writerows([str(average_score)])
            print('Average BeatAlign: '+str(average_score))


    generated_folders = get_subfolders("tested-zeggs")
    for generated_folder in generated_folders:
        audio_folder = 'zeggs_data/audio'   
        npy_folder = generated_folder  # npy文件夹路径
        csv_folder = 'zeggs_data/beatalign' 
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)

        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        csv_filename = f'results_{os.path.basename(generated_folder)}.csv'  # 生成带时间戳的CSV文件名

        all_audio_files = glob.glob(os.path.join(audio_folder, '*wav'))
        all_npy_files = glob.glob(os.path.join(npy_folder, '*.bvh'))

        results = []
        total_score = []
        for audio_file, npy_file in zip(all_audio_files, all_npy_files):
                        audio_feature = cache_audio_features(audio_file)
                        keypoints3d = read_npy_to_joints(load_bvh_features(npy_file,audio_feature.shape[0]))
                        motion_beats = motion_peak_onehot(keypoints3d)

                        try:
                            audio_beats = audio_feature[:keypoints3d.shape[0], -1]
                            beat_score = alignment_score(audio_beats, motion_beats, sigma=3)
                            results.append([os.path.basename(audio_file), os.path.basename(npy_file), beat_score])
                            total_score.append(beat_score)                       
                        except ValueError as e:
                            pass
                        except ZeroDivisionError as e:
                            pass
        
        average_score = np.mean(np.array(total_score))

        with open(os.path.join(csv_folder, csv_filename), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Audio File', 'NPY File', 'Alignment Score'])
            writer.writerows(results)
            writer.writerows([str(average_score)])
            print('Average BeatAlign: '+str(average_score))


 # if __name__ == '__main__':

#     audio_files_list = ['originAudio']
#     base_path = 'originNpy'

#     # 初始化一个空列表来存储子文件夹路径
#     motion_file_list = []

#     # 遍历基础路径下的所有子文件夹和文件
#     for root, dirs, files in os.walk(base_path):
#         # 只添加子文件夹路径到列表中（忽略文件）
#         for dir_name in dirs:
#             # 拼接成完整的子文件夹路径
#             dir_path = os.path.join(root, dir_name)
#             # 将子文件夹路径添加到列表中（如果需要，可以添加额外的检查或过滤）
#             motion_file_list.append(dir_path)


#     for audio_file_address in audio_files_list:

#         all_audio_files = glob.glob(audio_file_address)
#         for motion_file_address in motion_file_list:
#             all_motion_files = glob.glob(motion_file_address + '/*.npy')
#             # 每个动作文件夹
#             audio_file_list = []
#             motion_file_list = []
#             total_score = []
#             for audio_file in all_audio_files:  # 每个音频文件
#                 for motion_file in all_motion_files:
#                     if get_filename_without_extension(audio_file) in motion_file or os.path.basename(motion_file)[:-6] in get_filename_without_extension(audio_file):
#                         audio_file_list.append(audio_file)
#                         motion_file_list.append(motion_file)


#             print(
#                 f'----- Evaluate Beat Align on: '
#                 f'{motion_file_address[len("E:/ChineseHost_Generate_Result/"):]} with '
#                 f'{len(audio_file_list)} '
#                 f'files -----')
#             # try:
#             if len(audio_file_list)!=0:
#                 for audio_file, motionfile in zip(audio_file_list, motion_file_list):
#                     keypoints3d = read_npy_to_joints(motionfile)
#                     motion_beats = motion_peak_onehot(keypoints3d)
#                     audio_feature = cache_audio_features(audio_file)
#                     try:
#                         audio_beats = audio_feature[:keypoints3d.shape[0], -1]
#                         beat_score = alignment_score(audio_beats, motion_beats, sigma=3)
#                         total_score.append(beat_score)
#                     except ValueError as e:
#                         pass
#                     except ZeroDivisionError as e:
#                         pass
#                 print('Max: ', np.max(total_score), '  Min: ', np.min(total_score), '  std: ', np.std(total_score), '  Len: ', len(total_score))
#                 print('Average BeatAlign: ', np.mean(np.array(total_score)), '\n')要把这段逻辑改成，