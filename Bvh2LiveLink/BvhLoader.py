import asyncio
import websockets
import json
import numpy as np
import bvh
import os
import time
def get_bvh_files(bvh_folder):
        return [os.path.join(bvh_folder, f) for f in os.listdir(bvh_folder) if f.endswith('.bvh')]

def load_bvh_features(file_path):
    """
    Load feature vectors from a BVH file
    :param file_path: Path to the BVH file
    :param selected_joints: List of joint names to include in the feature vectors. If None, all joints are included.
    :return: Array of feature vectors
    """
    with open(file_path, 'r') as f:
        mocap = bvh.Bvh(f.read())

    # Get raw data for all frames
    all_frames = np.array(mocap.frames, dtype=np.float32)

    df_length_div_120 = all_frames.shape[1] / 120



    return df_length_div_120

async def main():
    bvh_folder = 'recorded_frames'
    bvh_files = get_bvh_files(bvh_folder)
    try:
        async with websockets.connect("ws://127.0.0.1:8086", ping_interval=30, ping_timeout=1000) as websocket:
            for i in range(len(bvh_files)):
                # 等待键盘输入 action_index 和 wav_length
                action_index = i
                wav_length = load_bvh_features(bvh_files[i])

                # 发送数据
                result = {"filename" : "key" ,"action_index": action_index, "wav_length": wav_length}
                await websocket.send(json.dumps(result))

                # 接收回复
                response = await websocket.recv()
                print(f"收到回复: {response}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())