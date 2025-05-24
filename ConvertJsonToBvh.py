import json
import math
import numpy as np
import os

from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(quaternion, order='xyz'):
    r = R.from_quat(quaternion)
    euler_angles = r.as_euler(order, degrees=True)
    x, y, z = euler_angles
    euler_angles = [z, y, x]
    return euler_angles
    
def json_to_bvh(json_data, output_file):
    # 确保数据结构正确
    if not json_data:
        print("输入的JSON数据为空。")
        return

    # 解析骨骼结构
    frames = json_data
    bone_index_to_name = {}
    frame_bones = frames[0]["bvh_source"]
    
    # 构建骨骼索引到名称的映射
    for index, bone_json in enumerate(frame_bones):
        name = bone_json["Name"]
        bone_index_to_name[index] = name

    # 构建骨骼的子骨骼映射
    bone_children = {i: [] for i in range(len(frame_bones))}
    for i, bone in enumerate(frame_bones):
        parent_index = bone["Parent"]
        if parent_index != -1:
            bone_children[parent_index].append(i)

    # 构建BVH文件头部
    bvh_header = "HIERARCHY\n"
    root_bone = frame_bones[0]
    bvh_header += f"ROOT {root_bone['Name']}\n"
    bvh_header += "{\n"
    bvh_header += f"OFFSET {-root_bone['Location'][0]} {-root_bone['Location'][1]} {-root_bone['Location'][2]}\n"
    bvh_header += "CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n"

    # 构建骨骼层次结构
    def build_bvh_hierarchy(bone_index, indent=1):
        bvh_part = ""
        for child_index in bone_children[bone_index]:
            child_bone = frame_bones[child_index]
            bvh_part += "  " * indent + f"JOINT {bone_index_to_name[child_index]}\n"
            bvh_part += "  " * indent + "{\n"
            bvh_part += "  " * (indent + 1) + f"OFFSET {-child_bone['Location'][0]} {-child_bone['Location'][1]} {-child_bone['Location'][2]}\n"
            bvh_part += "  " * (indent + 1) + "CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n"
            bvh_part += build_bvh_hierarchy(child_index, indent + 1)
            bvh_part += "  " * indent + "}\n"
        return bvh_part

    bvh_header += build_bvh_hierarchy(0)
    bvh_header += "}\n"

    # 构建动画数据部分
    num_frames = len(frames)
    bvh_motion = f"MOTION\nFrames: {num_frames}\nFrame Time: 0.00833\n"  # 假设帧率为30fps

    for frame in frames:
        frame_data = []
        for bone in frame["bvh_source"]:
            x, y, z = bone['Location']
            frame_data.extend([-x, -y, -z])

            x_e, y_e, z_e = quaternion_to_euler(bone['Rotation'],order='zyx')
            frame_data.extend([x_e, y_e, z_e])
            
        bvh_motion += " ".join(map(str, frame_data)) + "\n"

    # 写入BVH文件
    record_folder = "recorded_frames"
    if not os.path.exists(record_folder):
        os.makedirs(record_folder)

    with open(os.path.join(record_folder, output_file), 'w') as outfile:
        outfile.write(bvh_header)
        outfile.write(bvh_motion)

    print(f"JSON数据已成功转换为BVH文件并保存到 {output_file}")

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None
    except json.JSONDecodeError:
        print(f"JSON 文件格式错误: {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

#示例用法
# json_file_path = "recorded_frames_20250226200340.json"
# json_data = read_json_file(json_file_path)

# # 示例调用
# output_file = "output1.bvh"
# json_to_bvh(json_data, output_file)

# q = [-0.575,   -0.124,  -0.809, 0.0]

# # 测试不同的旋转顺序
# print("xyz:", quaternion_to_euler(q, order='xyz'))
# print("zyx:", quaternion_to_euler(q, order='zyx'))
# print("yzx:", quaternion_to_euler(q, order='yzx'))