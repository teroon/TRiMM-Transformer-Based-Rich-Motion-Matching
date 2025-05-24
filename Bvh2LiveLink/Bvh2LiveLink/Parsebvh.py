import math
import numpy as np
import time

# 四元数转换函数
def euler_to_quaternion(roll, pitch, yaw, order='xyz'):
    roll = math.radians(round(roll, 6))
    pitch = math.radians(round(pitch, 6))
    yaw = math.radians(round(yaw, 6))

    # 计算每个轴的四元数
    qx = [math.sin(roll / 2), 0, 0, math.cos(roll / 2)]
    qy = [0, math.sin(pitch / 2), 0, math.cos(pitch / 2)]
    qz = [0, 0, math.sin(yaw / 2), math.cos(yaw / 2)]

    # 按照指定顺序组合四元数
    if order == 'xyz':
        q = quaternion_multiply(quaternion_multiply(qx, qy), qz)
    elif order == 'xzy':
        q = quaternion_multiply(quaternion_multiply(qx, qz), qy)
    elif order == 'yxz':
        q = quaternion_multiply(quaternion_multiply(qy, qx), qz)
    elif order == 'yzx':
        q = quaternion_multiply(quaternion_multiply(qy, qz), qx)
    elif order == 'zxy':
        q = quaternion_multiply(quaternion_multiply(qz, qx), qy)
    elif order == 'zyx':
        q = quaternion_multiply(quaternion_multiply(qz, qy), qx)
    else:
        raise ValueError(f"Invalid rotation order: {order}")

    # 归一化四元数
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    q = [q[i] / norm for i in range(4)]

    # 标准化符号
    if q[3] < 0:
        q = [-q[i] for i in range(4)]

    return q  # 返回四元数

def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return [x, y, z, w]

def parse_bvh(bvh_file_path):
    start_time = time.perf_counter()

    with open(bvh_file_path, 'r') as f:
        bvh_data = f.readlines()
    end_time = time.perf_counter()
    print(f"读取文件耗时: {end_time - start_time:.6f}秒")

    bones = []
    bone_stack = []
    frame_data = []
    
    is_hierarchy_section = True
    start_time = time.perf_counter()
    # 解析 BVH 文件
    for line in bvh_data:
        line = line.strip()

        if is_hierarchy_section:
            if line.startswith("ROOT") or line.startswith("JOINT"):
                bone_name = line.split()[1]
                bone = {
                    'Name': bone_name,
                    'Parent': None,
                    'Location': [0.0, 0.0, 0.0],
                    'x_offset': 0.0,  # 存储 x_offset
                    'y_offset': 0.0,  # 存储 y_offset
                    'z_offset': 0.0,  # 存储 z_offset
                    'Rotation': [0.0, 0.0, 0.0, 1.0],
                    'Scale': [1.0, 1.0, 1.0],
                    'Channels': []
                }
                bones.append(bone)
                bone_stack.append(bone)
            elif line.startswith("OFFSET"):
                offset = list(map(float, line.split()[1:]))
                if bone_stack:
                    bone_stack[-1]['x_offset'] = offset[0]  # 存储 x_offset
                    bone_stack[-1]['y_offset'] = offset[1]  # 存储 y_offset
                    bone_stack[-1]['z_offset'] = offset[2]  # 存储 z_offset
                    bone_stack[-1]['Location'] = offset  # 仍然存储 Location
            elif line.startswith("CHANNELS"):
                channels = line.split()[2:]
                if bone_stack:
                    bone_stack[-1]['Channels'] = channels
            elif line.startswith("End Site"):
                bone_stack.append({'Name': 'End Site'})
            elif line.startswith("}") and bone_stack:
                last_bone = bone_stack.pop()
                if bone_stack and last_bone['Name'] != 'End Site':
                    last_bone['Parent'] = bone_stack[-1]['Name']
            elif line == "MOTION":
                is_hierarchy_section = False

        else:
            if line.startswith("Frames:"):
                frame_count = int(line.split()[1])
            elif line.startswith("Frame Time:"):
                frame_time = float(line.split()[2])
            elif line:
                frame_values = list(map(float, line.split()))
                frame_data.append(frame_values)
    end_time = time.perf_counter()
    print(f"解析bvh过程1耗时: {end_time - start_time:.6f}秒")
    
    start_time = time.perf_counter()
    all_frames = []
    for frame in frame_data:
        frame_bones = []
        data_iter = 0
        
        for bone in bones:
            num_position_channels = 3
            num_rotation_channels = 3 

            # 获取位置数据
            x_pos = frame[data_iter]
            y_pos = frame[data_iter + 1]
            z_pos = frame[data_iter + 2]

            if len(bone['Channels']) > num_position_channels: 
                # 存储欧拉角
                x_ang = frame[data_iter + num_position_channels]
                y_ang = frame[data_iter + num_position_channels + 1]
                z_ang = frame[data_iter + num_position_channels + 2]          
                euler_angles = [-x_ang, y_ang, -z_ang]
                
                # 转换为四元数
                # 指定旋转顺序
                rotation_order = 'xyz'

                # 调用 euler_to_quaternion 函数并传递旋转顺序参数
                bone['Rotation'] = euler_to_quaternion(*euler_angles, order=rotation_order)

            # 更新位置
            if bone['Parent'] is None:
                bone['Location'] = [x_pos, -y_pos, z_pos]
                # bone['Location'] = [0, 0, 80]
            else:
                bone['Location'] = [bone['x_offset'], -bone['y_offset'], bone['z_offset']]  # 使用 x_offset, y_offset, z_offset


            # 更新骨骼
            frame_bones.append({
                "Name": bone['Name'],
                "Location": bone['Location'],
                "Rotation": bone['Rotation'],
                "Scale": bone['Scale'],
                "Parent": bone.get('Parent', -1)
            })
            
            # 更新数据指针
            data_iter += len(bone['Channels'])
        
        all_frames.append(frame_bones)
    end_time = time.perf_counter()
    print(f"解析bvh过程2耗时: {end_time - start_time:.6f}秒")
    return all_frames