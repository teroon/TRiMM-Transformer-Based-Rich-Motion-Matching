import numpy as np

def rotation_matrix_to_axis_angle(R):
    """将旋转矩阵转换为轴-角表示法"""
    # 计算旋转角度
    trace = np.trace(R)
    theta = np.arccos((trace - 1) / 2)
    
    # 计算旋转轴
    if np.sin(theta) != 0:
        vx = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
        vy = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
        vz = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))
    else:
        # 如果旋转角度是 0 或 180 度
        vx, vy, vz = 1, 0, 0  # 默认返回绕 X 轴旋转
    
    # 返回旋转角度和旋转轴（单位向量）
    axis = np.array([vx, vy, vz])
    axis = axis / np.linalg.norm(axis)  # 归一化旋转轴
    return theta, axis

def calculate_rotation_matrix_from_positions(pos1, pos2, pos3):
    """假设这三个位置点用于构建旋转矩阵，实际应用中可能需要其他处理"""
    # 计算两个方向向量
    vec1 = pos2 - pos1
    vec2 = pos3 - pos2
    
    # 计算法向量（假设两个向量的叉积表示旋转轴）
    axis = np.cross(vec1, vec2)
    axis = axis / np.linalg.norm(axis)  # 归一化旋转轴
    
    # 计算旋转角度（假设为向量夹角）
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 防止数值误差
    
    # 构造旋转矩阵（这里只是简化的示例，通常需要通过四元数或其他方法来构造）
    # 使用旋转轴和角度生成旋转矩阵（可以使用Rodrigues'旋转公式）
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    I = np.eye(3)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    return R

# 输入 BVH 数据和 Blender 数据
bvh_positions = np.array([
    [-3.3085, -11.8409, -8.0939],
    [-3.1716, -11.7373, -8.0626],
    [-3.1133, -11.6363, -8.0597],
    [-3.0031, -11.5098, -8.0216]
])

blender_positions = np.array([
    [-1.56504, -10.0374, -10.2524],
    [-1.45201, -9.91951, -10.2229],
    [-1.41214, -9.81208, -10.2084],
    [-1.33078, -9.67841, -10.162]
])

# 假设我们从 BVH 和 Blender 数据中提取了 3 个关键点来计算旋转矩阵
R_BVH = calculate_rotation_matrix_from_positions(bvh_positions[0], bvh_positions[1], bvh_positions[2])
R_Blender = calculate_rotation_matrix_from_positions(blender_positions[0], blender_positions[1], blender_positions[2])

# 将旋转矩阵转换为轴-角表示
theta_BVH, axis_BVH = rotation_matrix_to_axis_angle(R_BVH)
theta_Blender, axis_Blender = rotation_matrix_to_axis_angle(R_Blender)

# 输出结果
print("BVH 旋转角度:", np.degrees(theta_BVH), "°")
print("BVH 旋转轴:", axis_BVH)
print("Blender 旋转角度:", np.degrees(theta_Blender), "°")
print("Blender 旋转轴:", axis_Blender)

# 比较旋转角度和旋转轴
angle_diff = np.abs(np.degrees(theta_BVH) - np.degrees(theta_Blender))
axis_diff = np.linalg.norm(axis_BVH - axis_Blender)
print(f"旋转角度差异: {angle_diff}°")
print(f"旋转轴差异: {axis_diff}")
