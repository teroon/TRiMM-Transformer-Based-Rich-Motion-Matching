import json

def generate_json(all_frames, source_name):
    # 创建映射
    bone_name_to_index = {}
    for frame in all_frames:
        for bone in frame:
            if bone['Name'] not in bone_name_to_index:
                bone_name_to_index[bone['Name']] = len(bone_name_to_index)

    bone_data = []
    for frame in all_frames:
        frame_data = []
        for bone in frame:
            parent_index = -1
            if bone['Parent'] != -1:
                parent_index = bone_name_to_index.get(bone['Parent'], -1)

            frame_data.append({
                "Name": bone['Name'],
                "Parent": parent_index,
                "Location": bone['Location'],
                "Rotation": bone['Rotation'],
                "Scale": bone['Scale']
            })
        bone_data.append(frame_data)

    final_json = []
    for frame_data in bone_data:
        final_json.append({
            "bvh_source": frame_data
        })

    return final_json

# 示例调用
# all_frames = [
#     [
#         {"Name": "Hips", "Parent": -1, "Location": [0.0, 0.0, 0.0], "Rotation": [-0.2300403187269106, 0.598488159746974, -0.7146375832762933, -0.2796184882509042], "Scale": [1.0, 1.0, 1.0]},
#         {"Name": "Spine", "Parent": 0, "Location": [0.0, 8.7478, -2.13942], "Rotation": [-0.2300403187269106, 0.598488159746974, -0.7146375832762933, -0.2796184882509042], "Scale": [1.0, 1.0, 1.0]}
#     ]
# ]
# source_name = "example_source"
# output_file = "output_1.json"

# bvh_json = generate_json(all_frames, source_name)
# save_json(bvh_json, output_file)