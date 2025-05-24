import asyncio
import json
import os
import queue
import socket
import threading
import time
import numpy as np
from scipy.spatial.transform import Slerp, Rotation as R
from Bvh2LiveLink.ConvertbvhToJson import generate_json
from Bvh2LiveLink.ConvertJsonToBvh import json_to_bvh
from Bvh2LiveLink.Parsebvh import parse_bvh
import websockets


class BvhProcessor:
    bvh_folder = 'recorded_frames'
    bvh_folder = 'recorded_frames'
    def __init__(self, ip_address="127.0.0.1", port=11111, frame_rate=120, transition_frame=40, bvh_folder='Data/all_bvh'):
        self.ip_address = ip_address
        self.port = port
        self.frame_rate = frame_rate
        self.frame_interval = 1 / frame_rate
        self.transition_frame = transition_frame

        self.bvh_folder = bvh_folder
        self.bvh_files = self.get_bvh_files()

        self.stop_event = threading.Event()
        self.switch_event = threading.Event()

        self.bvh_queue = queue.Queue()  # 用于存储用户输入的动作索引
        self.load_queue = queue.Queue()  # 用于存储需要加载的BVH文件路径
        self.next_bvh_json = None  # 存储新加载的BVH文件数据

        self.recording = False
        self.recorded_frames = []

        if not self.bvh_files:
            raise ValueError("Error: No BVH files found in the specified folder.")

    def get_bvh_files(self):
        return [os.path.join(self.bvh_folder, f) for f in os.listdir(self.bvh_folder) if f.endswith('.bvh')]

    def load_bvh_file(self, bvh_file_path):
        try:
            source_name = os.path.splitext(os.path.basename(bvh_file_path))[0]
            all_frames = parse_bvh(bvh_file_path)
            return generate_json(all_frames, source_name)
        except Exception as e:
            print(f"Error loading BVH file {bvh_file_path}: {e}")
            return None

    def lerp(self, loc1, loc2, t):
        loc1, loc2 = np.array(loc1), np.array(loc2)
        return (1 - t) * loc1 + t * loc2
    
    def cubic_lerp(self, loc1, loc2, t):
        loc1, loc2 = np.array(loc1), np.array(loc2)
        t_cubed = t ** 3
        t_squared = t ** 2
        factor1 = 2 * t_cubed - 3 * t_squared + 1
        factor2 = -2 * t_cubed + 3 * t_squared
        return factor1 * loc1 + factor2 * loc2

    def interpolate_frames(self, frame1, frame2, steps):
        interpolated_frames = []
        # 提前创建 Slerp 对象
        slerp_objects = []
        for bvh_source1, bvh_source2 in zip(frame1['bvh_source'], frame2['bvh_source']):
            slerp = Slerp([0, 1], R.from_quat([bvh_source1["Rotation"], bvh_source2["Rotation"]]))
            slerp_objects.append(slerp)

        for step in range(steps):
            t = step / float(steps)
            interpolated_frame = []
            for idx, (bvh_source1, bvh_source2) in enumerate(zip(frame1['bvh_source'], frame2['bvh_source'])):
                # 使用提前创建的 Slerp 对象
                slerp = slerp_objects[idx]
                interpolated_rot = slerp(t ** 3).as_quat().tolist()
                interpolated_loc = self.cubic_lerp(bvh_source1["Location"], bvh_source2["Location"], t).tolist()
                interpolated_frame.append({
                    "Name": bvh_source1["Name"],
                    "Parent": bvh_source1["Parent"],
                    "Location": interpolated_loc,
                    "Rotation": interpolated_rot,
                    "Scale": bvh_source1["Scale"]
                })
            interpolated_frames.append({"bvh_source": interpolated_frame})
        return interpolated_frames

    def send_data(self, current_frame_index, bvh_json, last_frame):
        cli = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while not self.stop_event.is_set():
            for i in range(current_frame_index[0], len(bvh_json)):
                if self.stop_event.is_set():
                    break
                if self.switch_event.is_set() and self.next_bvh_json is not None:
                    self.switch_event.clear()
                    start_time = time.perf_counter()
                    transition_frames = self.interpolate_frames(last_frame, self.next_bvh_json[0],
                                                                self.transition_frame)
                    end_time = time.perf_counter()
                    start_time = time.perf_counter()
                    for frame in transition_frames:
                        try:
                            cli.sendto(json.dumps(frame).encode('utf-8'), (self.ip_address, self.port))
                            if self.recording:
                                self.recorded_frames.append(frame)
                        except Exception as e:
                            print(f"Error sending data: {e}")
                        time.sleep(self.frame_interval)
                    
                    current_frame_index[0] = 0
                    bvh_json[:] = self.next_bvh_json  # 切换到新动作
                    self.next_bvh_json = None  # 清空新BVH数据
                    last_frame = bvh_json[0]["bvh_source"]
                    break
                try:
                    cli.sendto(json.dumps(bvh_json[i]).encode('utf-8'), (self.ip_address, self.port))
                    last_frame = bvh_json[i]
                    if self.recording:
                        self.recorded_frames.append(bvh_json[i])
                except Exception as e:
                    print(f"Error sending data: {e}")
                time.sleep(self.frame_interval)
        cli.close()

    def check_user_input(self, current_frame_index, bvh_json):
        last_file_index = 0
        while not self.stop_event.is_set():
            if not self.bvh_queue.empty():
                new_file_index = self.bvh_queue.get()
                if new_file_index != last_file_index and 0 <= new_file_index < len(self.bvh_files):
                    bvh_file_path = self.bvh_files[new_file_index]
                    self.load_queue.put(bvh_file_path)  # 将加载任务放入队列
                    print(f"准备加载 BVH 文件: {self.bvh_files[new_file_index]}")
                    last_file_index = new_file_index
                else:
                    print("Error: 输入的编号无效。继续当前动作。")
            time.sleep(0.1)  # 减少睡眠间隔以提高响应速度

    def load_thread_func(self):
        while not self.stop_event.is_set():
            if not self.load_queue.empty():
                bvh_file_path = self.load_queue.get()
                
                # 启动一个新线程来加载 BVH 文件
                load_thread = threading.Thread(target=self._load_bvh_file_async, args=(bvh_file_path,))
                load_thread.start()
            time.sleep(0.1)  # 减少睡眠间隔以提高响应速度

    def _load_bvh_file_async(self, bvh_file_path):
        try:
            start_time = time.perf_counter()
            self.next_bvh_json = self.load_bvh_file(bvh_file_path)  # 加载新BVH文件
            self.switch_event.set()  # 触发切换事件
            end_time = time.perf_counter()
            print(f"读取新BVH耗时: {end_time - start_time:.6f}秒")
        except Exception as e:
            print(f"Error loading BVH file in async thread: {e}")

    async def handle_message(self, message, websocket):
        try:
            data = json.loads(message)
            if "recording" in data:
                if data["recording"]:
                    self.recording = True
                    await websocket.send("录制已开始")
                else:
                    self.recording = False
                    self.save_recorded_frames()
                    await websocket.send("录制已结束，数据已保存")
            elif "action_index" in data:
                file_index = int(data["action_index"])
                filename=data["filename"]
                self.bvh_queue.put(file_index)
                self.filename=filename
                await websocket.send(f"已接收 BVH 文件编号: {file_index}")
            else:
                await websocket.send("无效的输入，请输入有效的指令。")
        except Exception as e:
            await websocket.send(f"解析错误: {e}")

    async def echo(self, websocket, path):
        async for message in websocket:
            print(f"接收到消息: {message}")
            start_time = time.perf_counter()
            await self.handle_message(message, websocket)
            end_time = time.perf_counter()
            print(f"接收消息耗时: {end_time - start_time:.6f}秒")

    async def run(self):
        # 动作索引初始值
        file_index = 0

        bvh_file_path = self.bvh_files[file_index]
        bvh_json = self.load_bvh_file(bvh_file_path)

        # 当前帧索引
        current_frame_index = [0]

        # 存储上一帧用于插值
        last_frame = bvh_json[0]["bvh_source"]

        # 启动发送、输入监听和加载线程
        send_thread = threading.Thread(target=self.send_data, args=(current_frame_index, bvh_json, last_frame))
        input_thread = threading.Thread(target=self.check_user_input, args=(current_frame_index, bvh_json))
        load_thread = threading.Thread(target=self.load_thread_func)
        send_thread.start()
        input_thread.start()
        load_thread.start()

        async with websockets.serve(self.echo, "127.0.0.1", 8086, ping_interval=30, ping_timeout=1000):
            print("WebSocket服务器已启动，正在监听127.0.0.1")
            await asyncio.Future()  # 保持服务器运行

    def save_recorded_frames(self):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = self.filename+".bvh"
        json_to_bvh(self.recorded_frames, output_file)
        self.recorded_frames = []


if __name__ == "__main__":
    processor = BvhProcessor()
    asyncio.run(processor.run())