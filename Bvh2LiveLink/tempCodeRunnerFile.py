import asyncio
import websockets
import json

async def main():
    try:
        async with websockets.connect("ws://10.203.3.208:8250") as websocket:
            while True:
                # 提示用户选择操作
                print("请选择操作：")
                print("1. 输入 action_index 和 wav_length")
                print("2. 开始录制")
                print("3. 结束录制并自动保存")
                print("4. 退出")
                choice = input("请输入选项编号: ")

                if choice == '1':
                    # 等待键盘输入 action_index 和 wav_length
                    action_index = input("请输入 action_index: ")
                    wav_length = input("请输入 wav_length: ")

                    # 发送数据
                    result = {"action_index": action_index, "wav_length": wav_length}
                    await websocket.send(json.dumps(result))

                    # 接收回复
                    response = await websocket.recv()
                    print(f"收到回复: {response}")
                elif choice == '2':
                    # 发送开始录制的指令
                    start_recording = {"recording": True}
                    await websocket.send(json.dumps(start_recording))

                    # 接收回复
                    response = await websocket.recv()
                    print(f"收到回复: {response}")
                elif choice == '3':
                    # 发送结束录制并自动保存的指令
                    stop_recording = {"recording": False, "save_recording": True}
                    await websocket.send(json.dumps(stop_recording))

                    # 接收回复
                    response = await websocket.recv()
                    print(f"收到回复: {response}")
                elif choice == '4':
                    break
                else:
                    print("无效的选项，请重新输入。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())