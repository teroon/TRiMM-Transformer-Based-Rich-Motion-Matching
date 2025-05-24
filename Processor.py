import torch.multiprocessing as multiprocessing
import time

import Inference
import Bvh2LiveLink

class Processor:
    def __init__(
            self
    ):
        self.bvh_processor = None
        self.inference_processor = None
        self.inference = Inference.Inference()
        self.audio_queue = multiprocessing.Queue()
        self.audio_path_queue = multiprocessing.Queue()
        self.text_queue = multiprocessing.Queue()
        self.action_queue = multiprocessing.Queue()
        self.is_audio_queue_empty = False
        self.is_audio_path_queue_empty = False
        self.is_text_queue_empty = False
        print("error0\n")

    def inference_processor_by_file(
            self,
    ):
        while True:

            audio_path = self.audio_path_queue.get()
            text = self.text_queue.get()
            action_index, wav_length = self.inference.inference_by_file(
                audio_path,
                text
            )
 
            print("error1\n")
            if self.audio_path_queue.empty():
                self.is_audio_queue_empty = True
                count = 0
                while count < 2:
                    count += 1
                    #time.sleep(wav_length)
                    if self.audio_path_queue.empty():
                        self.is_audio_path_queue_empty = True
                        print("Action Processor Inference Thread Waiting...This is " + str(count) + "Waiting...")
                    else:
                        self.is_audio_path_queue_empty = False
                        break

                if self.is_audio_path_queue_empty:
                    self.is_audio_path_queue_empty = False
                    break

                #queue.put(action_index)

    def inference_processor_by_data(
            self,
            queue
    ):
        print("0\n")
        while True:
            audio = self.audio_queue.get()
            text = self.text_queue.get()

            action_index, wav_length = self.inference.inference_by_data(
                audio,
                text
            )
            if self.audio_queue.empty():
                self.is_audio_queue_empty = True

                count = 0

                while count < 2:
                    count += 1
                    time.sleep(wav_length)
                    if self.audio_queue.empty():
                        self.is_audio_queue_empty = True
                        print("Action Processor Inference Thread Waiting...This is " + str(count) + "Waiting...")
                    else:
                        self.is_audio_queue_empty = False
                        break

            if self.is_audio_queue_empty:
                self.is_audio_path_queue_empty = False
                break

            queue.put(action_index)

    def bvh_processer(
            self,
            queue
    ):
        Bvh2LiveLink.main(queue)

    def multiprocessing_initial_by_file(
            self
    ):
        self.inference_processor = multiprocessing.Process(target=self.inference_processor_by_file, args=())
        self.bvh_processor = multiprocessing.Process(target=self.bvh_processor, args=(self.action_queue,))

    def multiprocessing_initial_by_data(
            self
    ):
        self.inference_processor = multiprocessing.Process(target=self.inference_processor_by_data, args=(self.action_queue,))
        self.bvh_processor = multiprocessing.Process(target=self.bvh_processor, args=(self.action_queue,))

    def multiprocessing_processor(
            self
    ):

        self.inference_processor.start()
        self.bvh_processor.start()

        self.inference_processor.join()
        self.bvh_processor.join()