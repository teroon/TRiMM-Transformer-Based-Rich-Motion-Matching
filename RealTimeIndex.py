import Processor
import torch.multiprocessing as multiprocessing

new_processor = Processor.Processor()
new_processor.audio_path_queue.put("Data/audio/好奇1.wav")
new_processor.text_queue.put("that's interesting")
new_processor.multiprocessing_initial_by_file()
new_processor.multiprocessing_processor()

