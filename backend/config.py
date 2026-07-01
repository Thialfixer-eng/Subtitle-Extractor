import os
import json

VERSION = "1.1.0"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    def __init__(self):
        self.language = "ch"
        self.mode = "fast"
        self.extract_frequency = 3
        self.threshold_text_similarity = 80
        self.drop_score = 75
        self.hardware_acceleration = False
        self.gpu_id = 0
        self.word_segmentation = True
        self.generate_txt = True
        self.debug_no_delete_cache = False
        self.debug_ocr_loss = False
        self.rec_batch_number = 6
        self.max_batch_size = 10
        self.tolerant_pixel_x = 10
        self.tolerant_pixel_y = 10
        self.subtitle_area_deviation_pixel = 20
        self.subtitle_area_deviation_rate = 20
        self.watermark_area_num = 5
        self.delete_empty_timestamp = True
        self.num_ocr_workers = 1

    @property
    def use_gpu(self):
        return self.hardware_acceleration

    @use_gpu.setter
    def use_gpu(self, value):
        self.hardware_acceleration = value


config = Config()
