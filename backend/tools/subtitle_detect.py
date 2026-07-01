import os
import numpy as np
from backend.config import config, BASE_DIR


class SubtitleDetect:
    def __init__(self):
        self.text_detector = None

    def detect_subtitle(self, img):
        if self.text_detector is None:
            self._init_detector()
        results = list(self.text_detector.predict(img))
        dt_polys = []
        for res in results:
            dt_polys.extend(res.get("dt_polys", []))
        return np.array(dt_polys, dtype=np.float32) if dt_polys else np.array([]), 0

    def _init_detector(self):
        det_model_dir = os.path.join(BASE_DIR, "backend", "models", "det")
        det_model_dir = det_model_dir if os.path.exists(det_model_dir) else None

        try:
            from paddleocr import PaddleOCR
            self.text_detector = PaddleOCR(
                lang=config.language,
                ocr_version="PP-OCRv3",
                device="cpu",
                enable_mkldnn=False,
                det_model_dir=det_model_dir,
                show_log=False,
            )
        except ImportError:
            self.text_detector = None

    def is_available(self):
        return self.text_detector is not None
