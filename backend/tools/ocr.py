import os
import sys
import traceback
import tempfile
import numpy as np
from PIL import Image
import cv2

from backend.config import config, BASE_DIR


def get_coordinates(dt_polys):
    coordinates = []
    for poly in dt_polys:
        x_coords = [point[0] for point in poly]
        y_coords = [point[1] for point in poly]
        xmin = min(x_coords)
        xmax = max(x_coords)
        ymin = min(y_coords)
        ymax = max(y_coords)
        coordinates.append((int(xmin), int(xmax), int(ymin), int(ymax)))
    return coordinates


class OcrRecogniser:
    def __init__(self):
        self.recogniser = None

    def init_model(self):
        from paddleocr import PaddleOCR

        use_gpu = config.use_gpu
        device = "gpu" if use_gpu else "cpu"

        if device == "gpu":
            try:
                import paddle
                if not paddle.is_compiled_with_cuda():
                    print("[OCR WARN] GPU requested but paddle has no CUDA support — falling back to CPU", file=sys.stderr)
                    device = "cpu"
                else:
                    try:
                        if not paddle.static.cuda_places():
                            print("[OCR WARN] GPU requested but no CUDA devices available — falling back to CPU", file=sys.stderr)
                            device = "cpu"
                    except Exception:
                        pass
            except Exception:
                device = "cpu"

        self.recogniser = PaddleOCR(
            lang=config.language,
            ocr_version="PP-OCRv3",
            device=device,
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_thresh=0.3,
            text_det_box_thresh=0.5,
            text_recognition_batch_size=config.rec_batch_number,
            text_rec_score_thresh=config.drop_score / 100.0,
        )

    def predict(self, image):
        if self.recogniser is None:
            self.init_model()

        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                return [], []
        elif isinstance(image, np.ndarray):
            pass
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            return [], []

        try:
            results = list(self.recogniser.predict_iter(image))
        except Exception as e:
            print(f"[OCR ERROR] predict() failed: {e}", file=sys.stderr)
            traceback.print_exc()
            return [], []

        dt_box = []
        rec_res = []

        for res in results:
            dt_polys = res.get("dt_polys", [])
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])
            rec_polys = res.get("rec_polys", [])

            if rec_polys and len(rec_polys) == len(rec_texts) == len(rec_scores):
                for poly, text, score in zip(rec_polys, rec_texts, rec_scores):
                    dt_box.append(np.array(poly, dtype=np.float32))
                    rec_res.append((text, score))
            elif len(dt_polys) == len(rec_texts) == len(rec_scores):
                for poly, text, score in zip(dt_polys, rec_texts, rec_scores):
                    dt_box.append(np.array(poly, dtype=np.float32))
                    rec_res.append((text, score))
            elif len(rec_texts) > 0 and len(rec_texts) == len(rec_scores):
                for i in range(len(rec_texts)):
                    poly = rec_polys[i] if i < len(rec_polys) else dt_polys[i] if i < len(dt_polys) else None
                    if poly is not None:
                        dt_box.append(np.array(poly, dtype=np.float32))
                        rec_res.append((rec_texts[i], rec_scores[i]))

        if not rec_res:
            return [], []

        coordinates = get_coordinates([p.tolist() for p in dt_box])
        sorted_idx = sorted(
            range(len(coordinates)),
            key=lambda i: (coordinates[i][2], coordinates[i][0])
        )

        dt_box = [dt_box[i] for i in sorted_idx]
        rec_res = [rec_res[i] for i in sorted_idx]

        return dt_box, rec_res
