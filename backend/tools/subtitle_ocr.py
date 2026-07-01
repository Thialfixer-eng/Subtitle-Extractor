import os
import sys
import cv2
import time
import traceback
import threading
import multiprocessing
from multiprocessing import Queue, Process
from collections import namedtuple
import unicodedata

from backend.config import config
from backend.bean.subtitle_area import SubtitleArea
from backend.tools.ocr import OcrRecogniser, get_coordinates


def frame_preprocess(subtitle_area, frame):
    if subtitle_area is None or subtitle_area.is_empty():
        return frame
    h, w = frame.shape[:2]
    x1 = max(0, subtitle_area.xmin)
    y1 = max(0, subtitle_area.ymin)
    x2 = min(w, subtitle_area.xmax)
    y2 = min(h, subtitle_area.ymax)
    if x2 > x1 and y2 > y1:
        return frame[y1:y2, x1:x2]
    return frame


def _inside_area(coordinate, area):
    xmin, xmax, ymin, ymax = coordinate
    return (area.xmin <= xmin and xmax <= area.xmax
            and area.ymin <= ymin and ymax <= area.ymax)


def _overlaps_area(coordinate, area):
    xmin, xmax, ymin, ymax = coordinate
    return not (xmax < area.xmin or xmin > area.xmax
                or ymax < area.ymin or ymin > area.ymax)


def ocr_worker(task_queue, raw_dir, sub_area, watermark_area, options, progress_queue, worker_id):
    text_recogniser = OcrRecogniser()
    text_recogniser.init_model()

    total_tasks = 0
    processed = 0
    drop_score = options.get("DROP_SCORE", 0.75)
    raw_path = os.path.join(raw_dir, f"raw_{worker_id}.txt")

    with open(raw_path, "w", encoding="utf-8") as raw_file:
        while True:
            try:
                data = task_queue.get()
                if data is None:
                    break

                total_frame_count, current_frame_no, dt_box, rec_res, current_frame_ms, subtitle_area, frame = data

                if current_frame_no == -1:
                    break

                if total_frame_count > 0 and total_tasks == 0:
                    total_tasks = total_frame_count
                    progress_queue.put((-2, total_tasks, worker_id))

                if dt_box is None or rec_res is None:
                    try:
                        crop = frame_preprocess(subtitle_area, frame)
                        dt_box, rec_res = text_recogniser.predict(crop)
                    except Exception:
                        print(f"[OCR ERROR] predict() failed for frame {current_frame_no}:", file=sys.stderr)
                        traceback.print_exc()
                        processed += 1
                        progress_queue.put((current_frame_no, processed, worker_id))
                        continue

                if not dt_box or not rec_res:
                    processed += 1
                    progress_queue.put((current_frame_no, processed, worker_id))
                    continue

                dt_box_list = dt_box if isinstance(dt_box, list) else (dt_box.tolist() if hasattr(dt_box, 'tolist') and len(dt_box) > 0 else [])
                if not dt_box_list:
                    processed += 1
                    progress_queue.put((current_frame_no, processed, worker_id))
                    continue

                coordinates = get_coordinates(dt_box_list)

                if subtitle_area is not None and not subtitle_area.is_empty():
                    dx, dy = subtitle_area.xmin, subtitle_area.ymin
                    coordinates = [
                        (xmin + dx, xmax + dx, ymin + dy, ymax + dy)
                        for (xmin, xmax, ymin, ymax) in coordinates
                    ]

                for coordinate, (text, score) in zip(coordinates, rec_res):
                    if score < drop_score:
                        continue

                    if watermark_area is not None and not watermark_area.is_empty():
                        if _inside_area(coordinate, watermark_area) or _overlaps_area(coordinate, watermark_area):
                            continue

                    if sub_area is not None and not sub_area.is_empty():
                        if not _inside_area(coordinate, sub_area):
                            continue

                    xmin, xmax, ymin, ymax = coordinate
                    line = f"{current_frame_no}\t{coordinate}\t{text}\n"
                    raw_file.write(line)

                processed += 1
                progress_queue.put((current_frame_no, processed, worker_id))
            except Exception:
                print(f"[OCR ERROR] Unhandled in worker loop:", file=sys.stderr)
                traceback.print_exc()
                continue

    progress_queue.put((-1, processed, worker_id))
    text_recogniser.recogniser = None


def async_start(video_path, raw_dir, sub_area, watermark_area, options, num_workers=1):
    task_queue = Queue(maxsize=500)
    progress_queue = Queue()

    processes = []
    for i in range(num_workers):
        p = Process(
            target=ocr_worker,
            args=(task_queue, raw_dir, sub_area, watermark_area, options, progress_queue, i),
            daemon=True,
        )
        p.start()
        processes.append(p)

    return processes, task_queue, progress_queue
