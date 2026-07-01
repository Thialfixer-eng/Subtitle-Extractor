import os
import shutil
import time
import threading
import multiprocessing as mp
from collections import namedtuple, OrderedDict
from pathlib import Path
import unicodedata

import cv2
import numpy as np
from Levenshtein import ratio
from tqdm import tqdm

from backend.config import config, BASE_DIR, VERSION
from backend.bean.subtitle_area import SubtitleArea
from backend.tools.ocr import OcrRecogniser, get_coordinates
from backend.tools.subtitle_ocr import async_start, frame_preprocess
from backend.tools import reformat


class SubtitleExtractor:
    def __init__(self, video_path, sub_area=None, watermark_area=None):
        self.video_path = video_path
        self.sub_area = sub_area
        self.watermark_area = watermark_area

        self.video_cap = cv2.VideoCapture(video_path)
        self.vd_name = Path(self.video_path).stem
        self.output_dir = os.path.join(BASE_DIR, "output")
        self.temp_output_dir = os.path.join(self.output_dir, self.vd_name)
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.frame_output_dir = os.path.join(self.temp_output_dir, "frames")
        self.subtitle_output_dir = os.path.join(self.temp_output_dir, "subtitle")
        self.raw_subtitle_path = os.path.join(self.subtitle_output_dir, "raw.txt")
        self.subtitle_output_path = os.path.join(self.output_dir, f"{self.vd_name}.srt")

        self.ocr = None
        self.is_finished = False
        self.is_cancelled = False
        self.is_paused = False
        self.progress_total = 300
        self.progress_frame_extract = 0
        self.progress_ocr = 0
        self.progress_post = 0
        self.subtitle_ocr_task_queue = None
        self.subtitle_ocr_progress_queue = None
        self.ocr_processes = []
        self.pause_event = None
        self.progress_callback = None
        self.lock = threading.RLock()

    def set_progress_callback(self, cb):
        self.progress_callback = cb

    def cancel(self):
        self.is_cancelled = True
        for p in self.ocr_processes:
            if p and p.is_alive():
                p.terminate()
                p.join(timeout=3)

    def pause(self):
        self.is_paused = True
        if self.pause_event:
            self.pause_event.clear()

    def resume(self):
        self.is_paused = False
        if self.pause_event:
            self.pause_event.set()

    def run(self):
        start_time = time.time()
        self._log(f"Video-subtitle-extractor v{VERSION}")
        self._log(f"Video: {self.video_path}")
        self._log(f"Frames: {self.frame_count} | FPS: {self.fps:.2f} | Resolution: {self.frame_width}x{self.frame_height}")
        self._log(f"Mode: {config.mode} | Language: {config.language} | GPU: {config.use_gpu}")
        self._log("")

        if self.is_cancelled:
            self._log("Cancelled.")
            self.is_finished = True
            return

        self._clean_cache()
        os.makedirs(self.frame_output_dir, exist_ok=True)
        os.makedirs(self.subtitle_output_dir, exist_ok=True)

        self._log("Extracting frames with OCR...")
        self.ocr = OcrRecogniser()
        self.ocr.init_model()

        self.pause_event = mp.Event()
        self.pause_event.set()

        num_workers = config.num_ocr_workers
        processes, task_queue, progress_queue = async_start(
            self.video_path,
            self.subtitle_output_dir,
            self.sub_area,
            self.watermark_area,
            self._get_ocr_options(),
            num_workers=num_workers,
        )
        self.subtitle_ocr_task_queue = task_queue
        self.subtitle_ocr_progress_queue = progress_queue
        self.ocr_processes = processes

        progress_thread = threading.Thread(target=self._monitor_ocr_progress, daemon=True)
        progress_thread.start()

        if config.mode == "fast":
            self._extract_frames_by_fps()
        else:
            self._extract_frame_by_det()

        for _ in processes:
            self.subtitle_ocr_task_queue.put(None)
        for p in processes:
            p.join()

        if self.is_cancelled:
            self._log("Cancelled.")
            self.is_finished = True
            self._clean_cache()
            return

        self._merge_raw_files()

        self._log("Deduplicating and generating SRT...")
        self.update_progress(post=20)
        self._generate_srt()
        self.update_progress(post=90)

        if self.is_cancelled:
            self._log("Cancelled.")
            self.is_finished = True
            return

        if config.word_segmentation:
            reformat.execute(self.subtitle_output_path, config.language)

        elapsed = time.time() - start_time
        self._log(f"Done! Output: {self.subtitle_output_path}")
        self._log(f"Time: {elapsed:.1f}s")

        self.update_progress(ocr=100, frame_extract=100, post=100)
        self.is_finished = True

        self._clean_cache()

        if config.generate_txt:
            self._srt_to_txt(self.subtitle_output_path)

    def _merge_raw_files(self):
        num_workers = config.num_ocr_workers
        lines = []
        for i in range(num_workers):
            path = os.path.join(self.subtitle_output_dir, f"raw_{i}.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        lines.append(line)
                os.remove(path)
        lines.sort(key=lambda l: int(l.split("\t")[0]) if "\t" in l else 0)
        with open(self.raw_subtitle_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def _extract_frames_by_fps(self):
        frame_interval = max(1, int(self.fps // config.extract_frequency))
        current_frame_no = 0
        total_frames = self.frame_count

        if self.sub_area is not None and not self.sub_area.is_empty():
            auto_area = self.sub_area
        else:
            auto_area = None

        with tqdm(total=total_frames, desc="Frames", unit="f", position=0, leave=False) as pbar:
            while current_frame_no < total_frames:
                if self.is_cancelled:
                    break
                while self.is_paused and not self.is_cancelled:
                    time.sleep(0.1)

                ret, frame = self.video_cap.read()
                if not ret:
                    break
                current_frame_no += 1

                if current_frame_no % frame_interval != 1:
                    self.progress_frame_extract = (current_frame_no / total_frames) * 100
                    pbar.update(1)
                    self._report_progress()
                    continue

                task = (self.frame_count, current_frame_no, None, None, None,
                        self.sub_area, frame)
                self.subtitle_ocr_task_queue.put(task)
                self.progress_frame_extract = (current_frame_no / total_frames) * 100
                pbar.update(1)
                self._report_progress()

        self.video_cap.release()

    def _extract_frame_by_det(self):
        current_frame_no = 0
        ocr_result_cache = {}
        start_frame_no = 0
        is_finding_start = True
        is_finding_end = False
        ocr_args_list = []

        with tqdm(total=self.frame_count, desc="Frames", unit="f", position=0, leave=False) as pbar:
            while self.video_cap.isOpened():
                if self.is_cancelled:
                    break
                while self.is_paused and not self.is_cancelled:
                    time.sleep(0.1)

                ret, frame = self.video_cap.read()
                if not ret:
                    break
                current_frame_no += 1
                pbar.update(1)

                if self.ocr is None:
                    self.ocr = OcrRecogniser()
                    self.ocr.init_model()

                crop = frame_preprocess(self.sub_area, frame)
                dt_boxes, rec_res = self.ocr.predict(crop)
                dt_box_list = dt_boxes if isinstance(dt_boxes, list) else (dt_boxes.tolist() if hasattr(dt_boxes, 'tolist') and len(dt_boxes) > 0 else [])
                coordinates = get_coordinates(dt_box_list)

                has_subtitle = len(coordinates) > 0

                if has_subtitle:
                    if is_finding_start:
                        start_frame_no = current_frame_no
                        text = " ".join([r[0] for r in rec_res])
                        ocr_result_cache[current_frame_no] = text
                        is_finding_start = False
                        is_finding_end = True
                        ocr_args_list.append((self.frame_count, current_frame_no, dt_boxes, rec_res,
                                              None, self.sub_area, frame))

                    elif is_finding_end:
                        prev_text = ocr_result_cache.get(start_frame_no, "")
                        current_text = " ".join([r[0] for r in rec_res])
                        if current_text and prev_text:
                            sim = ratio(prev_text.replace(" ", ""), current_text.replace(" ", ""))
                            if sim < config.threshold_text_similarity / 100.0:
                                end_frame_no = current_frame_no - 1
                                ocr_args_list.append((self.frame_count, end_frame_no, dt_boxes, rec_res,
                                                      None, self.sub_area, frame))
                                is_finding_end = False
                                is_finding_start = True
                                start_frame_no = current_frame_no
                                ocr_result_cache[current_frame_no] = current_text
                                ocr_args_list.append((self.frame_count, current_frame_no, dt_boxes, rec_res,
                                                      None, self.sub_area, frame))
                else:
                    if is_finding_end:
                        end_frame_no = current_frame_no - 1
                        is_finding_end = False
                        is_finding_start = True

                while len(ocr_args_list) > 0:
                    task = ocr_args_list.pop(0)
                    self.subtitle_ocr_task_queue.put(task)
                    self.progress_frame_extract = (current_frame_no / self.frame_count) * 100
                    self._report_progress()

        self.video_cap.release()

    def _generate_srt(self):
        self._concat_same_frame()
        with open(self.raw_subtitle_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        RawInfo = namedtuple("RawInfo", "no content")
        content_list = []
        for line in lines:
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            content_list.append(RawInfo(parts[0], parts[2].strip()))

        unique_subtitles = []
        i = 0
        n = len(content_list)

        while i < n:
            if self.is_cancelled:
                return
            start_frame = content_list[i].no
            j = i
            while j < n:
                if self.is_cancelled:
                    return
                self.update_progress(post=20 + int(65 * j / max(n, 1)))
                if j + 1 == n or ratio(
                    content_list[j].content.replace(" ", ""),
                    content_list[j + 1].content.replace(" ", "")
                ) < (config.threshold_text_similarity / 100.0):
                    end_frame = content_list[j].no
                    similar = content_list[i:j + 1]
                    best = max(similar, key=lambda x: len(x.content.replace(" ", "")))
                    unique_subtitles.append((start_frame, end_frame, best.content))
                    i = j + 1
                    break
                else:
                    j += 1

        if self.is_cancelled:
            return

        with open(self.subtitle_output_path, "w", encoding="utf-8") as f:
            for idx, (start, end, text) in enumerate(unique_subtitles, 1):
                start_time = self._frame_to_timecode(int(start))
                if abs(int(end) - int(start)) < self.fps:
                    end_time = self._frame_to_timecode(int(start) + int(self.fps))
                else:
                    end_time = self._frame_to_timecode(int(end))
                f.write(f"{idx}\n{start_time} --> {end_time}\n{text}\n\n")

    def _concat_same_frame(self):
        if not os.path.exists(self.raw_subtitle_path):
            return
        with open(self.raw_subtitle_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        grouped = OrderedDict()
        for line in lines:
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            frame_no, coordinate, content = parts
            if frame_no not in grouped:
                grouped[frame_no] = (coordinate, [])
            grouped[frame_no][1].append(content.strip())

        with open(self.raw_subtitle_path, "w", encoding="utf-8") as f:
            for frame_no, (coordinate, contents) in grouped.items():
                merged = " ".join(contents)
                merged = unicodedata.normalize("NFKC", merged)
                f.write(f"{frame_no}\t{coordinate}\t{merged}\n")

    def _monitor_ocr_progress(self):
        total_tasks = None
        worker_progress = {}
        while True:
            try:
                value = self.subtitle_ocr_progress_queue.get(timeout=30)
                if isinstance(value, tuple) and len(value) == 3 and value[0] == -2:
                    total_tasks = value[1]
                    continue
                if value == -1 or (isinstance(value, tuple) and value[0] == -1):
                    if isinstance(value, tuple) and len(value) == 3:
                        worker_progress[value[2]] = value[1]
                    done = len(worker_progress) >= config.num_ocr_workers
                    if done or value == -1:
                        self.update_progress(ocr=100)
                        return
                    continue
                if isinstance(value, tuple) and len(value) == 3:
                    _, processed, worker_id = value
                    worker_progress[worker_id] = processed
                    total_processed = sum(worker_progress.values())
                    if total_tasks and total_tasks > 0:
                        pct = min(99, total_processed / total_tasks * 100)
                    else:
                        pct = min(99, total_processed)
                    self.update_progress(ocr=pct)
            except Exception:
                self.update_progress(ocr=100)
                return

    def _frame_to_timecode(self, frame_no):
        total_ms = max(0, frame_no - 1) / self.fps * 1000
        ms = int(total_ms % 1000)
        total_sec = int(total_ms // 1000)
        s = total_sec % 60
        m = (total_sec // 60) % 60
        h = total_sec // 3600
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _get_ocr_options(self):
        return {
            "REC_CHAR_TYPE": config.language,
            "DROP_SCORE": config.drop_score / 100.0,
            "SUB_AREA_DEVIATION_RATE": config.subtitle_area_deviation_rate / 100.0,
            "DEBUG_OCR_LOSS": config.debug_ocr_loss,
        }

    def _clean_cache(self):
        if os.path.exists(self.temp_output_dir):
            if config.debug_no_delete_cache:
                self._log(f"Cache preserved: {self.temp_output_dir}")
            else:
                shutil.rmtree(self.temp_output_dir, True)

    def _srt_to_txt(self, srt_file):
        try:
            import pysrt
            subs = pysrt.open(srt_file, encoding="utf-8")
            txt_path = os.path.splitext(srt_file)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                for sub in subs:
                    f.write(f"{sub.text}\n")
            self._log(f"TXT: {txt_path}")
        except Exception:
            pass

    def update_progress(self, ocr=None, frame_extract=None, post=None):
        if ocr is not None:
            self.progress_ocr = max(0, min(100, ocr))
        if frame_extract is not None:
            self.progress_frame_extract = max(0, min(100, frame_extract))
        if post is not None:
            self.progress_post = max(0, min(100, post))
        self._report_progress()

    def _report_progress(self):
        if self.progress_callback:
            total = (self.progress_frame_extract + self.progress_ocr + self.progress_post) / 3
            self.progress_callback(total)

    def _log(self, *args, **kwargs):
        print(*args, **kwargs)
