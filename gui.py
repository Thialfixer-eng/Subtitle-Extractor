import os
import re
import copy
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from backend.config import config, VERSION
from backend.bean.subtitle_area import SubtitleArea
from backend.subtitle_extractor import SubtitleExtractor
from backend.translator import translate_srt_file, DEEPL_LANGUAGES, SRC_LANGUAGES, load_config, save_config, SrtParser, AssParser
from backend.dictionary import lookup as dict_lookup
from backend import wordlist


_HAS_CUDA = False
_CUDA_CHECKED = False


def _check_cuda():
    global _HAS_CUDA, _CUDA_CHECKED
    if _CUDA_CHECKED:
        return _HAS_CUDA
    _CUDA_CHECKED = True
    try:
        import paddle
        _HAS_CUDA = paddle.is_compiled_with_cuda()
        if _HAS_CUDA:
            try:
                _HAS_CUDA = len(paddle.static.cuda_places()) > 0
            except Exception:
                _HAS_CUDA = False
    except Exception:
        _HAS_CUDA = False
    return _HAS_CUDA

LANGUAGES = [
    ("ch", "Chinese"),
    ("en", "English"),
    ("japan", "Japanese"),
    ("korean", "Korean"),
    ("fr", "French"),
    ("de", "German"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("it", "Italian"),
    ("ru", "Russian"),
    ("ar", "Arabic"),
    ("vi", "Vietnamese"),
    ("tr", "Turkish"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("chinese_cht", "Chinese Traditional"),
]


class TextHandler:
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.update_idletasks()

    def flush(self):
        pass


class VideoPreviewWindow:
    HANDLE_SIZE = 8

    def __init__(self, parent, video_path, on_confirm, sub_area=None, wm_area=None):
        self.parent = parent
        self.video_path = video_path
        self.on_confirm = on_confirm

        self.cap = cv2.VideoCapture(video_path)
        self.vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vid_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        screen_w = self.parent.winfo_screenwidth()
        screen_h = self.parent.winfo_screenheight()
        max_w = min(960, screen_w - 120)
        max_h = min(700, screen_h - 160)
        ratio = self.vid_w / self.vid_h
        if max_w / max_h > ratio:
            self.disp_w = int(max_h * ratio)
            self.disp_h = max_h
        else:
            self.disp_w = max_w
            self.disp_h = int(max_w / ratio)
        self.scale = self.disp_w / self.vid_w

        self.sub_rect = None
        self.wm_rect = None
        if sub_area and not sub_area.is_empty():
            self.sub_rect = self._v2d_rect(sub_area.xmin, sub_area.ymin, sub_area.xmax, sub_area.ymax)
        if wm_area and not wm_area.is_empty():
            self.wm_rect = self._v2d_rect(wm_area.xmin, wm_area.ymin, wm_area.xmax, wm_area.ymax)

        self.active_mode = None
        self.drag_state = None
        self.drag_target = None
        self.drag_handle = None
        self.drag_start_xy = (0, 0)
        self.drag_start_rect = None
        self.current_frame_idx = 0
        self._disp_photo = None
        self._bg_image_id = None

        self._build_ui()
        self._set_mode("subtitle")
        self._load_frame(0)
        self._center_window()

    def _v2d(self, v):
        return int(v * self.scale)

    def _d2v(self, d):
        return int(d / self.scale) if self.scale > 0 else 0

    def _v2d_rect(self, x1, y1, x2, y2):
        return (self._v2d(x1), self._v2d(y1), self._v2d(x2), self._v2d(y2))

    def _d2v_rect(self, x1, y1, x2, y2):
        return (self._d2v(x1), self._d2v(y1), self._d2v(x2), self._d2v(y2))

    def _center_window(self):
        self.win.update_idletasks()
        pw = self.parent.winfo_width()
        ph = self.parent.winfo_height()
        px = self.parent.winfo_x()
        py = self.parent.winfo_y()
        ww = self.win.winfo_reqwidth()
        wh = self.win.winfo_reqheight()
        x = px + (pw - ww) // 2
        y = py + (ph - wh) // 2
        self.win.geometry(f"+{max(0,x)}+{max(0,y)}")

    def _build_ui(self):
        self.win = tk.Toplevel(self.parent)
        self.win.title(f"Preview — {os.path.basename(self.video_path)}")
        self.win.transient(self.parent)
        self.win.grab_set()
        self.win.resizable(False, False)

        top = ttk.Frame(self.win)
        top.pack(fill=tk.X, padx=6, pady=4)

        self.sub_btn = tk.Button(top, text="▢ Subtitle Area",
                                  command=lambda: self._set_mode("subtitle"),
                                  relief=tk.RAISED, bd=2, padx=8)
        self.sub_btn.pack(side=tk.LEFT, padx=2)

        self.wm_btn = tk.Button(top, text="▣ Watermark Area",
                                 command=lambda: self._set_mode("watermark"),
                                 relief=tk.RAISED, bd=2, padx=8)
        self.wm_btn.pack(side=tk.LEFT, padx=2)

        sep = ttk.Frame(top, width=2, relief=tk.SUNKEN)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        ttk.Button(top, text="Clear Active", command=self._clear_active).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Clear All", command=self._clear_all).pack(side=tk.LEFT, padx=2)

        sep2 = ttk.Frame(top, width=2, relief=tk.SUNKEN)
        sep2.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        ttk.Label(top, text=f"{self.vid_w}x{self.vid_h}").pack(side=tk.LEFT, padx=4)

        self.canvas = tk.Canvas(self.win, width=self.disp_w, height=self.disp_h,
                                bg="black", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=6, pady=2)

        seek_frame = ttk.Frame(self.win)
        seek_frame.pack(fill=tk.X, padx=6, pady=(0, 2))

        btn_f = ttk.Frame(seek_frame)
        btn_f.pack(side=tk.LEFT)
        for lbl, delta in [("⏮", -50), ("◀", -1), ("▶", 1), ("⏭", 50)]:
            ttk.Button(btn_f, text=lbl, width=3,
                       command=lambda d=delta: self._seek_rel(d)).pack(side=tk.LEFT, padx=1)

        self.seek_var = tk.DoubleVar(value=0)
        self.seek = ttk.Scale(seek_frame, from_=0, to=self.vid_frames - 1,
                               variable=self.seek_var, orient=tk.HORIZONTAL,
                               command=self._on_seek)
        self.seek.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.frame_label = ttk.Label(seek_frame, width=16, anchor=tk.CENTER,
                                      text=f"Frame 0 / {self.vid_frames - 1}")
        self.frame_label.pack(side=tk.LEFT)

        self.info_var = tk.StringVar()
        ttk.Label(self.win, textvariable=self.info_var, foreground="#888",
                   font=("", 8)).pack(fill=tk.X, padx=6, pady=(0, 2))

        bottom = ttk.Frame(self.win)
        bottom.pack(fill=tk.X, padx=6, pady=(0, 6))

        self.coord_var = tk.StringVar()
        ttk.Label(bottom, textvariable=self.coord_var, font=("Consolas", 9)).pack(side=tk.LEFT)

        bf = ttk.Frame(bottom)
        bf.pack(side=tk.RIGHT)
        ttk.Button(bf, text="Cancel", command=self._on_cancel).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Apply", command=self._on_apply).pack(side=tk.LEFT, padx=2)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_motion)
        self.win.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.win.bind("<Delete>", lambda e: self._clear_active())
        self.win.bind("<BackSpace>", lambda e: self._clear_active())
        self.win.bind("<Escape>", lambda e: self._on_cancel())
        self.win.bind("<Return>", lambda e: self._on_apply())

    def _set_mode(self, mode):
        self.active_mode = mode
        self.sub_btn.configure(relief=tk.SUNKEN if mode == "subtitle" else tk.RAISED)
        self.wm_btn.configure(relief=tk.SUNKEN if mode == "watermark" else tk.RAISED)
        if mode == "subtitle":
            self.info_var.set("Draw subtitle area (green) — click & drag to create, drag rect to move, drag handles to resize")
        else:
            self.info_var.set("Draw watermark exclusion area (red) — click & drag to create, drag rect to move, drag handles to resize")
        self._redraw()

    def _clear_active(self):
        if self.active_mode == "subtitle":
            self.sub_rect = None
        elif self.active_mode == "watermark":
            self.wm_rect = None
        self.drag_state = None
        self._redraw()

    def _clear_all(self):
        self.sub_rect = None
        self.wm_rect = None
        self.drag_state = None
        self._redraw()

    def _load_frame(self, frame_idx):
        frame_idx = max(0, min(frame_idx, self.vid_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        self._orig_frame = frame.copy()
        self.current_frame_idx = frame_idx

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        disp = cv2.resize(rgb, (self.disp_w, self.disp_h), interpolation=cv2.INTER_LINEAR)
        self._disp_photo = ImageTk.PhotoImage(Image.fromarray(disp))

        if self._bg_image_id is not None:
            try:
                self.canvas.delete(self._bg_image_id)
            except Exception:
                pass
        self._bg_image_id = self.canvas.create_image(0, 0, image=self._disp_photo, anchor=tk.NW)

        self.frame_label.configure(text=f"{frame_idx} / {self.vid_frames - 1}")
        self._redraw()

    def _on_seek(self, value):
        idx = int(float(value))
        if idx != self.current_frame_idx:
            self._load_frame(idx)

    def _seek_rel(self, delta):
        idx = max(0, min(self.current_frame_idx + delta, self.vid_frames - 1))
        self.seek_var.set(idx)
        self._load_frame(idx)

    def _get_handle_rects(self, rect):
        x1, y1, x2, y2 = rect
        h = self.HANDLE_SIZE
        hh = h // 2
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return {
            "tl": (x1 - hh, y1 - hh, x1 + hh, y1 + hh),
            "tm": (cx - hh, y1 - hh, cx + hh, y1 + hh),
            "tr": (x2 - hh, y1 - hh, x2 + hh, y1 + hh),
            "ml": (x1 - hh, cy - hh, x1 + hh, cy + hh),
            "mr": (x2 - hh, cy - hh, x2 + hh, cy + hh),
            "bl": (x1 - hh, y2 - hh, x1 + hh, y2 + hh),
            "bm": (cx - hh, y2 - hh, cx + hh, y2 + hh),
            "br": (x2 - hh, y2 - hh, x2 + hh, y2 + hh),
        }

    def _hit_test(self, x, y):
        targets = []
        if self.sub_rect:
            targets.append(("subtitle", self.sub_rect))
        if self.wm_rect:
            targets.append(("watermark", self.wm_rect))

        for target, rect in targets:
            handles = self._get_handle_rects(rect)
            for hname, hr in handles.items():
                if hr[0] <= x <= hr[2] and hr[1] <= y <= hr[3]:
                    return (target, hname)
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                return (target, None)
        return (None, None)

    def _normalize_rect(self, rect):
        if rect is None:
            return None
        x1, y1, x2, y2 = rect
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def _on_press(self, event):
        x, y = event.x, event.y
        target, handle = self._hit_test(x, y)

        if target and handle is not None:
            self.drag_state = "resizing"
            self.drag_target = target
            self.drag_handle = handle
            self.drag_start_xy = (x, y)
            rect = self.sub_rect if target == "subtitle" else self.wm_rect
            self.drag_start_rect = rect
            self._set_mode(target)
        elif target and handle is None:
            self.drag_state = "moving"
            self.drag_target = target
            self.drag_handle = None
            self.drag_start_xy = (x, y)
            rect = self.sub_rect if target == "subtitle" else self.wm_rect
            self.drag_start_rect = rect
            self._set_mode(target)
        elif self.active_mode:
            self.drag_state = "drawing"
            self.drag_target = self.active_mode
            self.drag_handle = None
            self.drag_start_xy = (x, y)
            self.drag_start_rect = None
            if self.active_mode == "subtitle":
                self.sub_rect = (x, y, x, y)
            else:
                self.wm_rect = (x, y, x, y)

    def _on_drag(self, event):
        x, y = event.x, event.y
        if self.drag_state == "drawing":
            sx, sy = self.drag_start_xy
            rect = (min(sx, x), min(sy, y), max(sx, x), max(sy, y))
            if self.drag_target == "subtitle":
                self.sub_rect = rect
            else:
                self.wm_rect = rect
            self._redraw()
        elif self.drag_state == "moving":
            dx = x - self.drag_start_xy[0]
            dy = y - self.drag_start_xy[1]
            sr = self.drag_start_rect
            if sr:
                rect = (sr[0] + dx, sr[1] + dy, sr[2] + dx, sr[3] + dy)
                if self.drag_target == "subtitle":
                    self.sub_rect = rect
                else:
                    self.wm_rect = rect
                self._redraw()
        elif self.drag_state == "resizing":
            dx = x - self.drag_start_xy[0]
            dy = y - self.drag_start_xy[1]
            sr = self.drag_start_rect
            if sr:
                x1, y1, x2, y2 = sr
                h = self.drag_handle
                if "l" in h:
                    x1 += dx
                if "r" in h:
                    x2 += dx
                if "t" in h:
                    y1 += dy
                if "b" in h:
                    y2 += dy
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                rect = (x1, y1, x2, y2)
                if self.drag_target == "subtitle":
                    self.sub_rect = rect
                else:
                    self.wm_rect = rect
                self._redraw()

    def _on_release(self, event):
        self.drag_state = None
        self.drag_target = None
        self.drag_handle = None
        if self.sub_rect:
            self.sub_rect = self._normalize_rect(self.sub_rect)
        if self.wm_rect:
            self.wm_rect = self._normalize_rect(self.wm_rect)
        self._redraw()

    def _on_motion(self, event):
        x, y = event.x, event.y
        target, handle = self._hit_test(x, y)
        if handle:
            h = handle
            if h in ("tl", "br"):
                self.canvas.configure(cursor="size_nw_se")
            elif h in ("tr", "bl"):
                self.canvas.configure(cursor="size_ne_sw")
            elif h in ("tm", "bm"):
                self.canvas.configure(cursor="size_we")
            elif h in ("ml", "mr"):
                self.canvas.configure(cursor="size_ns")
            else:
                self.canvas.configure(cursor="crosshair")
        elif target:
            self.canvas.configure(cursor="fleur")
        elif self.active_mode:
            self.canvas.configure(cursor="crosshair")
        else:
            self.canvas.configure(cursor="arrow")

    def _redraw(self):
        self.canvas.delete("overlay")

        if self.sub_rect:
            r = self._normalize_rect(self.sub_rect)
            if r:
                self.sub_rect = r
                x1, y1, x2, y2 = r
                active = self.active_mode == "subtitle"
                outline = "#00FF00" if active else "#44AA44"
                fill = "#00FF00" if active else "#44AA44"
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline,
                                              width=2, stipple="gray25", fill=fill, tags="overlay")
                for hr in self._get_handle_rects(r).values():
                    self.canvas.create_rectangle(hr, fill=outline, outline="white",
                                                  width=1, tags="overlay")

        if self.wm_rect:
            r = self._normalize_rect(self.wm_rect)
            if r:
                self.wm_rect = r
                x1, y1, x2, y2 = r
                active = self.active_mode == "watermark"
                outline = "#FF4444" if active else "#AA4444"
                fill = "#FF4444" if active else "#AA4444"
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline,
                                              width=2, stipple="gray25", fill=fill, tags="overlay")
                for hr in self._get_handle_rects(r).values():
                    self.canvas.create_rectangle(hr, fill=outline, outline="white",
                                                  width=1, tags="overlay")

        self._update_coord_display()

    def _update_coord_display(self):
        parts = []
        if self.sub_rect:
            r = self._normalize_rect(self.sub_rect)
            vx1, vy1, vx2, vy2 = self._d2v_rect(*r)
            parts.append(f"SUB: x=[{vx1}..{vx2}] y=[{vy1}..{vy2}]")
        if self.wm_rect:
            r = self._normalize_rect(self.wm_rect)
            vx1, vy1, vx2, vy2 = self._d2v_rect(*r)
            parts.append(f"WM: x=[{vx1}..{vx2}] y=[{vy1}..{vy2}]")
        if not parts:
            parts.append("No areas defined — click and drag to draw")
        res = "  |  ".join(parts)
        self.coord_var.set(f"Video {self.vid_w}x{self.vid_h}  |  {res}")

    def _on_apply(self):
        sub = None
        wm = None
        if self.sub_rect:
            r = self._normalize_rect(self.sub_rect)
            vx1, vy1, vx2, vy2 = self._d2v_rect(*r)
            sub = SubtitleArea(xmin=vx1, xmax=vx2, ymin=vy1, ymax=vy2)
        if self.wm_rect:
            r = self._normalize_rect(self.wm_rect)
            vx1, vy1, vx2, vy2 = self._d2v_rect(*r)
            wm = SubtitleArea(xmin=vx1, xmax=vx2, ymin=vy1, ymax=vy2)
        self.on_confirm(sub, wm)
        self._cleanup()

    def _on_cancel(self):
        self.on_confirm(None, None)
        self._cleanup()

    def _cleanup(self):
        if self.cap:
            self.cap.release()
        try:
            self.win.destroy()
        except Exception:
            pass


class SubtitleExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Subtitle Extractor v{VERSION}")
        self.root.geometry("1100x700")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.style = ttk.Style()
        saved_theme = load_config().get("theme", "Jasny")
        if saved_theme not in self.THEMES:
            saved_theme = "Jasny"
        self.theme_var = tk.StringVar(value=saved_theme)

        self.video_paths = []
        self.running = False
        self.extractor = None
        self.extraction_thread = None
        self._cancel_requested = False
        self._start_time = None

        self._build_ui()
        self._set_icon()

    THEMES = {
        "Jasny": {
            "bg": "#f0f0f0", "fg": "#000000", "sel_bg": "#0078d4", "input_bg": "#ffffff",
            "tree_bg": "#ffffff", "btn_bg": "#f0f0f0", "btn_active": "#e5f3ff",
            "trough": "#e0e0e0", "progress": "#0078d4", "separator": "#cccccc",
            "base_font_size": 9,
        },
        "Ciemny": {
            "bg": "#121212", "fg": "#e0e0e0", "sel_bg": "#094771", "input_bg": "#080808",
            "tree_bg": "#0e0e0e", "btn_bg": "#1c1c1c", "btn_active": "#2a2a2a",
            "trough": "#080808", "progress": "#0e639c", "separator": "#2a2a2a",
            "base_font_size": 10,
        },
        "Fioletowy": {
            "bg": "#2d1b4e", "fg": "#e8d5ff", "sel_bg": "#b44bd4", "input_bg": "#3d2569",
            "tree_bg": "#35225c", "btn_bg": "#4a2d7a", "btn_active": "#5e3a96",
            "trough": "#3d2569", "progress": "#d46bf0", "separator": "#5a3d80",
            "base_font_size": 10,
        },
        "Czerwony": {
            "bg": "#4a1212", "fg": "#ffd0d0", "sel_bg": "#e83030", "input_bg": "#5c1a1a",
            "tree_bg": "#521616", "btn_bg": "#6e2020", "btn_active": "#8a2a2a",
            "trough": "#5c1a1a", "progress": "#ff4040", "separator": "#753030",
            "base_font_size": 10,
        },
        "Niebieski": {
            "bg": "#0a2647", "fg": "#c8e0ff", "sel_bg": "#2d8fdb", "input_bg": "#12355c",
            "tree_bg": "#0e2e52", "btn_bg": "#16426e", "btn_active": "#1e558a",
            "trough": "#12355c", "progress": "#4db8ff", "separator": "#2a5480",
            "base_font_size": 10,
        },
        "Z\u0142oty": {
            "bg": "#3d2e0a", "fg": "#fff0c8", "sel_bg": "#d4a42d", "input_bg": "#4e3d14",
            "tree_bg": "#453510", "btn_bg": "#5e4a1a", "btn_active": "#78602a",
            "trough": "#4e3d14", "progress": "#f0c040", "separator": "#6b5a30",
            "base_font_size": 10,
        },
        "Zielony": {
            "bg": "#0a3d0a", "fg": "#c8ffc8", "sel_bg": "#2ddb2d", "input_bg": "#144e14",
            "tree_bg": "#0e4510", "btn_bg": "#1a5e1a", "btn_active": "#287828",
            "trough": "#144e14", "progress": "#40ff40", "separator": "#307030",
            "base_font_size": 10,
        },
        "R\u00f3\u017cowy": {
            "bg": "#3d0a2e", "fg": "#ffc8f0", "sel_bg": "#db2db4", "input_bg": "#4e143d",
            "tree_bg": "#451035", "btn_bg": "#5e1a4a", "btn_active": "#78285e",
            "trough": "#4e143d", "progress": "#ff40d4", "separator": "#703060",
            "base_font_size": 10,
        },
        "Pomara\u0144czowy": {
            "bg": "#3d1e08", "fg": "#ffe0c0", "sel_bg": "#db6e0a", "input_bg": "#4e2a10",
            "tree_bg": "#45240c", "btn_bg": "#5e3418", "btn_active": "#784520",
            "trough": "#4e2a10", "progress": "#ff8830", "separator": "#704830",
            "base_font_size": 10,
        },
        "Turkusowy": {
            "bg": "#08303d", "fg": "#c0fff5", "sel_bg": "#0ad4db", "input_bg": "#10404e",
            "tree_bg": "#0c3845", "btn_bg": "#18505e", "btn_active": "#206878",
            "trough": "#10404e", "progress": "#30fff0", "separator": "#306070",
            "base_font_size": 10,
        },
        "Granatowy": {
            "bg": "#0a0a3d", "fg": "#c8c8ff", "sel_bg": "#2d2ddb", "input_bg": "#14144e",
            "tree_bg": "#0e0e45", "btn_bg": "#1a1a5e", "btn_active": "#282878",
            "trough": "#14144e", "progress": "#4040ff", "separator": "#303070",
            "base_font_size": 10,
        },
        "Szary": {
            "bg": "#1a1a1a", "fg": "#e0e0e0", "sel_bg": "#cc6600", "input_bg": "#252525",
            "tree_bg": "#202020", "btn_bg": "#2d2d2d", "btn_active": "#3a3a3a",
            "trough": "#252525", "progress": "#ff7700", "separator": "#353535",
            "base_font_size": 10,
        },
        "Le\u015bny": {
            "bg": "#0a2e14", "fg": "#c8ffd0", "sel_bg": "#2ddb4e", "input_bg": "#103d1e",
            "tree_bg": "#0e3515", "btn_bg": "#184a28", "btn_active": "#226038",
            "trough": "#103d1e", "progress": "#40ff60", "separator": "#306040",
            "base_font_size": 10,
        },
        "Wi\u015bniowy": {
            "bg": "#3d0808", "fg": "#ffc0c0", "sel_bg": "#db1a1a", "input_bg": "#4e1010",
            "tree_bg": "#450c0c", "btn_bg": "#5e1818", "btn_active": "#782222",
            "trough": "#4e1010", "progress": "#ff3030", "separator": "#703030",
            "base_font_size": 10,
        },
        "Lawendowy": {
            "bg": "#e8e0f0", "fg": "#2a1035", "sel_bg": "#9b59b6", "input_bg": "#f8f4ff",
            "tree_bg": "#f8f4ff", "btn_bg": "#d8d0e8", "btn_active": "#c8b8e0",
            "trough": "#d8d0e8", "progress": "#9b59b6", "separator": "#c8b8d8",
            "base_font_size": 9,
        },
    }

    def _set_icon(self):
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
        if os.path.isfile(logo_path):
            try:
                img = tk.PhotoImage(file=logo_path)
                self.root.iconphoto(True, img)
                self._logo_img = img
            except Exception:
                pass

    def _build_ui(self):
        menubar = tk.Menu(self.root, tearoff=False)
        theme_menu = tk.Menu(menubar, tearoff=False)
        for name in self.THEMES:
            theme_menu.add_radiobutton(label=name, variable=self.theme_var, value=name,
                                       command=lambda n=name: self._apply_theme(n))
        menubar.add_cascade(label="Themes", menu=theme_menu)
        self.root.config(menu=menubar)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tab_extract = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab_extract, text="Extraction")

        tab_translate = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab_translate, text="Translation")

        tab_dict = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab_dict, text="Dictionary")

        tab_wordlist = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab_wordlist, text="Word Lists")

        tab_editor = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab_editor, text="Subtitle Editor")

        self._build_extract_ui(tab_extract)
        self._build_translate_ui(tab_translate)
        self._build_dict_ui(tab_dict)
        self._build_wordlist_ui(tab_wordlist)
        self._build_editor_ui(tab_editor)

        self._apply_theme(self.theme_var.get())
        self.root.state("zoomed")
        self._log(f"Subtitle Extractor v{VERSION} ready")

    def _apply_theme(self, name):
        if getattr(self, "_current_theme", None) == name:
            return
        self._current_theme = name
        cfg = load_config()
        cfg["theme"] = name
        save_config(cfg)
        import tkinter.font as tkfont
        t = self.THEMES[name]
        is_dark = t["base_font_size"] > 9
        base_font = tkfont.nametofont("TkDefaultFont")
        fixed_font = tkfont.nametofont("TkFixedFont")
        bg, fg = t["bg"], t["fg"]
        sel_bg, input_bg = t["sel_bg"], t["input_bg"]
        tree_bg, btn_bg, btn_active = t["tree_bg"], t["btn_bg"], t["btn_active"]
        trough, progress, sep = t["trough"], t["progress"], t["separator"]

        self.style.theme_use("clam" if name != "Jasny" else "vista")
        self.root.configure(bg=bg)
        base_font.configure(family="Segoe UI", size=t["base_font_size"])
        fixed_font.configure(family="Consolas", size=t["base_font_size"])

        fb = input_bg if is_dark else "#ffffff"
        tb = tree_bg if is_dark else "#ffffff"
        hlb = "#444444" if is_dark else sep

        self.style.configure(".", background=bg, foreground=fg, fieldbackground=fb, selectbackground=sel_bg, selectforeground=fg, font=base_font, borderwidth=0, focuscolor="#555555")
        self.style.configure("TLabel", background=bg, foreground=fg)
        self.style.configure("TFrame", background=bg)
        self.style.configure("TLabelframe", background=bg, foreground=fg, bordercolor=hlb, lightcolor=hlb, darkcolor=hlb, borderwidth=1)
        self.style.configure("TNotebook", background=bg, foreground=fg, tabmargins=[2, 5, 2, 0], borderwidth=0)
        self.style.configure("TNotebook.Tab", background=bg, foreground="#999999" if is_dark else fg, padding=[12, 4], borderwidth=0)
        self.style.map("TNotebook.Tab", background=[("selected", bg)], foreground=[("selected", fg)])
        self.style.configure("TButton", background=btn_bg, foreground=fg, borderwidth=1, focusthickness=2, focuscolor="#555555", relief=tk.FLAT)
        self.style.map("TButton", background=[("active", btn_active), ("pressed", sel_bg)], relief=[("pressed", tk.SUNKEN)])
        self.style.configure("TCheckbutton", background=bg, foreground=fg)
        self.style.map("TCheckbutton", background=[("active", btn_bg)])
        self.style.configure("TRadiobutton", background=bg, foreground=fg)
        self.style.map("TRadiobutton", background=[("active", btn_bg)])
        self.style.configure("TEntry", fieldbackground=fb, foreground=fg, borderwidth=1, lightcolor=hlb, darkcolor=hlb)
        self.style.configure("TSpinbox", fieldbackground=fb, foreground=fg, borderwidth=1, lightcolor=hlb, darkcolor=hlb)
        self.style.configure("TCombobox", fieldbackground=fb, foreground=fg, background=bg, arrowcolor=fg, borderwidth=1, lightcolor=hlb, darkcolor=hlb)
        self.style.map("TCombobox", fieldbackground=[("readonly", fb)], foreground=[("readonly", fg)], background=[("readonly", bg)])
        self.style.configure("TScale", background=bg, foreground=fg, troughcolor=trough, borderwidth=0)
        self.style.configure("TProgressbar", background=progress, troughcolor=trough, borderwidth=0)
        self.style.configure("Treeview", background=tb, foreground=fg, fieldbackground=tb, borderwidth=0)
        self.style.map("Treeview", background=[("selected", sel_bg)])
        self.style.configure("Treeview.Heading", background=btn_bg, foreground=fg, borderwidth=1, lightcolor=hlb, darkcolor=hlb, relief=tk.FLAT)
        self.style.map("Treeview.Heading", background=[("active", btn_active)])
        self.style.configure("TSeparator", background=hlb)

        self.log_text.configure(bg=fb, fg=fg, insertbackground=fg, font=fixed_font, relief=tk.FLAT, borderwidth=0, highlightbackground=hlb, highlightcolor=hlb, highlightthickness=1)
        self.trans_log.configure(bg=fb, fg=fg, insertbackground=fg, font=fixed_font, relief=tk.FLAT, borderwidth=0, highlightbackground=hlb, highlightcolor=hlb, highlightthickness=1)
        self.dict_menu.configure(bg=btn_bg, fg=fg, activebackground=sel_bg, activeforeground=fg, borderwidth=0)
        if hasattr(self, 'wl_listbox'):
            self.wl_listbox.configure(bg=fb, fg=fg, selectbackground=sel_bg, selectforeground=fg, highlightbackground=hlb, highlightcolor=hlb, highlightthickness=1)
        self.root.tk.eval(f"option add *TCombobox*Listbox.background {fb} widgetDefault")
        self.root.tk.eval(f"option add *TCombobox*Listbox.foreground {fg} widgetDefault")
        self.root.tk.eval(f"option add *TCombobox*Listbox.selectBackground {sel_bg} widgetDefault")
        self.root.tk.eval(f"option add *TCombobox*Listbox.selectForeground {fg} widgetDefault")

    def _build_extract_ui(self, parent):
        row = 0

        ttk.Label(parent, text="Video File(s):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.file_entry = ttk.Entry(parent)
        self.file_entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        file_btn_f = ttk.Frame(parent)
        file_btn_f.grid(row=row, column=2, padx=0, pady=2)
        self.preview_btn = ttk.Button(file_btn_f, text="Preview", command=self._open_preview, state=tk.DISABLED)
        self.preview_btn.pack(side=tk.LEFT, padx=1)
        ttk.Button(file_btn_f, text="Browse", command=self._browse_files).pack(side=tk.LEFT, padx=1)
        row += 1

        ttk.Label(parent, text="Language:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.lang_var = tk.StringVar(value="ch")
        lang_menu = ttk.Combobox(parent, textvariable=self.lang_var, values=[f"{k} ({v})" for k, v in LANGUAGES], state="readonly", width=30)
        lang_menu.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Mode:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.mode_var = tk.StringVar(value="fast")
        mode_frame = ttk.Frame(parent)
        mode_frame.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        for mode in ["fast", "auto", "accurate"]:
            ttk.Radiobutton(mode_frame, text=mode, variable=self.mode_var, value=mode).pack(side=tk.LEFT, padx=2)
        self.mode_var.trace_add("write", self._on_mode_change)
        row += 1

        ttk.Label(parent, text="Area (xmin,xmax,ymin,ymax):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.area_entry = ttk.Entry(parent)
        self.area_entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(parent, text="(leave empty for auto)").grid(row=row, column=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(parent, text="Watermark area (xmin,xmax,ymin,ymax):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.watermark_entry = ttk.Entry(parent)
        self.watermark_entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(parent, text="(detections here are excluded)").grid(row=row, column=2, sticky=tk.W, pady=2)
        row += 1

        fps_frame = ttk.Frame(parent)
        fps_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)
        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="3")
        self.fps_spin = ttk.Spinbox(fps_frame, from_=1, to=10, textvariable=self.fps_var, width=5)
        self.fps_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(fps_frame, text="  Similarity:").pack(side=tk.LEFT)
        self.sim_var = tk.StringVar(value="80")
        ttk.Spinbox(fps_frame, from_=50, to=100, textvariable=self.sim_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(fps_frame, text="  Drop Score:").pack(side=tk.LEFT)
        self.drop_var = tk.StringVar(value="75")
        ttk.Spinbox(fps_frame, from_=0, to=100, textvariable=self.drop_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(fps_frame, text="  Workers:").pack(side=tk.LEFT)
        self.workers_var = tk.StringVar(value="1")
        ttk.Spinbox(fps_frame, from_=1, to=16, textvariable=self.workers_var, width=3).pack(side=tk.LEFT, padx=5)
        row += 1

        opt_frame = ttk.Frame(parent)
        opt_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)

        ttk.Label(opt_frame, text="CPU mode", foreground="#888",
                   font=("", 8)).pack(side=tk.LEFT, padx=2)

        self.txt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Generate TXT", variable=self.txt_var).pack(side=tk.LEFT, padx=2)
        self.cache_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="Keep cache", variable=self.cache_var).pack(side=tk.LEFT, padx=2)
        row += 1

        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=5)

        self.start_btn = ttk.Button(btn_frame, text="Start", command=self._start_extraction)
        self.start_btn.pack(side=tk.LEFT, padx=2)

        self.pause_btn = ttk.Button(btn_frame, text="Pause", command=self._toggle_pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=2)

        self.cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self._cancel_extraction, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=2)
        row += 1

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("green.Horizontal.TProgressbar", background="#22c55e", troughcolor="#e2e8f0", bordercolor="#22c55e", lightcolor="#22c55e", darkcolor="#22c55e")

        self.progress = ttk.Progressbar(parent, mode="determinate", value=0, style="green.Horizontal.TProgressbar")
        self.progress.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=2)
        self.progress_label = ttk.Label(parent, text="")
        self.progress_label.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=0)
        row += 1

        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.grid(row=row, column=0, columnspan=3, sticky=tk.NSEW, pady=5)
        self.log_text = tk.Text(log_frame, height=14, wrap=tk.WORD, state=tk.NORMAL)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(row, weight=1)

    def _on_mode_change(self, *_args):
        mode = self.mode_var.get()
        is_fast = mode == "fast"
        defaults = {
            "fast": {"fps": "3", "sim": "80", "drop": "75", "workers": "4"},
            "auto": {"fps": "1", "sim": "75", "drop": "70", "workers": "2"},
            "accurate": {"fps": "1", "sim": "75", "drop": "70", "workers": "1"},
        }
        d = defaults.get(mode, defaults["fast"])
        self.fps_var.set(d["fps"])
        self.sim_var.set(d["sim"])
        self.drop_var.set(d["drop"])
        self.workers_var.set(d["workers"])
        self.fps_spin.configure(state=tk.NORMAL if is_fast else tk.DISABLED)

    def _build_translate_ui(self, parent):
        cfg = load_config()

        row = 0
        ttk.Label(parent, text="SRT File:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.srt_entry = ttk.Entry(parent)
        self.srt_entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(parent, text="Browse", command=self._browse_srt).grid(row=row, column=2, padx=0, pady=2)
        row += 1

        ttk.Label(parent, text="Source language:").grid(row=row, column=0, sticky=tk.W, pady=2)
        src_vals = [f"{k} ({v})" for k, v in SRC_LANGUAGES.items()]
        self.src_lang_var = tk.StringVar(value="auto (Auto-detect)")
        ttk.Combobox(parent, textvariable=self.src_lang_var, values=src_vals, state="readonly", width=30).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        row += 1

        ttk.Label(parent, text="Target language:").grid(row=row, column=0, sticky=tk.W, pady=2)
        tgt_vals = [f"{k} ({v})" for k, v in DEEPL_LANGUAGES.items()]
        self.tgt_lang_var = tk.StringVar(value="PL (Polish)")
        ttk.Combobox(parent, textvariable=self.tgt_lang_var, values=tgt_vals, state="readonly", width=30).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        row += 1

        svc_frame = ttk.Frame(parent)
        svc_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)
        ttk.Label(svc_frame, text="Service:").pack(side=tk.LEFT)
        self.svc_var = tk.StringVar(value=cfg.get("service", "local"))
        self.svc_combo = ttk.Combobox(svc_frame, textvariable=self.svc_var, values=["dictionary", "dictionary2", "dictionary3", "baidu", "local", "libre (MyMemory)", "huggingface", "deepl", "openai", "google"], state="readonly", width=16)
        self.svc_combo.pack(side=tk.LEFT, padx=5)
        self.svc_combo.bind("<<ComboboxSelected>>", self._on_svc_change)
        self.api_key_label = ttk.Label(svc_frame, text="API Key:")
        self.api_key_label.pack(side=tk.LEFT, padx=(15, 0))
        self.api_key_var = tk.StringVar(value=self._get_saved_key(cfg))
        self.api_entry = ttk.Entry(svc_frame, textvariable=self.api_key_var, width=30, show="*")
        self.api_entry.pack(side=tk.LEFT, padx=5)
        row += 1

        tinfo = ttk.Label(parent, text="dictionary = gloss (first def)  |  dictionary2 = gloss (all defs)  |  local = offline  |  libre (MyMemory) = free  |  huggingface  |  deepl  |  openai  |  google",
                          foreground="#888", font=("", 7), wraplength=650)
        tinfo.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=0)
        row += 1

        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=5)
        self.translate_btn = ttk.Button(btn_frame, text="Translate", command=self._start_translation)
        self.translate_btn.pack(side=tk.LEFT, padx=2)
        self.translate_cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self._cancel_translation, state=tk.DISABLED)
        self.translate_cancel_btn.pack(side=tk.LEFT, padx=2)
        row += 1

        self.trans_progress = ttk.Progressbar(parent, mode="determinate", value=0)
        self.trans_progress.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=2)
        self.trans_progress_label = ttk.Label(parent, text="")
        self.trans_progress_label.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=0)
        row += 1

        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.grid(row=row, column=0, columnspan=3, sticky=tk.NSEW, pady=5)
        self.trans_log = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.NORMAL)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.trans_log.yview)
        self.trans_log.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.trans_log.pack(fill=tk.BOTH, expand=True)

        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(row, weight=1)

        self._on_svc_change()

    def _build_dict_ui(self, parent):
        row = 0

        ttk.Label(parent, text="Search:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.dict_entry = ttk.Entry(parent)
        self.dict_entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        self.dict_entry.bind("<Return>", lambda e: self._dict_lookup())
        btn_f = ttk.Frame(parent)
        btn_f.grid(row=row, column=2, padx=0, pady=2)
        ttk.Button(btn_f, text="Lookup", command=self._dict_lookup).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_f, text="Copy", command=self._dict_copy_selection).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_f, text="Save", command=self._dict_save_word).pack(side=tk.LEFT, padx=1)
        ttk.Label(btn_f, text="  Dict:").pack(side=tk.LEFT, padx=(8, 0))
        self.dict_lang_var = tk.StringVar(value="en")
        ttk.Combobox(btn_f, textvariable=self.dict_lang_var, values=["en", "pl", "cc-pl"], state="readonly", width=4).pack(side=tk.LEFT, padx=1)
        row += 1

        self.dict_mode_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.dict_mode_var, foreground="#555", font=("", 8)).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 2))
        row += 1

        columns = ("trad", "simp", "pinyin", "hsk", "strokes", "defs")
        self.dict_tree = ttk.Treeview(parent, columns=columns, show="headings", height=14)
        self.dict_tree.heading("trad", text="Traditional")
        self.dict_tree.heading("simp", text="Simplified")
        self.dict_tree.heading("pinyin", text="Pinyin")
        self.dict_tree.heading("hsk", text="HSK")
        self.dict_tree.heading("strokes", text="Strokes")
        self.dict_tree.heading("defs", text="Definitions")
        self.dict_tree.column("trad", width=100)
        self.dict_tree.column("simp", width=100)
        self.dict_tree.column("pinyin", width=150)
        self.dict_tree.column("hsk", width=40, anchor=tk.CENTER)
        self.dict_tree.column("strokes", width=55, anchor=tk.CENTER)
        self.dict_tree.column("defs", width=400)

        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.dict_tree.yview)
        self.dict_tree.configure(yscrollcommand=scrollbar.set)
        self.dict_tree.grid(row=row, column=0, columnspan=3, sticky=tk.NSEW, pady=2)
        scrollbar.grid(row=row, column=3, sticky=tk.NS, pady=2)
        row += 1

        self.dict_status_var = tk.StringVar(value="Enter Chinese, Pinyin, or English  |  en=CC-CEDICT, pl=WikDict, cc-pl=CC-CEDICT+PL")
        ttk.Label(parent, textvariable=self.dict_status_var, foreground="#888").grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)
        row += 1

        # Context menu + keyboard copy
        self.dict_menu = tk.Menu(self.root, tearoff=False)
        self.dict_menu.add_command(label="Copy", command=self._dict_copy_selection, accelerator="Ctrl+C")
        self.dict_menu.add_command(label="Save to word list", command=self._dict_save_word)
        self.dict_tree.bind("<Button-3>", self._dict_show_context_menu)
        self.dict_tree.bind("<Control-c>", lambda e: self._dict_copy_selection())
        self.dict_tree.bind("<Control-C>", lambda e: self._dict_copy_selection())

        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(row - 2, weight=1)

    def _dict_show_context_menu(self, event):
        if self.dict_tree.identify_row(event.y):
            self.dict_menu.tk_popup(event.x_root, event.y_root)

    def _dict_copy_selection(self, event=None):
        sel = self.dict_tree.selection()
        if not sel:
            return
        lines = []
        for item in sel:
            vals = self.dict_tree.item(item, "values")
            if vals[1] and vals[1].startswith("--"):
                lines.append(vals[0])
            else:
                parts = [v for v in vals if v]
                lines.append("\t".join(parts))
        text = "\n".join(lines)
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

    def _dict_save_word(self):
        sel = self.dict_tree.selection()
        if not sel:
            messagebox.showinfo("Save Word", "Select a word entry first")
            return
        item = self.dict_tree.item(sel[0], "values")
        if not item or not item[1]:
            messagebox.showinfo("Save Word", "Select a word entry (not a separator)")
            return
        lists = wordlist.load_lists()
        if not lists:
            result = messagebox.askyesno("Save Word", "No word lists exist. Create one now?")
            if not result:
                return
            self._wordlist_create_dialog()
            lists = wordlist.load_lists()
            if not lists:
                return

        top = tk.Toplevel(self.root)
        top.title("Save to Word List")
        top.transient(self.root)
        top.grab_set()
        top.resizable(False, False)

        ttk.Label(top, text="Select list:").pack(padx=10, pady=(10, 2))
        listbox = tk.Listbox(top, height=6, width=40)
        listbox.pack(padx=10, pady=2)
        for lst in lists:
            cnt = len(lst.get("words", []))
            listbox.insert(tk.END, f"{lst['name']}  ({cnt} words)")
        listbox.selection_set(0)

        def do_save():
            sel_idx = listbox.curselection()
            if not sel_idx:
                return
            name = lists[sel_idx[0]]["name"]
            entry = {
                "simp": item[1],
                "trad": item[0],
                "pinyin": item[2],
                "defs": item[5].split(" / ") if item[5] else [],
                "hsk": item[3],
                "strokes": item[4],
            }
            if wordlist.add_word(name, entry):
                self.dict_status_var.set(f"Saved '{entry['simp']}' to '{name}'")
                self._refresh_wordlist_display()
            top.destroy()

        bf = ttk.Frame(top)
        bf.pack(padx=10, pady=(5, 10))
        ttk.Button(bf, text="Save", command=do_save).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Cancel", command=top.destroy).pack(side=tk.LEFT, padx=2)

    def _wordlist_create_dialog(self, initial="", callback=None):
        top = tk.Toplevel(self.root)
        top.title("New Word List")
        top.transient(self.root)
        top.grab_set()
        top.resizable(False, False)

        ttk.Label(top, text="List name:").pack(padx=10, pady=(10, 2))
        var = tk.StringVar(value=initial)
        entry = ttk.Entry(top, textvariable=var, width=40)
        entry.pack(padx=10, pady=2)
        entry.focus_set()
        entry.select_range(0, tk.END)

        def do_create():
            name = var.get().strip()
            if not name:
                return
            if wordlist.add_list(name):
                self._refresh_wordlist_display()
                if callback:
                    callback(name)
            else:
                messagebox.showwarning("New List", f"List '{name}' already exists")
            top.destroy()

        entry.bind("<Return>", lambda e: do_create())
        bf = ttk.Frame(top)
        bf.pack(padx=10, pady=(5, 10))
        ttk.Button(bf, text="Create", command=do_create).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Cancel", command=top.destroy).pack(side=tk.LEFT, padx=2)

    def _dict_insert_separator(self, text):
        sep = "-- " + text + " " + "-" * 60
        self.dict_tree.insert("", tk.END, values=(sep, "", "", "", "", ""), tags=("sep",))

    def _dict_add_entry(self, r):
        vals = (r["trad"], r["simp"], r["pinyin"],
                r.get("hsk", ""), r.get("strokes", ""),
                " / ".join(r["defs"]))
        self.dict_tree.insert("", tk.END, values=vals)

    def _dict_lookup(self):
        text = self.dict_entry.get().strip()
        if not text:
            return
        lang = self.dict_lang_var.get()

        for item in self.dict_tree.get_children():
            self.dict_tree.delete(item)

        self.dict_status_var.set("Loading dictionary...")
        self.dict_mode_var.set("")
        self.root.update_idletasks()

        def do_lookup():
            try:
                result = dict_lookup(text, lang=lang)
                mode = result.get("mode", "unknown")
                entries = result.get("entries", [])
                characters = result.get("characters", [])
                total = result.get("total", 0)

                self.root.after(0, lambda m=mode: self.dict_mode_var.set({
                    "chinese": "Chinese search  |  * for wildcard  |  auto-detects pinyin/english",
                    "pinyin": "Pinyin search  |  tone numbers optional (ni3hao3 or nihao)  |  v for \u00fc",
                    "english": "English search  |  * for wildcard  |  case-insensitive",
                    "polish": "S\u0142ownik chi\u0144sko-polski  |  szukaj po chi\u0144sku lub polsku",
                }.get(m, "")))

                if not entries and not characters:
                    self.root.after(0, lambda: self.dict_status_var.set("No entries found"))
                    return

                # Insert entries grouped by section
                current_section = None
                for r in entries:
                    section = r.get("_section", "")
                    if section != current_section:
                        current_section = section
                        label = {
                            "exact": "\"  Exact match / Dok\u0142adne  \"",
                            "substring": "Contains / Zawiera \"" + text + "\"",
                            "pinyin": "Pinyin: " + text,
                            "english": "English: " + text,
                            "wildcard": "Pattern: " + text,
                            "wyszukiwanie PL": "Polish def / Po polsku: \"" + text + "\"",
                        }.get(section, section)
                        self.root.after(0, lambda l=label: self._dict_insert_separator(l))
                    self.root.after(0, lambda r=r: self._dict_add_entry(r))

                # Character breakdown
                if characters:
                    self.root.after(0, lambda: self._dict_insert_separator("Character breakdown"))
                    for ch_info in characters:
                        ch = ch_info["char"]
                        ch_entries = ch_info["entries"]
                        if ch_entries:
                            for r in ch_entries:
                                self.root.after(0, lambda r=r: self._dict_add_entry(r))
                        else:
                            vals = (ch, ch, "", "", "", "(no entry)")
                            self.root.after(0, lambda v=vals: self.dict_tree.insert("", tk.END, values=v))

                self.root.after(0, lambda: self.dict_status_var.set(
                    f"Found {total} entr{'y' if total == 1 else 'ies'} ({mode})"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda m=str(e): self.dict_status_var.set(f"Error: {m}"))

        threading.Thread(target=do_lookup, daemon=True).start()

    # ---- Word Lists UI ----

    def _build_wordlist_ui(self, parent):
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        ttk.Label(left, text="Listy słówek / Word Lists", font=("", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))

        lf_top = ttk.Frame(left)
        lf_top.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(lf_top, text="+ New List", command=self._wl_new_list).pack(side=tk.LEFT, padx=1)
        ttk.Button(lf_top, text="Rename", command=self._wl_rename_list).pack(side=tk.LEFT, padx=1)
        ttk.Button(lf_top, text="Delete", command=self._wl_delete_list).pack(side=tk.LEFT, padx=1)

        self.wl_listbox = tk.Listbox(left, height=8, exportselection=False)
        self.wl_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.wl_listbox.bind("<<ListboxSelect>>", self._wl_on_select)

        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        rf_top = ttk.Frame(right)
        rf_top.pack(fill=tk.X, pady=(0, 5))
        self.wl_title_var = tk.StringVar(value="Select a list")
        ttk.Label(rf_top, textvariable=self.wl_title_var, font=("", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(rf_top, text="Export Anki", command=self._wl_export_anki).pack(side=tk.RIGHT, padx=1)
        ttk.Button(rf_top, text="Delete Word", command=self._wl_delete_word).pack(side=tk.RIGHT, padx=1)

        columns = ("simp", "trad", "pinyin", "hsk", "defs")
        self.wl_tree = ttk.Treeview(right, columns=columns, show="headings", height=12)
        self.wl_tree.heading("simp", text="Simplified")
        self.wl_tree.heading("trad", text="Traditional")
        self.wl_tree.heading("pinyin", text="Pinyin")
        self.wl_tree.heading("hsk", text="HSK")
        self.wl_tree.heading("defs", text="Definitions")
        self.wl_tree.column("simp", width=100)
        self.wl_tree.column("trad", width=100)
        self.wl_tree.column("pinyin", width=150)
        self.wl_tree.column("hsk", width=40, anchor=tk.CENTER)
        self.wl_tree.column("defs", width=400)
        self.wl_tree.pack(fill=tk.BOTH, expand=True)

        self.wl_status_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.wl_status_var, foreground="#888").pack(anchor=tk.W, pady=(2, 0))

        self._refresh_wordlist_display()

    def _refresh_wordlist_display(self):
        self.wl_listbox.delete(0, tk.END)
        lists = wordlist.load_lists()
        for lst in lists:
            cnt = len(lst.get("words", []))
            self.wl_listbox.insert(tk.END, f"{lst['name']}  ({cnt})")

        if not self.wl_tree.get_children():
            for item in self.wl_tree.get_children():
                self.wl_tree.delete(item)
            self.wl_title_var.set("Select a list")

    def _wl_new_list(self):
        self._wordlist_create_dialog()

    def _wl_rename_list(self):
        sel = self.wl_listbox.curselection()
        if not sel:
            messagebox.showinfo("Rename", "Select a list first")
            return
        lists = wordlist.load_lists()
        old_name = lists[sel[0]]["name"]

        top = tk.Toplevel(self.root)
        top.title("Rename List")
        top.transient(self.root)
        top.grab_set()
        top.resizable(False, False)

        ttk.Label(top, text="New name:").pack(padx=10, pady=(10, 2))
        var = tk.StringVar(value=old_name)
        entry = ttk.Entry(top, textvariable=var, width=40)
        entry.pack(padx=10, pady=2)
        entry.focus_set()
        entry.select_range(0, tk.END)

        def do_rename():
            name = var.get().strip()
            if not name or name == old_name:
                top.destroy()
                return
            if wordlist.rename_list(old_name, name):
                self._refresh_wordlist_display()
            top.destroy()

        entry.bind("<Return>", lambda e: do_rename())
        bf = ttk.Frame(top)
        bf.pack(padx=10, pady=(5, 10))
        ttk.Button(bf, text="Rename", command=do_rename).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Cancel", command=top.destroy).pack(side=tk.LEFT, padx=2)

    def _wl_delete_list(self):
        sel = self.wl_listbox.curselection()
        if not sel:
            messagebox.showinfo("Delete", "Select a list first")
            return
        lists = wordlist.load_lists()
        name = lists[sel[0]]["name"]
        cnt = len(lists[sel[0]].get("words", []))
        if messagebox.askyesno("Delete", f"Delete list '{name}' ({cnt} words)?"):
            wordlist.delete_list(name)
            self._refresh_wordlist_display()

    def _wl_on_select(self, event=None):
        for item in self.wl_tree.get_children():
            self.wl_tree.delete(item)
        sel = self.wl_listbox.curselection()
        if not sel:
            self.wl_title_var.set("Select a list")
            return
        lists = wordlist.load_lists()
        lst = lists[sel[0]]
        self.wl_title_var.set(f"{lst['name']}  ({len(lst['words'])} words)")
        for w in lst["words"]:
            vals = (w.get("simp", ""), w.get("trad", ""), w.get("pinyin", ""),
                    w.get("hsk", ""), " / ".join(w.get("defs", [])))
            self.wl_tree.insert("", tk.END, values=vals)

    def _wl_delete_word(self):
        sel = self.wl_listbox.curselection()
        if not sel:
            return
        wsel = self.wl_tree.selection()
        if not wsel:
            messagebox.showinfo("Delete", "Select a word in the list")
            return
        lists = wordlist.load_lists()
        list_name = lists[sel[0]]["name"]
        idx = self.wl_tree.index(wsel[0])
        if messagebox.askyesno("Delete", f"Remove this word from '{list_name}'?"):
            wordlist.remove_word(list_name, idx)
            self._wl_on_select()

    def _wl_export_anki(self):
        sel = self.wl_listbox.curselection()
        if not sel:
            messagebox.showinfo("Export", "Select a list first")
            return
        lists = wordlist.load_lists()
        name = lists[sel[0]]["name"]
        cnt = len(lists[sel[0]].get("words", []))
        if cnt == 0:
            messagebox.showinfo("Export", "List is empty")
            return
        path = filedialog.asksaveasfilename(
            title="Export to Anki",
            defaultextension=".tsv",
            filetypes=[("TSV (Anki)", "*.tsv"), ("All files", "*.*")],
            initialfile=f"{name}_anki.tsv",
        )
        if path:
            n = wordlist.export_anki(name, path)
            if n > 0:
                self.wl_status_var.set(f"Exported {n} words to {os.path.basename(path)}")

    # ---- Subtitle Editor UI ----

    def _build_editor_ui(self, parent):
        row = 0
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text="SRT File:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.ed_srt_entry = ttk.Entry(parent)
        self.ed_srt_entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        btn_f = ttk.Frame(parent)
        btn_f.grid(row=row, column=2, padx=0, pady=2, sticky=tk.EW)
        ttk.Button(btn_f, text="Browse", width=6, command=self._ed_browse).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_f, text="Load", width=5, command=self._ed_load).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_f, text="Save", width=5, command=self._ed_save).pack(side=tk.LEFT, padx=1)
        ttk.Separator(btn_f, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=3, pady=2)
        ttk.Button(btn_f, text="Merge", width=5, command=self._ed_merge_lines).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_f, text="Split", width=5, command=self._ed_split_line).pack(side=tk.LEFT, padx=1)
        ttk.Separator(btn_f, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=3, pady=2)
        ttk.Button(btn_f, text="Shift", width=5, command=self._ed_shift_timings).pack(side=tk.LEFT, padx=1)
        ttk.Separator(btn_f, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=3, pady=2)
        self.ed_undo_btn = ttk.Button(btn_f, text="Undo", width=5, command=self._ed_undo, state=tk.DISABLED)
        self.ed_undo_btn.pack(side=tk.LEFT, padx=1)
        self.ed_redo_btn = ttk.Button(btn_f, text="Redo", width=5, command=self._ed_redo, state=tk.DISABLED)
        self.ed_redo_btn.pack(side=tk.LEFT, padx=1)
        ttk.Separator(btn_f, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=3, pady=2)
        ttk.Button(btn_f, text="Video", width=5, command=self._ed_open_video).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_f, text="Popup", width=5, command=self._ed_open_video_popup).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_f, text="Translate", width=8, command=self._ed_show_original).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_f, text="Bulk Ops", width=7, command=self._ed_bulk_operations).pack(side=tk.LEFT, padx=1)
        mb = tk.Menubutton(btn_f, text="Tools", relief=tk.RAISED, padx=3)
        mb.menu = tk.Menu(mb, tearoff=False)
        mb["menu"] = mb.menu
        mb.menu.add_command(label="Convert...", command=self._ed_convert)
        mb.menu.add_command(label="Send to Translation", command=self._ed_send_to_translation)
        mb.pack(side=tk.LEFT, padx=1)
        row += 1

        filter_f = ttk.Frame(parent)
        filter_f.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=(0, 2))
        ttk.Label(filter_f, text="Filter:").pack(side=tk.LEFT, padx=(0, 4))
        self.ed_filter_var = tk.StringVar()
        self.ed_filter_entry = ttk.Entry(filter_f, textvariable=self.ed_filter_var, width=30)
        self.ed_filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.ed_filter_entry.bind("<Return>", lambda e: self._ed_filter())
        ttk.Button(filter_f, text="Find", command=self._ed_filter, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Button(filter_f, text="Clear", command=self._ed_filter_clear, width=6).pack(side=tk.LEFT, padx=1)
        row += 1

        ed_container = ttk.Frame(parent)
        ed_container.grid(row=row, column=0, columnspan=4, sticky=tk.NSEW, pady=2)
        parent.rowconfigure(row, weight=1)
        row += 1

        self._ed_static_pw = ttk.PanedWindow(ed_container, orient=tk.HORIZONTAL)
        self._ed_static_pw.pack(fill=tk.BOTH, expand=True)

        self._ed_static_video_f = ttk.Frame(self._ed_static_pw)

        editor_f = ttk.Frame(self._ed_static_pw)
        self._ed_static_pw.add(editor_f, weight=1)

        tree_frame = ttk.Frame(editor_f)
        tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        columns = ("idx", "timing", "text")
        self.ed_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=10, selectmode="extended")
        self.ed_tree.heading("idx", text="#")
        self.ed_tree.heading("timing", text="Timing")
        self.ed_tree.heading("text", text="Text")
        self.ed_tree.column("idx", width=40, anchor=tk.CENTER)
        self.ed_tree.column("timing", width=200)
        self.ed_tree.column("text", width=500)
        self.ed_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.ed_tree.yview)
        self.ed_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.ed_tree.bind("<Double-1>", self._ed_edit_cell)
        self.ed_tree.bind("<Return>", self._ed_edit_cell)
        self.ed_tree.bind("<Button-3>", self._ed_context_menu)
        self.ed_tree.bind("<<TreeviewSelect>>", self._ed_on_select)
        self.ed_tree.bind("<Button-1>", self._ed_tree_idx_click, add="+")
        self.ed_tree.bind("<Double-1>", self._ed_tree_idx_double, add="+")
        self._ed_context = tk.Menu(self.root, tearoff=False)
        self._ed_context.add_command(label="Look up in Dictionary", command=self._ed_lookup_dict)
        self._ed_context.add_command(label="Translate Mode", command=self._ed_show_original)
        self._ed_context.add_command(label="Bulk Operations...", command=self._ed_bulk_operations)
        self._ed_context.add_separator()
        self._ed_context.add_command(label="Copy Lines  Ctrl+C", command=self._ed_copy_lines)
        self._ed_context.add_command(label="Paste Lines  Ctrl+V", command=self._ed_paste_lines)
        self._ed_context.add_command(label="Copy Text", command=self._ed_copy_text)
        self._ed_context.add_command(label="Paste Text", command=self._ed_paste_text)
        self._ed_context.add_command(label="Select All  Ctrl+A", command=self._ed_select_all)
        self._ed_context.add_separator()
        self._ed_context.add_command(label="Delete line(s)  Del", command=self._ed_delete_lines)
        self._ed_context.add_command(label="Duplicate  Ctrl+D", command=self._ed_duplicate_line)
        add_menu = tk.Menu(self._ed_context, tearoff=False)
        add_menu.add_command(label="Before", command=self._ed_add_line_before)
        add_menu.add_command(label="After", command=self._ed_add_line_after)
        self._ed_context.add_cascade(label="Add line", menu=add_menu)
        self._ed_context.add_separator()
        self._ed_context.add_command(label="Merge", command=self._ed_merge_lines)
        self._ed_context.add_command(label="Split", command=self._ed_split_line)
        self._ed_context.add_command(label="Shift Timers...", command=self._ed_shift_timings)
        self._ed_context.add_command(label="Swap Timing", command=self._ed_swap_timing)
        self._ed_context.add_separator()
        time_menu = tk.Menu(self._ed_context, tearoff=False)
        time_menu.add_command(label="Set Start \u2190 Video", command=self._ed_ass_set_start_from_video)
        time_menu.add_command(label="Set End \u2190 Video", command=self._ed_ass_set_end_from_video)
        self._ed_context.add_cascade(label="Video Time", menu=time_menu)
        self._ed_context.add_command(label="Toggle Comment", command=self._ed_toggle_comment)
        self._ed_context.add_command(label="Edit Style", command=self._ed_ass_context_style)
        self._ed_context.add_command(label="Edit Line...", command=self._ed_edit_line_dialog)
        self._ed_context.add_separator()
        sort_menu = tk.Menu(self._ed_context, tearoff=False)
        sort_menu.add_command(label="By Number", command=self._ed_sort_by_number)
        sort_menu.add_command(label="By Time", command=self._ed_sort_by_time)
        sort_menu.add_command(label="By Text", command=self._ed_sort_by_text)
        self._ed_context.add_cascade(label="Sort", menu=sort_menu)
        self._ed_context.add_separator()
        self._ed_context.add_command(label="Undo  Ctrl+Z", command=self._ed_undo)
        self._ed_context.add_command(label="Redo  Ctrl+Y", command=self._ed_redo)

        ed_bottom = ttk.Frame(editor_f)
        ed_bottom.pack(side=tk.BOTTOM, fill=tk.X)

        # ---- ASS Tag Toolbar (scrollable, compact) ----
        tag_container = ttk.Frame(ed_bottom)
        tag_container.pack(fill=tk.X, pady=(1, 0))
        tag_container.columnconfigure(0, weight=1)
        tag_canvas = tk.Canvas(tag_container, height=22, highlightthickness=0)
        tag_scroll = ttk.Scrollbar(tag_container, orient=tk.HORIZONTAL, command=tag_canvas.xview)
        tag_f = ttk.Frame(tag_canvas)
        tag_f.bind("<Configure>", lambda e: tag_canvas.configure(scrollregion=tag_canvas.bbox("all")))
        tag_canvas.create_window((0, 0), window=tag_f, anchor="nw")
        tag_canvas.configure(xscrollcommand=tag_scroll.set)
        tag_canvas.grid(row=0, column=0, sticky=tk.EW)
        tag_scroll.grid(row=1, column=0, sticky=tk.EW)
        self._ed_ass_tags_visible = True
        for lbl, tag in [
            ("B", "{\\b1}"), ("/B", "{\\b0}"),
            ("I", "{\\i1}"), ("/I", "{\\i0}"),
            ("U", "{\\u1}"), ("/U", "{\\u0}"),
            ("S", "{\\s1}"), ("/S", "{\\s0}"),
            ("fs20", "{\\fs20}"), ("fs30", "{\\fs30}"), ("fs40", "{\\fs40}"),
            ("fnArial", "{\\fnArial}"), ("fnSimHei", "{\\fnSimHei}"),
            ("c", "{\\c&HFFFFFF&}"), ("2c", "{\\2c&HFFFFFF&}"),
            ("bord", "{\\bord2}"), ("shad", "{\\shad2}"),
            ("pos", "{\\pos(100,200)}"), ("move", "{\\move(0,0,100,0)}"),
            ("fad", "{\\fad(200,200)}"),
        ]:
            btn = tk.Button(tag_f, text=lbl, padx=2, font=("TkDefaultFont", 7),
                            command=lambda t=tag: self._ed_ass_insert_tag(t))
            btn.pack(side=tk.LEFT, padx=1)
        ttk.Separator(tag_f, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)
        for color_key, fg in [("primary_color", "red"), ("secondary_color", "green"),
                              ("outline_color", "yellow"), ("shadow_color", "gray")]:
            tk.Button(tag_f, text="\u25A0", fg=fg, padx=2, font=("TkDefaultFont", 7),
                      command=lambda k=color_key: self._ed_ass_color_pick(k)).pack(side=tk.LEFT, padx=1)

        # ---- Edit Box (large, Aegisub-style) ----
        text_frame = ttk.LabelFrame(ed_bottom, text="Edit Box", padding=2)
        text_frame.pack(fill=tk.X, pady=(2, 0))
        self.ed_ass_text = tk.Text(text_frame, height=6, wrap=tk.WORD, font=("Consolas", 10))
        self.ed_ass_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ed_ass_text.bind("<Return>", lambda e: self._ed_ass_apply_text() or "break")
        self.ed_ass_text.bind("<KP_Enter>", lambda e: self._ed_ass_apply_text() or "break")
        self.ed_ass_text.bind("<Shift-Return>", lambda e: self.ed_ass_text.insert(tk.INSERT, "\n") or "break")
        self.ed_ass_text.bind("<Control-Return>", lambda e: self._ed_ass_apply_text() or "break")
        btn_f = ttk.Frame(text_frame)
        btn_f.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
        ttk.Button(btn_f, text="Apply", width=6, command=self._ed_ass_apply_text).pack(pady=(0, 2))
        ttk.Button(btn_f, text="Revert", width=6, command=self._ed_ass_revert_text).pack()

        # StringVars for line properties/timing (used by Edit Line dialog and Apply)
        self.ed_ass_start_var = tk.StringVar()
        self.ed_ass_end_var = tk.StringVar()
        self.ed_ass_dur_var = tk.StringVar()
        self.ed_ass_layer_var = tk.StringVar(value="0")
        self.ed_ass_style_var = tk.StringVar()
        self.ed_ass_actor_var = tk.StringVar()
        self.ed_ass_effect_var = tk.StringVar()
        self.ed_ass_ml_var = tk.StringVar(value="10")
        self.ed_ass_mr_var = tk.StringVar(value="10")
        self.ed_ass_mv_var = tk.StringVar(value="10")
        self.ed_ass_comment_var = tk.BooleanVar(value=False)
        self._ed_ass_style_names = []

        self.ed_status_var = tk.StringVar(value="Load an SRT file to edit")
        ttk.Label(parent, textvariable=self.ed_status_var, foreground="#888").grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=2)
        row += 1

        self._ed_data = []
        self._ed_path = None
        self._ed_undo_stack = []
        self._ed_redo_stack = []
        self._ed_undo_lock = False
        self._ed_clipboard = []
        self._ed_ass_mode = False
        self._ed_ass_styles = {}
        self._ed_ass_selected_idx = None

        self.root.bind("<Control-z>", lambda e: self._ed_undo())
        self.root.bind("<Control-Z>", lambda e: self._ed_undo())
        self.root.bind("<Control-y>", lambda e: self._ed_redo())
        self.root.bind("<Control-Y>", lambda e: self._ed_redo())
        self.root.bind("<Control-f>", lambda e: self._ed_filter_focus())
        self.root.bind("<Control-F>", lambda e: self._ed_filter_focus())
        self.root.bind("<Control-s>", lambda e: self._ed_save())
        self.root.bind("<Control-S>", lambda e: self._ed_save())
        self.root.bind("<Control-c>", lambda e: self._ed_copy_lines() or "break")
        self.root.bind("<Control-C>", lambda e: self._ed_copy_lines() or "break")
        self.root.bind("<Control-v>", lambda e: self._ed_paste_lines() or "break")
        self.root.bind("<Control-V>", lambda e: self._ed_paste_lines() or "break")
        self.root.bind("<Control-a>", lambda e: self._ed_select_all() or "break")
        self.root.bind("<Control-A>", lambda e: self._ed_select_all() or "break")
        self.root.bind("<Control-d>", lambda e: self._ed_duplicate_line() or "break")
        self.root.bind("<Control-D>", lambda e: self._ed_duplicate_line() or "break")
        self.root.bind("<Delete>", lambda e: self._ed_delete_lines() or "break")
        self.root.bind("<space>", lambda e: None)

        self._ed_video_player = None
        self._ed_video_top = None
        self._ed_video_sync_id = None

    def _ed_browse(self):
        path = filedialog.askopenfilename(title="Select subtitle file", filetypes=[("Subtitle files", "*.srt *.ass"), ("SRT files", "*.srt"), ("ASS files", "*.ass"), ("All files", "*.*")])
        if path:
            self.ed_srt_entry.delete(0, tk.END)
            self.ed_srt_entry.insert(0, path)
            self._ed_load()

    def _ed_load(self, path=None):
        if path is None:
            path = self.ed_srt_entry.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Select a valid subtitle file")
            return
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")
            return

        is_ass = path.lower().endswith(".ass")
        if is_ass:
            blocks, styles = AssParser.parse_ass(content)
            self._ed_data = blocks
            self._ed_ass_styles = styles
            self._ed_ass_refresh_style_cb()
            self._ed_ass_mode = True
            self.ed_status_var.set("ASS Mode: ON")
        else:
            blocks = SrtParser.parse_blocks(content)
            self._ed_data = blocks
        self._ed_path = path
        self._ed_originals = [b["text"] for b in blocks]
        self._ed_undo_stack.clear()
        self._ed_redo_stack.clear()
        self._ed_push_undo()

        self._ed_refresh_tree()
        self.ed_status_var.set(f"Loaded {len(blocks)} subtitles from {os.path.basename(path)}")

    # ---- Time helpers ----

    @staticmethod
    def _parse_srt_ms(t):
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", t.strip())
        if m:
            h, mi, s, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4])
            return h * 3600000 + mi * 60000 + s * 1000 + ms
        return 0

    @staticmethod
    def _format_srt_ms(ms):
        h = ms // 3600000
        r = ms % 3600000
        mi = r // 60000
        r = r % 60000
        s = r // 1000
        ml = r % 1000
        return f"{h:01d}:{mi:02d}:{s:02d},{ml:03d}"

    @staticmethod
    def _apply_timing_shift(timing, shift_ms):
        parts = timing.split("-->")
        if len(parts) != 2:
            return timing
        start = max(0, SubtitleExtractorGUI._parse_srt_ms(parts[0]) + shift_ms)
        end = max(0, SubtitleExtractorGUI._parse_srt_ms(parts[1]) + shift_ms)
        return f"{SubtitleExtractorGUI._format_srt_ms(start)} --> {SubtitleExtractorGUI._format_srt_ms(end)}"

    # ---- Undo / Redo ----

    def _ed_push_undo(self):
        if not self._ed_undo_lock:
            self._ed_undo_stack.append(copy.deepcopy(self._ed_data))
            self._ed_redo_stack.clear()
            if len(self._ed_undo_stack) > 50:
                self._ed_undo_stack.pop(0)
            self._ed_update_undo_buttons()

    def _ed_refresh_tree(self):
        for item in self.ed_tree.get_children():
            self.ed_tree.delete(item)
        for b in self._ed_data:
            self.ed_tree.insert("", tk.END, values=(b["index"], b["timing"], b["text"]))

    def _ed_update_undo_buttons(self):
        self.ed_undo_btn.configure(state=tk.NORMAL if self._ed_undo_stack else tk.DISABLED)
        self.ed_redo_btn.configure(state=tk.NORMAL if self._ed_redo_stack else tk.DISABLED)

    def _ed_undo(self):
        if not self._ed_undo_stack:
            return
        self._ed_redo_stack.append(copy.deepcopy(self._ed_data))
        self._ed_data = self._ed_undo_stack.pop()
        self._ed_undo_lock = True
        self._ed_refresh_tree()
        self._ed_undo_lock = False
        self._ed_update_undo_buttons()
        self._ed_video_update_subs()
        self.ed_status_var.set("Undo")

    def _ed_redo(self):
        if not self._ed_redo_stack:
            return
        self._ed_undo_stack.append(copy.deepcopy(self._ed_data))
        self._ed_data = self._ed_redo_stack.pop()
        self._ed_undo_lock = True
        self._ed_refresh_tree()
        self._ed_undo_lock = False
        self._ed_update_undo_buttons()
        self._ed_video_update_subs()
        self.ed_status_var.set("Redo")

    def _ed_renumber(self):
        for i, item in enumerate(self.ed_tree.get_children(), 1):
            vals = list(self.ed_tree.item(item, "values"))
            vals[0] = str(i)
            self.ed_tree.item(item, values=tuple(vals))
        for i, b in enumerate(self._ed_data, 1):
            b["index"] = str(i)

    def _ed_merge_lines(self):
        sel = self.ed_tree.selection()
        if len(sel) < 2:
            messagebox.showinfo("Merge", "Select at least 2 consecutive subtitles")
            return
        indices = [self.ed_tree.index(item) for item in sel]
        if indices != list(range(indices[0], indices[-1] + 1)):
            messagebox.showwarning("Merge", "Subtitles must be consecutive (no gaps)")
            return
        self._ed_push_undo()
        first_idx = indices[0]
        texts = [self._ed_data[i]["text"] for i in indices]
        merged_text = "\n".join(texts)
        start_t = self._ed_data[first_idx]["timing"].split("-->")[0].strip()
        end_t = self._ed_data[indices[-1]]["timing"].split("-->")[1].strip()
        merged_timing = f"{start_t} --> {end_t}"
        merged = {"index": str(first_idx + 1), "timing": merged_timing, "text": merged_text}
        for i in reversed(indices):
            self._ed_data.pop(i)
            self.ed_tree.delete(sel[indices.index(i)])
        self._ed_data.insert(first_idx, merged)
        self.ed_tree.insert("", first_idx, values=(merged["index"], merged["timing"], merged["text"]))
        self._ed_renumber()
        self.ed_status_var.set(f"Merged {len(indices)} lines into 1")
        self._ed_video_update_subs()

    def _ed_split_line(self):
        sel = self.ed_tree.selection()
        if len(sel) != 1:
            messagebox.showinfo("Split", "Select one subtitle to split")
            return
        self._ed_push_undo()
        idx = self.ed_tree.index(sel[0])
        block = self._ed_data[idx]
        text = block["text"]
        if not text:
            return
        timing = block["timing"]
        parts = timing.split("-->")
        if len(parts) != 2:
            return
        t_start = SubtitleExtractorGUI._parse_srt_ms(parts[0])
        t_end = SubtitleExtractorGUI._parse_srt_ms(parts[1])
        t_mid = (t_start + t_end) // 2

        # Find a good split point: sentence boundary or midpoint
        split_at = None
        for sep in ["\n", "。", "！", "？", ". ", "! ", "? ", ".", "! ", "?"]:
            pos = text.find(sep)
            if pos != -1 and pos < len(text) // 2 + 10:
                split_at = pos + len(sep)
                break
        if split_at is None:
            split_at = len(text) // 2

        text_a = text[:split_at].strip()
        text_b = text[split_at:].strip()
        if not text_a or not text_b:
            split_at = len(text) // 2
            text_a = text[:split_at].strip()
            text_b = text[split_at:].strip()

        timing_a = f"{parts[0].strip()} --> {SubtitleExtractorGUI._format_srt_ms(t_mid)}"
        timing_b = f"{SubtitleExtractorGUI._format_srt_ms(t_mid)} --> {parts[1].strip()}"

        block_a = {"index": str(idx + 1), "timing": timing_a, "text": text_a}
        block_b = {"index": str(idx + 2), "timing": timing_b, "text": text_b}

        self._ed_data[idx] = block_a
        self._ed_data.insert(idx + 1, block_b)
        self.ed_tree.item(sel[0], values=(block_a["index"], block_a["timing"], block_a["text"]))
        self.ed_tree.insert("", idx + 1, values=(block_b["index"], block_b["timing"], block_b["text"]))
        self._ed_renumber()
        self.ed_status_var.set("Split into 2 subtitles")
        self._ed_video_update_subs()

    def _ed_shift_timings(self):
        sel = self.ed_tree.selection()
        if not self._ed_data:
            return
        top = tk.Toplevel(self.root)
        top.title("Shift Timings")
        top.transient(self.root)
        top.grab_set()
        top.resizable(False, False)

        ttk.Label(top, text="Shift (seconds, can be negative):").pack(padx=10, pady=(10, 2))
        var = tk.StringVar(value="0")
        entry = ttk.Entry(top, textvariable=var, width=10)
        entry.pack(padx=10, pady=2)
        entry.focus_set()
        entry.select_range(0, tk.END)

        has_sel = bool(sel)
        scope = tk.StringVar(value="sel" if has_sel else "all")
        ttk.Radiobutton(top, text="All subtitles", variable=scope, value="all").pack(anchor=tk.W, padx=10)
        ttk.Radiobutton(top, text=f"Selected only ({len(sel)} lines)" if has_sel else "Selected only", variable=scope, value="sel").pack(anchor=tk.W, padx=10)

        def do_shift():
            try:
                sec = float(var.get().strip())
            except ValueError:
                return
            shift_ms = int(sec * 1000)
            self._ed_push_undo()
            items = list(sel) if scope.get() == "sel" else self.ed_tree.get_children()
            for item in items:
                idx = self.ed_tree.index(item)
                block = self._ed_data[idx]
                block["timing"] = self._apply_timing_shift(block["timing"], shift_ms)
                vals = list(self.ed_tree.item(item, "values"))
                vals[1] = block["timing"]
                self.ed_tree.item(item, values=tuple(vals))
            self.ed_status_var.set(f"Shifted by {sec:+.1f}s")
            self._ed_video_update_subs()
            top.destroy()

        entry.bind("<Return>", lambda e: do_shift())
        bf = ttk.Frame(top)
        bf.pack(padx=10, pady=(5, 10))
        ttk.Button(bf, text="Apply", command=do_shift).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Cancel", command=top.destroy).pack(side=tk.LEFT, padx=2)

    def _ed_edit_cell(self, event):
        item = self.ed_tree.selection()
        if not item:
            return
        col = self.ed_tree.identify_column(event.x)
        col_idx = int(col.replace("#", "")) - 1
        if col_idx not in (1, 2):
            return

        x, y, width, height = self.ed_tree.bbox(item[0], column=col)
        if not width or not height:
            return
        value = self.ed_tree.item(item[0], "values")[col_idx]

        entry = ttk.Entry(self.ed_tree, width=width)
        entry.place(x=x, y=y, width=width, height=height)
        entry.insert(0, value)
        entry.focus_set()
        entry.select_range(0, tk.END)

        def on_confirm(event=None):
            new_val = entry.get()
            entry.destroy()
            if new_val == value:
                return
            self._ed_push_undo()
            vals = list(self.ed_tree.item(item[0], "values"))
            vals[col_idx] = new_val
            self.ed_tree.item(item[0], values=tuple(vals))
            idx = self.ed_tree.index(item[0])
            if idx < len(self._ed_data):
                key = "timing" if col_idx == 1 else "text"
                self._ed_data[idx][key] = new_val
            self._ed_video_update_subs()
            self.ed_status_var.set("Edited — don't forget to Save")

        def on_cancel(event=None):
            entry.destroy()

        entry.bind("<Return>", on_confirm)
        entry.bind("<Escape>", on_cancel)
        entry.bind("<FocusOut>", on_confirm)

    def _ed_save(self, path=None):
        if not self._ed_data:
            messagebox.showinfo("Save", "No data to save")
            return
        if path is None:
            path = self._ed_path
        if path is None:
            ftypes = [("ASS subtitles", "*.ass"), ("SRT subtitles", "*.srt"), ("All files", "*.*")]
            path = filedialog.asksaveasfilename(
                title="Save subtitles",
                defaultextension=".ass" if self._ed_ass_mode else ".srt",
                filetypes=ftypes,
            )
            if not path:
                return
        is_ass = path.lower().endswith(".ass")
        if is_ass:
            styles = self._ed_ass_styles if self._ed_ass_styles else {"Default": AssParser.default_style()}
            content = AssParser.blocks_to_ass(self._ed_data, styles)
        else:
            content = SrtParser.blocks_to_srt(self._ed_data)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        self._ed_path = path
        self.ed_status_var.set(f"Saved to {os.path.basename(path)}")

    def _ed_send_to_translation(self):
        if not self._ed_path or not os.path.isfile(self._ed_path):
            if self._ed_data:
                tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_editor_temp.srt")
                self._ed_save(tmp)
                if not os.path.isfile(tmp):
                    return
            else:
                messagebox.showinfo("Send", "Load or create subtitles first")
                return
        self.srt_entry.delete(0, tk.END)
        self.srt_entry.insert(0, self._ed_path)
        p = Path(self._ed_path)
        out = str(p.parent / f"{p.stem}_translated.srt")
        self._trans_out_path = out
        self.ed_status_var.set(f"Sent to Translation tab: {os.path.basename(self._ed_path)}")

    def _ed_delete_lines(self):
        sel = self.ed_tree.selection()
        if not sel:
            return
        if not messagebox.askyesno("Delete", f"Delete {len(sel)} subtitle(s)?"):
            return
        self._ed_push_undo()
        indices = sorted([self.ed_tree.index(item) for item in sel], reverse=True)
        for i in indices:
            self._ed_data.pop(i)
        self._ed_refresh_tree()
        self._ed_renumber()
        self.ed_status_var.set(f"Deleted {len(sel)} line(s)")
        self._ed_video_update_subs()

    def _ed_duplicate_line(self):
        sel = self.ed_tree.selection()
        if not sel:
            return
        self._ed_push_undo()
        idx = self.ed_tree.index(sel[0])
        block = self._ed_data[idx]
        dup = {"index": str(idx + 2), "timing": block["timing"], "text": block["text"]}
        self._ed_data.insert(idx + 1, dup)
        self._ed_refresh_tree()
        self._ed_renumber()
        self.ed_status_var.set("Duplicated")
        self._ed_video_update_subs()

    def _ed_add_line_before(self):
        sel = self.ed_tree.selection()
        if not sel:
            return
        self._ed_push_undo()
        idx = self.ed_tree.index(sel[0])
        block = self._ed_data[idx]
        parts = block["timing"].split("-->")
        if len(parts) == 2:
            t_start = SubtitleExtractorGUI._parse_srt_ms(parts[0])
            new_end = SubtitleExtractorGUI._format_srt_ms(t_start)
            new_start = SubtitleExtractorGUI._format_srt_ms(max(0, t_start - 2000))
            new_timing = f"{new_start} --> {new_end}"
        else:
            new_timing = block["timing"]
        entry = {"index": str(idx + 1), "timing": new_timing, "text": ""}
        self._ed_data.insert(idx, entry)
        self._ed_refresh_tree()
        self._ed_renumber()
        self.ed_status_var.set("Added line before")
        self._ed_video_update_subs()

    def _ed_add_line_after(self):
        sel = self.ed_tree.selection()
        if not sel:
            return
        self._ed_push_undo()
        idx = self.ed_tree.index(sel[0])
        block = self._ed_data[idx]
        parts = block["timing"].split("-->")
        if len(parts) == 2:
            t_end = SubtitleExtractorGUI._parse_srt_ms(parts[1])
            new_start = SubtitleExtractorGUI._format_srt_ms(t_end)
            new_timing = f"{new_start} --> {SubtitleExtractorGUI._format_srt_ms(t_end + 2000)}"
        else:
            new_timing = "00:00:00,000 --> 00:00:02,000"
        entry = {"index": str(idx + 2), "timing": new_timing, "text": ""}
        self._ed_data.insert(idx + 1, entry)
        self._ed_refresh_tree()
        self._ed_renumber()
        self.ed_status_var.set("Added line after")
        self._ed_video_update_subs()

    # ---- Filter ----

    def _ed_filter_focus(self):
        self.ed_filter_entry.focus_set()
        self.ed_filter_entry.select_range(0, tk.END)

    def _ed_filter(self):
        q = self.ed_filter_var.get().strip().lower()
        self._ed_refresh_tree()
        if not q:
            return
        hidden = 0
        for item in self.ed_tree.get_children():
            vals = self.ed_tree.item(item, "values")
            text = (vals[2] if len(vals) > 2 else "").lower()
            timing = (vals[1] if len(vals) > 1 else "").lower()
            if q not in text and q not in timing:
                self.ed_tree.detach(item)
                hidden += 1
        total = len(self._ed_data)
        shown = total - hidden
        self.ed_status_var.set(f"Filter '{q}': {shown} shown, {hidden} hidden")

    def _ed_filter_clear(self):
        self.ed_filter_var.set("")
        self._ed_refresh_tree()
        self.ed_status_var.set(f"Showing all {len(self._ed_data)} subtitles")

    # ---- Format conversion ----

    @staticmethod
    def _srt_to_ass(blocks):
        lines = ["[Script Info]", "ScriptType: v4.00+", "WrapStyle: 0",
                  "", "[V4+ Styles]",
                  "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                  "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                  "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
        for b in blocks:
            t = b["timing"]
            parts = t.split("-->")
            if len(parts) == 2:
                start = parts[0].strip().replace(",", ".")
                end = parts[1].strip().replace(",", ".")
            else:
                start, end = "0:00:00.00", "0:00:00.00"
            text = b["text"].replace("\n", "\\N")
            lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
        return "\n".join(lines)

    @staticmethod
    def _ass_to_srt(ass_text):
        blocks = []
        idx = 0
        for line in ass_text.split("\n"):
            if line.startswith("Dialogue:"):
                idx += 1
                parts = line.split(",", 9)
                if len(parts) >= 10:
                    start = parts[1].strip().replace(".", ",")
                    end = parts[2].strip().replace(".", ",")
                    text = parts[9].replace("\\N", "\n").replace("\\n", "\n")
                    timing = f"{start} --> {end}"
                    blocks.append({"index": str(idx), "timing": timing, "text": text})
        return blocks

    def _ed_convert(self):
        if not self._ed_data:
            messagebox.showinfo("Convert", "No subtitles loaded")
            return
        path = filedialog.asksaveasfilename(
            title="Convert to...",
            defaultextension=".ass",
            filetypes=[("ASS subtitles", "*.ass"), ("SRT subtitles", "*.srt"), ("All files", "*.*")],
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext == ".ass":
            content = self._srt_to_ass(self._ed_data)
        elif ext == ".srt":
            content = SrtParser.blocks_to_srt(self._ed_data)
        else:
            content = self._srt_to_ass(self._ed_data)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        self.ed_status_var.set(f"Converted and saved to {os.path.basename(path)}")

    # ---- Compare ----

    def _ed_compare(self):
        path_other = filedialog.askopenfilename(title="Select second SRT to compare", filetypes=[("SRT files", "*.srt"), ("All files", "*.*")])
        if not path_other:
            return
        try:
            with open(path_other, "r", encoding="utf-8-sig") as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")
            return
        other_blocks = SrtParser.parse_blocks(content)

        top = tk.Toplevel(self.root)
        top.title(f"Compare: {os.path.basename(self._ed_path or 'current')} vs {os.path.basename(path_other)}")
        top.transient(self.root)
        top.geometry("1000x600")

        paned = ttk.PanedWindow(top, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left = ttk.Frame(paned)
        paned.add(left, weight=1)
        ttk.Label(left, text="Current", font=("", 9, "bold")).pack(anchor=tk.W)
        l_cols = ("idx", "timing", "text")
        l_tree = ttk.Treeview(left, columns=l_cols, show="headings", height=16)
        for c in l_cols:
            l_tree.heading(c, text=c.title())
        l_tree.column("idx", width=30, anchor=tk.CENTER)
        l_tree.column("timing", width=160)
        l_tree.column("text", width=250)
        l_tree.pack(fill=tk.BOTH, expand=True)
        for b in self._ed_data:
            l_tree.insert("", tk.END, values=(b["index"], b["timing"], b["text"]))

        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        ttk.Label(right, text="Compared file", font=("", 9, "bold")).pack(anchor=tk.W)
        r_tree = ttk.Treeview(right, columns=l_cols, show="headings", height=16)
        for c in l_cols:
            r_tree.heading(c, text=c.title())
        r_tree.column("idx", width=30, anchor=tk.CENTER)
        r_tree.column("timing", width=160)
        r_tree.column("text", width=250)
        r_tree.pack(fill=tk.BOTH, expand=True)
        for b in other_blocks:
            r_tree.insert("", tk.END, values=(b["index"], b["timing"], b["text"]))

        ttk.Label(top, text="Differences are highlighted in yellow", foreground="#888").pack(anchor=tk.W, padx=5)

        def highlight_diffs():
            l_items = l_tree.get_children()
            r_items = r_tree.get_children()
            max_len = max(len(l_items), len(r_items))
            style = ttk.Style()
            style.configure("Diff.Treeview", rowheight=20, fieldbackground="#ffffcc")
            for i in range(max_len):
                lv = l_tree.item(l_items[i], "values") if i < len(l_items) else ("", "", "")
                rv = r_tree.item(r_items[i], "values") if i < len(r_items) else ("", "", "")
                if lv[2] != rv[2]:
                    if i < len(l_items):
                        l_tree.item(l_items[i], tags=("diff",))
                    if i < len(r_items):
                        r_tree.item(r_items[i], tags=("diff",))
            l_tree.tag_configure("diff", background="#ffffcc")
            r_tree.tag_configure("diff", background="#ffffcc")

        ttk.Button(top, text="Highlight differences", command=highlight_diffs).pack(pady=5)

    def _ed_show_original(self):
        sel = self.ed_tree.selection()
        if not sel:
            return
        if not hasattr(self, '_ed_originals'):
            messagebox.showinfo("Original", "No original data available (reload the file)")
            return
        idx = self.ed_tree.index(sel[0])
        if idx >= len(self._ed_originals):
            return
        current = self._ed_data[idx]["text"]
        original = self._ed_originals[idx]
        top = tk.Toplevel(self.root)
        top.title("Translate Mode")
        top.transient(self.root)
        top.grab_set()
        top.geometry("550x350")
        ttk.Label(top, text="Original:", font=("", 9, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 0))
        orig_w = tk.Text(top, height=3, wrap=tk.WORD, fg="#888")
        orig_w.insert("1.0", original)
        orig_w.configure(state=tk.DISABLED)
        orig_w.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(top, text="Current (editable):", font=("", 9, "bold")).pack(anchor=tk.W, padx=10, pady=(5, 0))
        cur_w = tk.Text(top, height=3, wrap=tk.WORD)
        cur_w.insert("1.0", current)
        cur_w.pack(fill=tk.X, padx=10, pady=2)

        bf = ttk.Frame(top)
        bf.pack(fill=tk.X, padx=10, pady=5)

        def save_text():
            new_text = cur_w.get("1.0", tk.END).strip()
            if new_text and new_text != current:
                self._ed_push_undo()
                self._ed_data[idx]["text"] = new_text
                self._ed_refresh_tree()
                self.ed_status_var.set("Subtitle text updated")

        def lookup_dict():
            text = cur_w.get("1.0", tk.END).strip()
            if text:
                self.dict_entry.delete(0, tk.END)
                self.dict_entry.insert(0, text)
                self._dict_lookup()
                self.notebook.select(2)
                top.destroy()

        ttk.Button(bf, text="Update text", command=save_text).pack(side=tk.LEFT, padx=2)
        ttk.Separator(bf, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Button(bf, text="Look up in Dictionary", command=lookup_dict).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Close", command=top.destroy).pack(side=tk.RIGHT, padx=2)

    # ---- Bulk Operations ----

    def _ed_bulk_operations(self):
        if not self._ed_data:
            messagebox.showinfo("Bulk Ops", "Load subtitles first")
            return
        sel = self.ed_tree.selection()
        if not sel:
            messagebox.showinfo("Bulk Ops", "Select one or more subtitle lines first")
            return

        indices = [self.ed_tree.index(item) for item in sel]
        top = tk.Toplevel(self.root)
        top.title("Bulk Operations")
        top.transient(self.root)
        top.grab_set()
        top.geometry("500x420")

        nb = ttk.Notebook(top)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Tab 1: Apply Tags ---
        tag_f = ttk.Frame(nb)
        nb.add(tag_f, text="Apply Tags")
        ttk.Label(tag_f, text="Wrap selected lines with:").pack(anchor=tk.W, padx=10, pady=(10, 2))
        tag_entry = ttk.Entry(tag_f, width=50)
        tag_entry.insert(0, "{\\b1}")
        tag_entry.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(tag_f, text="Closing tag (leave empty for self-closing):").pack(anchor=tk.W, padx=10, pady=(5, 2))
        tag_close = ttk.Entry(tag_f, width=50)
        tag_close.insert(0, "{\\b0}")
        tag_close.pack(fill=tk.X, padx=10, pady=2)
        ttk.Button(tag_f, text="Apply Tags",
                   command=lambda: self._ed_bulk_apply_tags(indices, tag_entry, tag_close, top)).pack(pady=10)

        # --- Tab 2: Strip Tags ---
        strip_f = ttk.Frame(nb)
        nb.add(strip_f, text="Strip Tags")
        ttk.Label(strip_f, text="Remove all ASS tags ({\\ ... }) from selected lines?").pack(pady=(20, 5))
        ttk.Button(strip_f, text="Strip All Tags",
                   command=lambda: self._ed_bulk_strip_tags(indices, top)).pack(pady=10)
        ttk.Label(strip_f, text="Also remove \\N / \\n line breaks").pack()

        # --- Tab 3: Find & Replace ---
        fr_f = ttk.Frame(nb)
        nb.add(fr_f, text="Find & Replace")
        ttk.Label(fr_f, text="Find:").pack(anchor=tk.W, padx=10, pady=(10, 2))
        find_var = tk.StringVar(value="")
        find_e = ttk.Entry(fr_f, textvariable=find_var, width=50)
        find_e.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(fr_f, text="Replace with:").pack(anchor=tk.W, padx=10, pady=(5, 2))
        repl_var = tk.StringVar(value="")
        repl_e = ttk.Entry(fr_f, textvariable=repl_var, width=50)
        repl_e.pack(fill=tk.X, padx=10, pady=2)
        case_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fr_f, text="Case sensitive", variable=case_var).pack(anchor=tk.W, padx=10, pady=(5, 0))
        ttk.Button(fr_f, text="Replace All",
                   command=lambda: self._ed_bulk_replace(indices, find_var, repl_var, case_var, top)).pack(pady=10)

        # --- Tab 4: Shift Timing ---
        shift_f = ttk.Frame(nb)
        nb.add(shift_f, text="Shift Timing")
        ttk.Label(shift_f, text="Shift selected lines by (ms):").pack(anchor=tk.W, padx=10, pady=(10, 2))
        shift_var = tk.StringVar(value="100")
        ttk.Spinbox(shift_f, from_=-10000, to=10000, textvariable=shift_var, width=10).pack(anchor=tk.W, padx=10)
        ttk.Label(shift_f, text="(+ forward, - backward)", foreground="#888").pack(anchor=tk.W, padx=10)
        ttk.Button(shift_f, text="Shift Timing",
                   command=lambda: self._ed_bulk_shift(indices, shift_var, top)).pack(pady=10)

    def _ed_bulk_apply_tags(self, indices, tag_entry, tag_close, top):
        tag_open = tag_entry.get().strip()
        tag_close_str = tag_close.get().strip()
        if not tag_open:
            return
        self._ed_push_undo()
        for i in indices:
            text = self._ed_data[i]["text"]
            if tag_close_str:
                self._ed_data[i]["text"] = f"{tag_open}{text}{tag_close_str}"
            else:
                self._ed_data[i]["text"] = f"{tag_open}{text}"
        self._ed_refresh_tree()
        self._ed_video_update_subs()
        self.ed_status_var.set(f"Tags applied to {len(indices)} line(s)")
        top.destroy()

    def _ed_bulk_strip_tags(self, indices, top):
        self._ed_push_undo()
        for i in indices:
            text = self._ed_data[i]["text"]
            stripped = re.sub(r"\{[^}]*\}", "", text)
            stripped = stripped.replace("\\N", "\n").replace("\\n", "\n")
            stripped = stripped.replace("\\h", " ")
            self._ed_data[i]["text"] = stripped
        self._ed_refresh_tree()
        self._ed_video_update_subs()
        self.ed_status_var.set(f"Tags stripped from {len(indices)} line(s)")
        top.destroy()

    def _ed_bulk_replace(self, indices, find_var, repl_var, case_var, top):
        find = find_var.get()
        repl = repl_var.get()
        if not find:
            return
        flags = 0 if case_var.get() else re.IGNORECASE
        self._ed_push_undo()
        count = 0
        for i in indices:
            text = self._ed_data[i]["text"]
            new_text, n = re.subn(re.escape(find), repl, text, flags=flags)
            if n > 0:
                self._ed_data[i]["text"] = new_text
                count += n
        self._ed_refresh_tree()
        self._ed_video_update_subs()
        self.ed_status_var.set(f"Replaced {count} occurrence(s) in {len(indices)} line(s)")
        top.destroy()

    def _ed_bulk_shift(self, indices, shift_var, top):
        try:
            shift_ms = int(shift_var.get())
        except ValueError:
            return
        if shift_ms == 0:
            return
        self._ed_push_undo()
        for i in indices:
            timing = self._ed_data[i]["timing"]
            self._ed_data[i]["timing"] = self._apply_timing_shift(timing, shift_ms)
        self._ed_refresh_tree()
        self._ed_video_update_subs()
        self.ed_status_var.set(f"Timing shifted by {shift_ms}ms for {len(indices)} line(s)")
        top.destroy()

    # ---- Video Player ----

    def _ed_video_srt_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vlc_subs.srt")

    def _ed_video_ass_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vlc_subs.ass")

    def _ed_video_write_srt(self):
        if self._ed_ass_mode and self._ed_ass_styles:
            path = self._ed_video_ass_path()
            content = AssParser.blocks_to_ass(self._ed_data, self._ed_ass_styles)
        else:
            path = self._ed_video_srt_path()
            content = SrtParser.blocks_to_srt(self._ed_data)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _ed_open_video(self):
        path = filedialog.askopenfilename(title="Select video file", filetypes=[("Video files", "*.mp4;*.mkv;*.avi;*.mov;*.wmv;*.flv;*.ts"), ("All files", "*.*")])
        if not path:
            return
        if self._ed_video_player is not None:
            self._ed_video_close()
        if getattr(self, '_ed_video_sync_id', None):
            self.root.after_cancel(self._ed_video_sync_id)
            self._ed_video_sync_id = None

        if not self._ed_data:
            messagebox.showinfo("Video", "Load subtitles first")
            return

        self._ed_video_write_srt()
        sub_path = self._ed_video_ass_path() if (self._ed_ass_mode and self._ed_ass_styles) else self._ed_video_srt_path()

        try:
            import vlc
        except ImportError:
            messagebox.showerror("Error", "python-vlc not installed")
            return

        vf = self._ed_static_video_f
        for w in vf.winfo_children():
            w.destroy()

        canvas = tk.Canvas(vf, bg="black", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(vf)
        controls.pack(fill=tk.X, padx=2, pady=1)
        self._ed_video_play_btn = ttk.Button(controls, text="Pause", command=self._ed_video_play_pause)
        self._ed_video_play_btn.pack(side=tk.LEFT, padx=1)
        ttk.Button(controls, text="Stop", command=self._ed_video_stop).pack(side=tk.LEFT, padx=1)
        self._ed_video_pos_var = tk.StringVar(value="0:00 / 0:00")
        ttk.Label(controls, textvariable=self._ed_video_pos_var).pack(side=tk.LEFT, padx=8)
        self._ed_video_scale = ttk.Scale(controls, from_=0, to=1000, orient=tk.HORIZONTAL, command=self._ed_video_seek)
        self._ed_video_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(controls, text="Vol:").pack(side=tk.LEFT, padx=(5, 0))
        self._ed_video_vol = ttk.Scale(controls, from_=0, to=100, orient=tk.HORIZONTAL, value=50)
        self._ed_video_vol.pack(side=tk.LEFT, padx=2)
        self._ed_video_vol.configure(command=lambda v: self._ed_video_player.audio_set_volume(int(float(v))) if self._ed_video_player else None)
        ttk.Separator(controls, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)
        ttk.Button(controls, text="Popup", command=self._ed_open_video_popup_from_static).pack(side=tk.LEFT, padx=1)

        self._ed_static_pw.insert(0, vf, weight=1)

        self.root.geometry("1500x750")
        self._ed_video_player = None
        self._ed_video_top = None
        self._ed_video_canvas = canvas
        self._ed_video_path = path
        self._ed_video_seeking = False
        self._ed_video_is_popup = False

        vf.update_idletasks()

        try:
            instance = vlc.Instance("--quiet --no-video-title-show")
            player = instance.media_player_new()
            media = instance.media_new(path)
            media.add_option(f":sub-file={sub_path}")
            player.set_media(media)
            player.set_hwnd(canvas.winfo_id())
            player.audio_set_volume(50)

            self._ed_video_player = player
            self._ed_video_instance = instance

            player.play()
            player.video_set_spu(1)
            self._ed_video_sync()
        except Exception as e:
            messagebox.showerror("VLC Error", f"Failed to initialize VLC:\n{e}")
            self._ed_video_close()

    def _ed_open_video_popup_from_static(self):
        path = getattr(self, '_ed_video_path', None)
        if path:
            self._ed_video_close()
            self._open_video_popup_internal(path)

    def _ed_open_video_popup(self):
        path = filedialog.askopenfilename(title="Select video file (popup)", filetypes=[("Video files", "*.mp4;*.mkv;*.avi;*.mov;*.wmv;*.flv;*.ts"), ("All files", "*.*")])
        if not path:
            return
        self._open_video_popup_internal(path)

    def _open_video_popup_internal(self, path):
        if self._ed_video_player is not None:
            self._ed_video_close()
        if getattr(self, '_ed_video_sync_id', None):
            self.root.after_cancel(self._ed_video_sync_id)
            self._ed_video_sync_id = None

        if not self._ed_data:
            messagebox.showinfo("Video", "Load subtitles first")
            return

        self._ed_video_write_srt()
        sub_path = self._ed_video_ass_path() if (self._ed_ass_mode and self._ed_ass_styles) else self._ed_video_srt_path()

        try:
            import vlc
        except ImportError:
            messagebox.showerror("Error", "python-vlc not installed")
            return

        top = tk.Toplevel(self.root)
        top.title(f"Video — {os.path.basename(path)}")
        top.geometry("900x700")
        top.protocol("WM_DELETE_WINDOW", self._ed_video_close)

        video_frame = ttk.Frame(top)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        canvas = tk.Canvas(video_frame, bg="black", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(top)
        controls.pack(fill=tk.X, padx=5, pady=2)
        self._ed_video_play_btn = ttk.Button(controls, text="Pause", command=self._ed_video_play_pause)
        self._ed_video_play_btn.pack(side=tk.LEFT, padx=1)
        ttk.Button(controls, text="Stop", command=self._ed_video_stop).pack(side=tk.LEFT, padx=1)
        ttk.Separator(controls, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)
        self._ed_video_pos_var = tk.StringVar(value="0:00 / 0:00")
        ttk.Label(controls, textvariable=self._ed_video_pos_var).pack(side=tk.LEFT, padx=8)
        self._ed_video_scale = ttk.Scale(controls, from_=0, to=1000, orient=tk.HORIZONTAL, command=self._ed_video_seek)
        self._ed_video_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(controls, text="Vol:").pack(side=tk.LEFT, padx=(5, 0))
        self._ed_video_vol = ttk.Scale(controls, from_=0, to=100, orient=tk.HORIZONTAL, value=50)
        self._ed_video_vol.pack(side=tk.LEFT, padx=2)
        self._ed_video_vol.configure(command=lambda v: self._ed_video_player.audio_set_volume(int(float(v))) if self._ed_video_player else None)

        self._ed_video_player = None
        self._ed_video_top = top
        self._ed_video_canvas = canvas
        self._ed_video_path = path
        self._ed_video_seeking = False
        self._ed_video_is_popup = True

        top.update_idletasks()

        try:
            instance = vlc.Instance("--quiet --no-video-title-show")
            player = instance.media_player_new()
            media = instance.media_new(path)
            media.add_option(f":sub-file={sub_path}")
            player.set_media(media)
            player.set_hwnd(canvas.winfo_id())
            player.audio_set_volume(50)

            self._ed_video_player = player
            self._ed_video_instance = instance

            player.play()
            player.video_set_spu(1)
            self._ed_video_sync()
        except Exception as e:
            messagebox.showerror("VLC Error", f"Failed to initialize VLC:\n{e}")
            top.destroy()

    def _ed_video_play_pause(self):
        if not self._ed_video_player:
            return
        if self._ed_video_player.is_playing():
            self._ed_video_player.pause()
            self._ed_video_play_btn.configure(text="Play")
        else:
            self._ed_video_player.play()
            self._ed_video_play_btn.configure(text="Pause")

    def _ed_video_stop(self):
        if not self._ed_video_player:
            return
        self._ed_video_player.stop()
        self._ed_video_play_btn.configure(text="Play")
        self._ed_video_pos_var.set("0:00 / 0:00")
        self._ed_video_scale.set(0)

    def _ed_video_seek(self, val):
        if not self._ed_video_player or self._ed_video_seeking:
            return
        length = self._ed_video_player.get_length()
        if length > 0:
            self._ed_video_player.set_position(float(val) / 1000.0)

    def _ed_video_update_subs(self):
        if not self._ed_video_player:
            return
        self._ed_video_write_srt()
        sub_path = self._ed_video_ass_path() if (self._ed_ass_mode and self._ed_ass_styles) else self._ed_video_srt_path()
        try:
            from vlc import MediaSlaveType
            import pathlib
            sub_uri = pathlib.Path(sub_path).as_uri()
            self._ed_video_player.add_slave(MediaSlaveType.subtitle, sub_uri, True)
            self._ed_video_player.video_set_spu(1)
        except Exception:
            saved_time = self._ed_video_player.get_time()
            self._ed_video_player.stop()
            import vlc
            media = self._ed_video_instance.media_new(self._ed_video_path)
            media.add_option(f":sub-file={sub_path}")
            self._ed_video_player.set_media(media)
            self._ed_video_player.play()
            self._ed_video_player.video_set_spu(1)
            if saved_time > 0:
                self._ed_video_player.set_time(int(saved_time))

    def _ed_video_sync(self):
        if not self._ed_video_player:
            return
        try:
            if self._ed_video_player.is_playing():
                length = self._ed_video_player.get_length()
                t = self._ed_video_player.get_time()
                if length > 0 and t > 0:
                    pos = t / length
                    self._ed_video_seeking = True
                    self._ed_video_scale.set(pos * 1000)
                    self._ed_video_seeking = False
                    self._ed_video_pos_var.set(f"{t // 60000}:{(t // 1000) % 60:02d} / "
                                                f"{length // 60000}:{(length // 1000) % 60:02d}")
                    children = self.ed_tree.get_children()
                    for i, b in enumerate(self._ed_data):
                        parts = b["timing"].split("-->")
                        if len(parts) == 2:
                            t_start = self._parse_srt_ms(parts[0])
                            t_end = self._parse_srt_ms(parts[1])
                            if t_start <= t < t_end and i < len(children):
                                self.ed_tree.selection_set(children[i])
                                self.ed_tree.see(children[i])
                                break
        except Exception:
            pass
        try:
            self._ed_video_sync_id = self.root.after(200, self._ed_video_sync)
        except Exception:
            self._ed_video_sync_id = None

    def _ed_video_close(self):
        if getattr(self, '_ed_video_sync_id', None):
            self.root.after_cancel(self._ed_video_sync_id)
        if getattr(self, '_ed_video_player', None):
            try:
                self._ed_video_player.stop()
                self._ed_video_player.release()
            except Exception:
                pass
            self._ed_video_player = None
        if hasattr(self, '_ed_video_instance') and self._ed_video_instance:
            try:
                self._ed_video_instance.release()
            except Exception:
                pass
        if hasattr(self, '_ed_video_top') and self._ed_video_top:
            try:
                self._ed_video_top.destroy()
            except Exception:
                pass
            self._ed_video_top = None
        if hasattr(self, '_ed_static_pw') and hasattr(self, '_ed_static_video_f'):
            try:
                self._ed_static_pw.forget(self._ed_static_video_f)
            except Exception:
                pass
            if not getattr(self, '_ed_video_is_popup', True):
                self.root.geometry("1100x700")

    # ---- ASS / Advanced Editor ----

    def _ed_on_select(self, event):
        sel = self.ed_tree.selection()
        if sel:
            vals = self.ed_tree.item(sel[0], "values")
            self._ed_ass_selected_idx = self.ed_tree.index(sel[0])
            raw = self._ed_data[self._ed_ass_selected_idx]["text"]
            self._ed_ass_revert_text(raw)
            b = self._ed_data[self._ed_ass_selected_idx]
            parts = b["timing"].split("-->")
            if len(parts) == 2:
                self.ed_ass_start_var.set(parts[0].strip())
                self.ed_ass_end_var.set(parts[1].strip())
                try:
                    dur = self._parse_srt_ms(parts[1]) - self._parse_srt_ms(parts[0])
                    self.ed_ass_dur_var.set(f"{dur}ms" if dur >= 0 else "")
                except Exception:
                    self.ed_ass_dur_var.set("")
            self.ed_ass_layer_var.set(b.get("layer", "0"))
            if self._ed_ass_mode and self._ed_ass_styles:
                style = b.get("style", "Default")
                if style in self._ed_ass_styles:
                    self.ed_ass_style_var.set(style)
            self.ed_ass_actor_var.set(b.get("actor", ""))
            self.ed_ass_effect_var.set(b.get("effect", ""))
            self.ed_ass_ml_var.set(b.get("margin_l", "0"))
            self.ed_ass_mr_var.set(b.get("margin_r", "0"))
            self.ed_ass_mv_var.set(b.get("margin_v", "0"))
            self.ed_ass_comment_var.set(b.get("comment", False))
            self.ed_status_var.set(f"Line {b['index']}: {b['timing']}")
        else:
            self._ed_ass_selected_idx = None
            self.ed_ass_text.delete("1.0", tk.END)

    def _parse_tag_string(self, tag_str):
        m = re.match(r'\{(\\(?:[a-z]+|\d*c))([^}]*)\}', tag_str)
        if m:
            return m.group(1), m.group(2)
        return None, None

    def _get_current_style(self):
        style = {}
        if self._ed_ass_selected_idx is not None and self._ed_ass_selected_idx < len(self._ed_data):
            sname = self._ed_data[self._ed_ass_selected_idx].get("style", "Default")
            style = self._ed_ass_styles.get(sname, {})
        return style

    def _raw_cursor_pos(self):
        return len(self.ed_ass_text.get("1.0", tk.INSERT))

    def _ed_ass_toggle_tag(self, tag_name):
        text_w = self.ed_ass_text
        full_text = text_w.get("1.0", tk.END).rstrip("\n")
        cursor_pos = self._raw_cursor_pos()
        style = self._get_current_style()
        new_text = AssParser.toggle_binary_tag(full_text, tag_name, style, cursor_pos)
        text_w.delete("1.0", tk.END)
        text_w.insert("1.0", new_text)
        text_w.focus_set()

    def _ed_ass_wrap_selection(self, tag):
        text_w = self.ed_ass_text
        start = text_w.index(tk.SEL_FIRST)
        end = text_w.index(tk.SEL_LAST)
        selected = text_w.get(start, end)
        closing = AssParser.get_closing_tag(tag)
        text_w.delete(start, end)
        text_w.insert(start, tag + selected + closing)
        text_w.focus_set()

    def _ed_ass_toggle_with_selection(self, tag_name):
        text_w = self.ed_ass_text
        full_text = text_w.get("1.0", tk.END).rstrip("\n")
        sel_start = len(text_w.get("1.0", tk.SEL_FIRST))
        sel_end = len(text_w.get("1.0", tk.SEL_LAST))
        style = self._get_current_style()
        new_text = AssParser.toggle_with_selection(full_text, tag_name, style, sel_start, sel_end)
        text_w.delete("1.0", tk.END)
        text_w.insert("1.0", new_text)
        text_w.focus_set()

    def _ed_ass_set_tag_at_cursor(self, tag_name, tag_str):
        text_w = self.ed_ass_text
        cursor_pos = self._raw_cursor_pos()
        full_text = text_w.get("1.0", tk.END).rstrip("\n")
        tag_value = tag_str[len("{" + tag_name):-1]
        new_text, _ = AssParser.insert_tag_at_pos(full_text, tag_name, tag_value, cursor_pos)
        text_w.delete("1.0", tk.END)
        text_w.insert("1.0", new_text)
        text_w.focus_set()

    def _ed_ass_insert_tag(self, tag):
        try:
            tag_name, _ = self._parse_tag_string(tag)
            if not tag_name:
                self.ed_ass_text.insert(tk.INSERT, tag)
                self.ed_ass_text.focus_set()
                return

            text_w = self.ed_ass_text
            has_sel = bool(text_w.tag_ranges(tk.SEL))

            if tag_name in ("\\b", "\\i", "\\u", "\\s"):
                if has_sel:
                    self._ed_ass_toggle_with_selection(tag_name)
                else:
                    self._ed_ass_toggle_tag(tag_name)
            elif has_sel:
                self._ed_ass_wrap_selection(tag)
            else:
                self._ed_ass_set_tag_at_cursor(tag_name, tag)
        except Exception:
            pass

    def _ed_ass_apply_text(self):
        if self._ed_ass_selected_idx is None or self._ed_ass_selected_idx >= len(self._ed_data):
            return
        new_text = self.ed_ass_text.get("1.0", tk.END).rstrip("\n")
        idx = self._ed_ass_selected_idx
        changed = new_text != self._ed_data[idx]["text"]
        b = self._ed_data[idx]
        b["text"] = new_text
        start = self.ed_ass_start_var.get().strip()
        end = self.ed_ass_end_var.get().strip()
        if start and end:
            new_timing = f"{start} --> {end}"
            if new_timing != b["timing"]:
                changed = True
                b["timing"] = new_timing
        b["layer"] = self.ed_ass_layer_var.get()
        style_val = self.ed_ass_style_var.get()
        if style_val and self._ed_ass_mode:
            if b.get("style") != style_val:
                changed = True
                b["style"] = style_val
        b["actor"] = self.ed_ass_actor_var.get().strip()
        b["effect"] = self.ed_ass_effect_var.get().strip()
        try:
            b["margin_l"] = str(int(self.ed_ass_ml_var.get()))
        except ValueError:
            b["margin_l"] = "0"
        try:
            b["margin_r"] = str(int(self.ed_ass_mr_var.get()))
        except ValueError:
            b["margin_r"] = "0"
        try:
            b["margin_v"] = str(int(self.ed_ass_mv_var.get()))
        except ValueError:
            b["margin_v"] = "0"
        b["comment"] = bool(self.ed_ass_comment_var.get())
        if changed:
            self._ed_push_undo()
        self._ed_refresh_tree()
        self._ed_video_update_subs()
        self.ed_status_var.set("Line applied")

    def _ed_ass_revert_text(self, raw_text=None):
        if raw_text is None:
            if self._ed_ass_selected_idx is not None and self._ed_ass_selected_idx < len(self._ed_data):
                raw_text = self._ed_data[self._ed_ass_selected_idx]["text"]
            else:
                return
        self.ed_ass_text.delete("1.0", tk.END)
        self.ed_ass_text.insert("1.0", raw_text)

    def _ed_timing_seek(self, time_str):
        if not getattr(self, '_ed_video_player', None):
            return
        try:
            t = self._parse_srt_ms(time_str.strip())
            if t > 0:
                self._ed_video_player.set_time(int(t))
                self._ed_video_player.play()
                if hasattr(self, '_ed_video_play_btn'):
                    self._ed_video_play_btn.configure(text="Pause")
        except Exception:
            pass

    def _ed_tree_idx_click(self, event):
        col = self.ed_tree.identify_column(event.x)
        if col != "#1":
            return
        item = self.ed_tree.identify_row(event.y)
        if not item:
            return
        idx = self.ed_tree.index(item)
        if idx >= len(self._ed_data):
            return
        b = self._ed_data[idx]
        parts = b["timing"].split("-->")
        if len(parts) != 2 or not getattr(self, '_ed_video_player', None):
            return
        try:
            t = self._parse_srt_ms(parts[0])
            if t > 0:
                self._ed_video_player.set_time(int(t))
                self._ed_video_player.pause()
                if hasattr(self, '_ed_video_play_btn'):
                    self._ed_video_play_btn.configure(text="Play")
        except Exception:
            pass

    def _ed_tree_idx_double(self, event):
        col = self.ed_tree.identify_column(event.x)
        if col != "#1":
            return
        item = self.ed_tree.identify_row(event.y)
        if not item:
            return
        idx = self.ed_tree.index(item)
        if idx >= len(self._ed_data):
            return
        b = self._ed_data[idx]
        parts = b["timing"].split("-->")
        if len(parts) != 2 or not getattr(self, '_ed_video_player', None):
            return
        try:
            t = self._parse_srt_ms(parts[0])
            if t > 0:
                self._ed_video_player.set_time(int(t))
                self._ed_video_player.play()
                if hasattr(self, '_ed_video_play_btn'):
                    self._ed_video_play_btn.configure(text="Pause")
        except Exception:
            pass

    def _ed_ass_set_start_from_video(self):
        if not self._ed_video_player or self._ed_ass_selected_idx is None:
            return
        t = self._ed_video_player.get_time()
        if t > 0:
            self.ed_ass_start_var.set(self._format_srt_ms(t))
            self._update_duration_display()

    def _ed_ass_set_end_from_video(self):
        if not self._ed_video_player or self._ed_ass_selected_idx is None:
            return
        t = self._ed_video_player.get_time()
        if t > 0:
            self.ed_ass_end_var.set(self._format_srt_ms(t))
            self._update_duration_display()

    def _update_duration_display(self):
        try:
            start = self._parse_srt_ms(self.ed_ass_start_var.get().strip())
            end = self._parse_srt_ms(self.ed_ass_end_var.get().strip())
            dur = end - start
            self.ed_ass_dur_var.set(f"{dur}ms" if dur >= 0 else "")
        except Exception:
            self.ed_ass_dur_var.set("")

    def _ed_ass_change_style(self, event=None):
        if self._ed_ass_selected_idx is None or self._ed_ass_selected_idx >= len(self._ed_data):
            return
        new_style = self.ed_ass_style_var.get()
        if new_style and new_style != self._ed_data[self._ed_ass_selected_idx].get("style"):
            self._ed_push_undo()
            self._ed_data[self._ed_ass_selected_idx]["style"] = new_style
            self._ed_refresh_tree()
            self.ed_status_var.set(f"Style changed to {new_style}")

    def _ed_ass_toggle_mode(self):
        self._ed_ass_mode = not self._ed_ass_mode
        self.ed_status_var.set(f"ASS mode {'enabled' if self._ed_ass_mode else 'disabled'}")
        if self._ed_ass_mode and not self._ed_ass_styles:
            self._ed_ass_auto_styles()
        self._ed_ass_revert_text()

    def _ed_ass_auto_styles(self):
        self._ed_ass_styles = {"Default": AssParser.default_style()}
        self._ed_ass_styles["Alt"] = dict(AssParser.default_style(), name="Alt", fontsize=18, alignment=8, primary_color="&H00FFFF00&")
        self._ed_ass_styles["Karaoke"] = dict(AssParser.default_style(), name="Karaoke", fontsize=24, alignment=2, primary_color="&H00FF88FF&")
        self._ed_ass_refresh_style_cb()

    def _ed_ass_refresh_style_cb(self):
        self._ed_ass_style_names = list(self._ed_ass_styles.keys()) if self._ed_ass_styles else []
        if self._ed_ass_style_names and self.ed_ass_style_var.get() not in self._ed_ass_style_names:
            self.ed_ass_style_var.set(self._ed_ass_style_names[0])

    def _ed_ass_style_manager(self):
        self._ed_ass_auto_styles()
        self.AssStyleDialog(self.root, self._ed_ass_styles, self._on_ass_styles_updated)

    def _ed_ass_context_style(self):
        if self._ed_ass_selected_idx is None or not self._ed_ass_styles:
            return
        style_name = self._ed_data[self._ed_ass_selected_idx].get("style", "Default")
        self._ed_ass_auto_styles()
        self.AssStyleDialog(self.root, self._ed_ass_styles, self._on_ass_styles_updated, select_name=style_name)

    def _ed_edit_line_dialog(self):
        sel = self.ed_tree.selection()
        if not sel or not self._ed_data:
            return
        idx = self.ed_tree.index(sel[0])
        if idx >= len(self._ed_data):
            return
        b = self._ed_data[idx]

        top = tk.Toplevel(self.root)
        top.title(f"Edit Line #{b['index']}")
        top.transient(self.root)
        top.grab_set()
        top.resizable(False, False)

        # --- Timing ---
        timing_f = ttk.LabelFrame(top, text="Timing", padding=2)
        timing_f.grid(row=0, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=(5, 2))

        ttk.Label(timing_f, text="Start:").pack(side=tk.LEFT)
        start_var = tk.StringVar(value=b["timing"].split("-->")[0].strip() if "-->" in b["timing"] else "")
        start_entry = ttk.Entry(timing_f, textvariable=start_var, width=16, font=("Consolas", 9))
        start_entry.pack(side=tk.LEFT, padx=2)

        ttk.Label(timing_f, text="End:").pack(side=tk.LEFT, padx=(8, 0))
        end_var = tk.StringVar(value=b["timing"].split("-->")[1].strip() if "-->" in b["timing"] else "")
        end_entry = ttk.Entry(timing_f, textvariable=end_var, width=16, font=("Consolas", 9))
        end_entry.pack(side=tk.LEFT, padx=2)

        ttk.Label(timing_f, text="Dur:").pack(side=tk.LEFT, padx=(8, 0))
        dur_var = tk.StringVar()
        if "-->" in b["timing"]:
            parts = b["timing"].split("-->")
            try:
                dur = self._parse_srt_ms(parts[1]) - self._parse_srt_ms(parts[0])
                dur_var.set(f"{dur}ms" if dur >= 0 else "")
            except Exception:
                pass
        ttk.Label(timing_f, textvariable=dur_var, width=10, font=("Consolas", 9)).pack(side=tk.LEFT)

        # --- Properties ---
        props_f = ttk.LabelFrame(top, text="Properties", padding=2)
        props_f.grid(row=1, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(props_f, text="Style:").pack(side=tk.LEFT)
        style_var = tk.StringVar(value=b.get("style", ""))
        style_cb = ttk.Combobox(props_f, textvariable=style_var, width=14, state="readonly")
        style_cb.configure(values=self._ed_ass_style_names)
        style_cb.pack(side=tk.LEFT, padx=2)

        sep1 = ttk.Separator(props_f, orient=tk.VERTICAL)
        sep1.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        ttk.Label(props_f, text="Layer:").pack(side=tk.LEFT)
        layer_var = tk.StringVar(value=b.get("layer", "0"))
        ttk.Spinbox(props_f, from_=-10, to=10, textvariable=layer_var, width=4).pack(side=tk.LEFT, padx=2)

        sep2 = ttk.Separator(props_f, orient=tk.VERTICAL)
        sep2.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        ttk.Label(props_f, text="Actor:").pack(side=tk.LEFT)
        actor_var = tk.StringVar(value=b.get("actor", ""))
        ttk.Entry(props_f, textvariable=actor_var, width=12).pack(side=tk.LEFT, padx=2)

        ttk.Label(props_f, text="Effect:").pack(side=tk.LEFT, padx=(4, 0))
        effect_var = tk.StringVar(value=b.get("effect", ""))
        ttk.Entry(props_f, textvariable=effect_var, width=12).pack(side=tk.LEFT, padx=2)

        sep3 = ttk.Separator(props_f, orient=tk.VERTICAL)
        sep3.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        ttk.Label(props_f, text="Margins L:").pack(side=tk.LEFT)
        ml_var = tk.StringVar(value=b.get("margin_l", "10"))
        ttk.Spinbox(props_f, from_=0, to=999, textvariable=ml_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(props_f, text="R:").pack(side=tk.LEFT)
        mr_var = tk.StringVar(value=b.get("margin_r", "10"))
        ttk.Spinbox(props_f, from_=0, to=999, textvariable=mr_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(props_f, text="V:").pack(side=tk.LEFT)
        mv_var = tk.StringVar(value=b.get("margin_v", "10"))
        ttk.Spinbox(props_f, from_=0, to=999, textvariable=mv_var, width=5).pack(side=tk.LEFT, padx=2)

        comment_var = tk.BooleanVar(value=b.get("comment", False))
        ttk.Checkbutton(props_f, text="Comment", variable=comment_var).pack(side=tk.LEFT, padx=(8, 0))

        # --- Buttons ---
        btn_f = ttk.Frame(top)
        btn_f.grid(row=2, column=0, columnspan=2, pady=8)

        def on_ok():
            self._ed_push_undo()
            start = start_var.get().strip()
            end = end_var.get().strip()
            if start and end:
                b["timing"] = f"{start} --> {end}"
            b["layer"] = layer_var.get()
            new_style = style_var.get()
            if new_style:
                b["style"] = new_style
            b["actor"] = actor_var.get().strip()
            b["effect"] = effect_var.get().strip()
            try:
                b["margin_l"] = str(int(ml_var.get()))
            except ValueError:
                b["margin_l"] = "0"
            try:
                b["margin_r"] = str(int(mr_var.get()))
            except ValueError:
                b["margin_r"] = "0"
            try:
                b["margin_v"] = str(int(mv_var.get()))
            except ValueError:
                b["margin_v"] = "0"
            b["comment"] = bool(comment_var.get())
            self._ed_refresh_tree()
            self._ed_video_update_subs()
            self.ed_status_var.set(f"Line #{b['index']} updated")
            items = self.ed_tree.get_children()
            if idx < len(items):
                self.ed_tree.selection_set(items[idx])
                self.ed_tree.see(items[idx])
                self._ed_on_select(None)
            top.destroy()

        ttk.Button(btn_f, text="OK", width=8, command=on_ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_f, text="Cancel", width=8, command=top.destroy).pack(side=tk.LEFT, padx=4)

        def _update_dur(*args):
            try:
                s = self._parse_srt_ms(start_var.get().strip())
                e = self._parse_srt_ms(end_var.get().strip())
                d = e - s
                dur_var.set(f"{d}ms" if d >= 0 else "")
            except Exception:
                dur_var.set("")
        start_var.trace_add("write", _update_dur)
        end_var.trace_add("write", _update_dur)

        top.wait_window()

    def _on_ass_styles_updated(self, styles):
        self._ed_ass_styles = styles
        self._ed_ass_refresh_style_cb()
        self.ed_status_var.set(f"Styles updated ({len(styles)} styles)")

    def _ed_ass_color_pick(self, color_key):
        tag_map = {"primary_color": "\\c", "secondary_color": "\\2c", "outline_color": "\\3c", "shadow_color": "\\4c"}
        tag_name = tag_map.get(color_key, "\\c")
        cp = self.AssColorPicker(self.root)
        if cp.result is not None:
            text_w = self.ed_ass_text
            tag_value = cp.result
            if text_w.tag_ranges(tk.SEL):
                sel_start = len(text_w.get("1.0", tk.SEL_FIRST))
                sel_end = len(text_w.get("1.0", tk.SEL_LAST))
                full_text = text_w.get("1.0", tk.END).rstrip("\n")
                new_text, shift = AssParser.insert_tag_at_pos(full_text, tag_name, tag_value, sel_start)
                new_text, _ = AssParser.insert_tag_at_pos(new_text, tag_name, tag_value, sel_end + shift)
                text_w.delete("1.0", tk.END)
                text_w.insert("1.0", new_text)
            else:
                cursor_pos = self._raw_cursor_pos()
                full_text = text_w.get("1.0", tk.END).rstrip("\n")
                new_text, _ = AssParser.insert_tag_at_pos(full_text, tag_name, tag_value, cursor_pos)
                text_w.delete("1.0", tk.END)
                text_w.insert("1.0", new_text)
            text_w.focus_set()


    class AssColorPicker:
        def __init__(self, parent):
            self.result = None
            self.win = tk.Toplevel(parent)
            self.win.title("ASS Color Picker")
            self.win.transient(parent)
            self.win.grab_set()
            self.win.geometry("360x280")

            main = ttk.Frame(self.win, padding=8)
            main.pack(fill=tk.BOTH, expand=True)

            ttk.Label(main, text="ASS color format: &HBBGGRR", font=("TkDefaultFont", 9, "italic")).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))

            ttk.Label(main, text="Blue (B):").grid(row=1, column=0, sticky=tk.W)
            self.b_var = tk.IntVar(value=255)
            self.b_scale = ttk.Scale(main, from_=0, to=255, variable=self.b_var, command=self._update_preview, length=200)
            self.b_scale.grid(row=1, column=1, padx=4)
            self.b_label = ttk.Label(main, text="255", width=4)
            self.b_label.grid(row=1, column=2, sticky=tk.W)

            ttk.Label(main, text="Green (G):").grid(row=2, column=0, sticky=tk.W)
            self.g_var = tk.IntVar(value=255)
            self.g_scale = ttk.Scale(main, from_=0, to=255, variable=self.g_var, command=self._update_preview, length=200)
            self.g_scale.grid(row=2, column=1, padx=4)
            self.g_label = ttk.Label(main, text="255", width=4)
            self.g_label.grid(row=2, column=2, sticky=tk.W)

            ttk.Label(main, text="Red (R):").grid(row=3, column=0, sticky=tk.W)
            self.r_var = tk.IntVar(value=255)
            self.r_scale = ttk.Scale(main, from_=0, to=255, variable=self.r_var, command=self._update_preview, length=200)
            self.r_scale.grid(row=3, column=1, padx=4)
            self.r_label = ttk.Label(main, text="255", width=4)
            self.r_label.grid(row=3, column=2, sticky=tk.W)

            self.preview = tk.Canvas(main, width=80, height=40, bg="white", highlightthickness=1, highlightbackground="#ccc")
            self.preview.grid(row=4, column=0, columnspan=3, pady=8)
            self.hex_var = tk.StringVar(value="&H00FFFFFF")
            ttk.Label(main, textvariable=self.hex_var, font=("Consolas", 11)).grid(row=5, column=0, columnspan=3, pady=2)

            btn_f = ttk.Frame(self.win, padding=6)
            btn_f.pack(fill=tk.X)
            ttk.Button(btn_f, text="OK", command=self._ok).pack(side=tk.RIGHT, padx=2)
            ttk.Button(btn_f, text="Cancel", command=self.win.destroy).pack(side=tk.RIGHT, padx=2)

            self._update_preview()

        def _update_preview(self, *_):
            b = self.b_var.get()
            g = self.g_var.get()
            r = self.r_var.get()
            self.b_label.configure(text=str(b))
            self.g_label.configure(text=str(g))
            self.r_label.configure(text=str(r))
            hex_str = f"&H00{b:02X}{g:02X}{r:02X}"
            self.hex_var.set(hex_str)
            tk_color = f"#{r:02x}{g:02x}{b:02x}"
            self.preview.configure(bg=tk_color)

        def _ok(self):
            b = self.b_var.get()
            g = self.g_var.get()
            r = self.r_var.get()
            self.result = f"&H00{b:02X}{g:02X}{r:02X}"
            self.win.destroy()


    class AssStyleDialog:
        def __init__(self, parent, styles, on_save, select_name=None):
            self.parent = parent
            self.styles = styles
            self.on_save = on_save
            self.result = dict(styles)
            self.current_name = None
            self._select_name = select_name

            self.win = tk.Toplevel(parent)
            self.win.title("ASS Style Manager")
            self.win.transient(parent)
            self.win.grab_set()
            self.win.geometry("700x520")

            main = ttk.Frame(self.win, padding=6)
            main.pack(fill=tk.BOTH, expand=True)
            main.columnconfigure(1, weight=1)

            ttk.Label(main, text="Styles:", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky=tk.W)

            list_frame = ttk.Frame(main)
            list_frame.grid(row=1, column=0, sticky=tk.NS, padx=(0, 8))
            self.style_lb = tk.Listbox(list_frame, width=16, height=12)
            self.style_lb.pack(side=tk.LEFT, fill=tk.Y, expand=True)
            sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.style_lb.yview)
            self.style_lb.configure(yscrollcommand=sb.set)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            self.style_lb.bind("<<ListboxSelect>>", self._on_select)

            btn_frame = ttk.Frame(main)
            btn_frame.grid(row=2, column=0, pady=4)
            ttk.Button(btn_frame, text="+", width=3, command=self._add_style).pack(side=tk.LEFT, padx=1)
            ttk.Button(btn_frame, text="-", width=3, command=self._delete_style).pack(side=tk.LEFT, padx=1)

            self._build_editor(main)
            self._rebuild_list()
            if self.style_lb.size() > 0:
                select_idx = 0
                if self._select_name:
                    for i in range(self.style_lb.size()):
                        if self.style_lb.get(i) == self._select_name:
                            select_idx = i
                            break
                self.style_lb.selection_set(select_idx)
                self._on_select()

            btn_row = ttk.Frame(self.win, padding=6)
            btn_row.pack(fill=tk.X)
            ttk.Button(btn_row, text="OK", command=self._save).pack(side=tk.RIGHT, padx=2)
            ttk.Button(btn_row, text="Cancel", command=self.win.destroy).pack(side=tk.RIGHT, padx=2)

            self.win.protocol("WM_DELETE_WINDOW", self.win.destroy)

        _FIELDS = [
            ("fontname", "Font:"), ("fontsize", "Size:"), ("bold", "Bold:"),
            ("italic", "Italic:"), ("underline", "Underline:"), ("strikeout", "StrikeOut:"),
            ("scale_x", "Scale X:"), ("scale_y", "Scale Y:"), ("spacing", "Spacing:"),
            ("angle", "Angle:"), ("border_style", "Border:"), ("outline", "Outline:"),
            ("shadow", "Shadow:"), ("alignment", "Align:"),
            ("margin_l", "Margin L:"), ("margin_r", "Margin R:"), ("margin_v", "Margin V:"),
        ]
        _COLOR_FIELDS = ["primary_color", "secondary_color", "outline_color", "shadow_color"]
        _INT_FIELDS = {"fontsize", "bold", "italic", "underline", "strikeout", "scale_x", "scale_y",
                        "spacing", "angle", "border_style", "outline", "shadow", "alignment",
                        "margin_l", "margin_r", "margin_v", "encoding"}
        _BOOL_FIELDS = {"bold", "italic", "underline", "strikeout"}

        def _build_editor(self, parent):
            ef = ttk.LabelFrame(parent, text="Style Properties", padding=6)
            ef.grid(row=1, column=1, sticky=tk.NSEW, rowspan=2)
            parent.rowconfigure(1, weight=1)

            self._widgets = {}
            row = 0
            for key, label in self._FIELDS:
                ttk.Label(ef, text=label).grid(row=row, column=0, sticky=tk.W, pady=1)
                if key in self._BOOL_FIELDS:
                    w = ttk.Combobox(ef, values=["0", "1"], width=6, state="readonly")
                elif key == "alignment":
                    w = ttk.Combobox(ef, values=[str(i) for i in range(1, 10)], width=6, state="readonly")
                elif key == "border_style":
                    w = ttk.Combobox(ef, values=["1", "3"], width=6, state="readonly")
                elif key in ("fontsize", "outline", "shadow", "spacing", "angle", "scale_x", "scale_y"):
                    w = ttk.Spinbox(ef, from_=0, to=999, width=6)
                elif key in ("margin_l", "margin_r", "margin_v"):
                    w = ttk.Spinbox(ef, from_=0, to=999, width=6)
                else:
                    w = ttk.Entry(ef, width=20)
                w.grid(row=row, column=1, sticky=tk.W, padx=4, pady=1)
                self._widgets[key] = w
                row += 1

            ttk.Label(ef, text="Colors:", font=("TkDefaultFont", 9, "bold")).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(6, 2))
            row += 1
            for key in self._COLOR_FIELDS:
                ttk.Label(ef, text=key.replace("_", " ").title() + ":").grid(row=row, column=0, sticky=tk.W, pady=1)
                w = ttk.Entry(ef, width=22)
                w.grid(row=row, column=1, sticky=tk.W, padx=4, pady=1)
                self._widgets[key] = w
                row += 1

        def _rebuild_list(self):
            self.style_lb.delete(0, tk.END)
            for name in self.result:
                self.style_lb.insert(tk.END, name)

        def _on_select(self, event=None):
            sel = self.style_lb.curselection()
            if not sel:
                return
            name = self.style_lb.get(sel[0])
            self.current_name = name
            s = self.result[name]
            for key, w in self._widgets.items():
                val = s.get(key, "")
                if key in self._BOOL_FIELDS:
                    w.set(str(val) if val in (0, 1, "0", "1") else "0")
                elif key in ("alignment", "border_style"):
                    w.set(str(val))
                elif key in self._INT_FIELDS:
                    w.set(str(val))
                else:
                    w.delete(0, tk.END)
                    w.insert(0, str(val))

        def _read_values(self):
            vals = {}
            for key, w in self._widgets.items():
                if isinstance(w, ttk.Entry):
                    val = w.get().strip()
                elif isinstance(w, ttk.Combobox):
                    val = w.get().strip()
                else:
                    val = w.get().strip()
                if key in self._BOOL_FIELDS:
                    vals[key] = 1 if val in ("1", "true", "True", "-1") else 0
                elif key in self._INT_FIELDS:
                    try:
                        vals[key] = int(val) if val else 0
                    except ValueError:
                        vals[key] = 0
                else:
                    vals[key] = val
            return vals

        def _add_style(self):
            import tkinter.simpledialog as sd
            name = sd.askstring("New Style", "Style name:", parent=self.win)
            if not name or name in self.result:
                return
            base = AssParser.default_style()
            base["name"] = name
            self.result[name] = base
            self._rebuild_list()
            self.style_lb.selection_clear(0, tk.END)
            self.style_lb.selection_set(tk.END)
            self._on_select()

        def _delete_style(self):
            sel = self.style_lb.curselection()
            if not sel:
                return
            name = self.style_lb.get(sel[0])
            if name == "Default":
                messagebox.showinfo("Cannot delete", "Default style cannot be deleted", parent=self.win)
                return
            del self.result[name]
            self._rebuild_list()
            if self.style_lb.size() > 0:
                self.style_lb.selection_set(0)
                self._on_select()

        def _save(self):
            if self.current_name and self.current_name in self.result:
                self.result[self.current_name] = self._read_values()
            self.on_save(dict(self.result))
            self.win.destroy()


    # ---- Context Menu Operations ----

    def _ed_copy_lines(self):
        sel = self.ed_tree.selection()
        if not sel:
            return
        self._ed_clipboard = []
        for item in sel:
            idx = self.ed_tree.index(item)
            if idx < len(self._ed_data):
                self._ed_clipboard.append(copy.deepcopy(self._ed_data[idx]))
        self.ed_status_var.set(f"Copied {len(self._ed_clipboard)} line(s)")

    def _ed_paste_lines(self):
        if not self._ed_clipboard:
            return
        sel = self.ed_tree.selection()
        insert_idx = len(self._ed_data)
        if sel:
            insert_idx = self.ed_tree.index(sel[0]) + 1
        self._ed_push_undo()
        for i, block in enumerate(self._ed_clipboard):
            new = copy.deepcopy(block)
            self._ed_data.insert(insert_idx + i, new)
        self._ed_refresh_tree()
        self._ed_renumber()
        self._ed_video_update_subs()
        self.ed_status_var.set(f"Pasted {len(self._ed_clipboard)} line(s)")

    def _ed_copy_text(self):
        sel = self.ed_tree.selection()
        if not sel:
            return
        texts = []
        for item in sel:
            idx = self.ed_tree.index(item)
            if idx < len(self._ed_data):
                texts.append(self._ed_data[idx]["text"])
        text = "\n".join(texts)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.ed_status_var.set("Text copied to clipboard")

    def _ed_paste_text(self):
        sel = self.ed_tree.selection()
        if not sel:
            return
        try:
            text = self.root.clipboard_get()
        except Exception:
            return
        self._ed_push_undo()
        for item in sel:
            idx = self.ed_tree.index(item)
            if idx < len(self._ed_data):
                self._ed_data[idx]["text"] = text
        self._ed_refresh_tree()
        self._ed_video_update_subs()
        self.ed_status_var.set("Text pasted")

    def _ed_select_all(self):
        self.ed_tree.selection_set(self.ed_tree.get_children())

    def _ed_swap_timing(self):
        sel = self.ed_tree.selection()
        if not sel:
            return
        self._ed_push_undo()
        for item in sel:
            idx = self.ed_tree.index(item)
            if idx < len(self._ed_data):
                parts = self._ed_data[idx]["timing"].split("-->")
                if len(parts) == 2:
                    self._ed_data[idx]["timing"] = f"{parts[1].strip()} --> {parts[0].strip()}"
        self._ed_refresh_tree()
        self._ed_video_update_subs()
        self.ed_status_var.set("Timing swapped")

    def _ed_toggle_comment(self):
        sel = self.ed_tree.selection()
        if not sel:
            return
        self._ed_push_undo()
        for item in sel:
            idx = self.ed_tree.index(item)
            if idx < len(self._ed_data):
                current = self._ed_data[idx].get("comment", False)
                self._ed_data[idx]["comment"] = not current
        self._ed_refresh_tree()
        self.ed_status_var.set("Comment toggled")

    def _ed_sort_by_number(self):
        if not self._ed_data:
            return
        self._ed_push_undo()
        self._ed_data.sort(key=lambda b: int(b.get("index", 0)))
        self._ed_refresh_tree()
        self._ed_renumber()
        self.ed_status_var.set("Sorted by number")

    def _ed_sort_by_time(self):
        if not self._ed_data:
            return
        self._ed_push_undo()
        def sort_key(b):
            parts = b["timing"].split("-->")
            if len(parts) == 2:
                return self._parse_srt_ms(parts[0])
            return 0
        self._ed_data.sort(key=sort_key)
        self._ed_refresh_tree()
        self._ed_renumber()
        self.ed_status_var.set("Sorted by time")

    def _ed_sort_by_text(self):
        if not self._ed_data:
            return
        self._ed_push_undo()
        self._ed_data.sort(key=lambda b: b.get("text", ""))
        self._ed_refresh_tree()
        self._ed_renumber()
        self.ed_status_var.set("Sorted by text")

    def _ed_context_menu(self, event):
        item = self.ed_tree.identify_row(event.y)
        if item:
            self.ed_tree.selection_set(item)
            self._ed_context.tk_popup(event.x_root, event.y_root)

    def _ed_lookup_dict(self):
        sel = self.ed_tree.selection()
        if not sel:
            return
        vals = self.ed_tree.item(sel[0], "values")
        text = vals[2] if len(vals) > 2 else ""
        text = text.strip()
        if text:
            self.dict_entry.delete(0, tk.END)
            self.dict_entry.insert(0, text)
            self._dict_lookup()
            self.notebook.select(2)

    def _log_trans(self, msg):
        self.trans_log.insert(tk.END, msg + "\n")
        self.trans_log.see(tk.END)
        self.trans_log.update_idletasks()

    def _browse_srt(self):
        path = filedialog.askopenfilename(title="Select SRT file", filetypes=[("SRT files", "*.srt"), ("All files", "*.*")])
        if path:
            self.srt_entry.delete(0, tk.END)
            self.srt_entry.insert(0, path)
            # Auto-fill output name
            p = Path(path)
            out = str(p.parent / f"{p.stem}_translated.srt")
            self._trans_out_path = out

    def _get_saved_key(self, cfg):
        svc = self.svc_var.get().split(" ")[0] if hasattr(self, 'svc_var') else cfg.get("service", "local")
        return cfg.get(f"{svc}_key", "") if svc != "local" else ""

    def _on_svc_change(self, event=None):
        cfg = load_config()
        svc = self.svc_var.get().split(" ")[0]
        needs_key = svc in ("deepl", "openai", "google", "huggingface")
        self.api_key_label.configure(state=tk.NORMAL if needs_key else tk.DISABLED)
        self.api_entry.configure(state=tk.NORMAL if needs_key else tk.DISABLED)
        if needs_key:
            self.api_key_var.set(cfg.get(f"{svc}_key", ""))
        else:
            self.api_key_var.set("")

    def _start_translation(self):
        srt_path = self.srt_entry.get().strip()
        if not srt_path or not os.path.isfile(srt_path):
            messagebox.showerror("Error", "Select an SRT file first")
            return

        src_raw = self.src_lang_var.get()
        tgt_raw = self.tgt_lang_var.get()
        src = src_raw.split(" ")[0] if src_raw else "auto"
        tgt = tgt_raw.split(" ")[0] if tgt_raw else "PL"
        svc = self.svc_var.get().split(" ")[0]
        api_key = self.api_key_var.get().strip()

        needs_key = svc in ("deepl", "openai", "google", "huggingface")
        if needs_key and not api_key:
            messagebox.showerror("Error", f"Enter your {svc.title()} API Key")
            return

        # Save config (skip for local)
        cfg = load_config()
        cfg["service"] = svc
        if svc != "local":
            cfg["deepl_key"] = api_key if svc == "deepl" else cfg.get("deepl_key", "")
            cfg["openai_key"] = api_key if svc == "openai" else cfg.get("openai_key", "")
            cfg["google_key"] = api_key if svc == "google" else cfg.get("google_key", "")
            cfg["libre_key"] = api_key if svc == "libre" else cfg.get("libre_key", "")
            cfg["hf_key"] = api_key if svc == "huggingface" else cfg.get("hf_key", "")
        save_config(cfg)

        p = Path(srt_path)
        suffix = {"dictionary": "gloss", "dictionary2": "gloss2", "dictionary3": "gloss3"}.get(svc, tgt)
        out_path = str(p.parent / f"{p.stem}_{suffix}.srt")

        self.translate_btn.configure(state=tk.DISABLED)
        self.translate_cancel_btn.configure(state=tk.NORMAL)
        self._trans_cancel = False

        self._log_trans(f"Translating: {p.name}")
        self._log_trans(f"  {src_raw} -> {tgt_raw} via {svc}")
        self.trans_progress.configure(value=0)
        self.trans_progress_label.configure(text="0%")

        self._trans_thread = threading.Thread(target=self._translate_worker, args=(srt_path, out_path, src, tgt, svc, api_key), daemon=True)
        self._trans_thread.start()
        self._poll_trans_progress()

    def _translate_worker(self, srt_path, out_path, src, tgt, svc, api_key):
        def progress(pct):
            self.root.after(0, lambda: self.trans_progress.configure(value=pct))

        def cancelled():
            return self._trans_cancel

        try:
            result = translate_srt_file(srt_path, out_path, src, tgt, svc, api_key, progress, cancelled)
            if result is None:
                self.root.after(0, lambda: self._log_trans("Cancelled."))
            else:
                self.root.after(0, lambda: self._log_trans(f"Done! {out_path}"))
        except Exception as e:
            import traceback
            err_msg = str(e)
            traceback.print_exc()
            self.root.after(0, lambda m=err_msg: self._log_trans(f"Error: {m}"))
        finally:
            self.root.after(0, self._on_translate_done)

    def _poll_trans_progress(self):
        if hasattr(self, '_trans_thread') and self._trans_thread and self._trans_thread.is_alive():
            self.root.after(500, self._poll_trans_progress)
        else:
            self._on_translate_done()

    def _on_translate_done(self):
        self.translate_btn.configure(state=tk.NORMAL)
        self.translate_cancel_btn.configure(state=tk.DISABLED)
        self.trans_progress.configure(value=100)
        self.trans_progress_label.configure(text="100%")

    def _cancel_translation(self):
        self._trans_cancel = True
        self._log_trans("Cancelling...")

    def _browse_files(self):
        files = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
                ("All files", "*.*"),
            ],
        )
        if files:
            self.video_paths = list(files)
            self.file_entry.delete(0, tk.END)
            if len(files) == 1:
                self.file_entry.insert(0, files[0])
                self.preview_btn.configure(state=tk.NORMAL)
            else:
                self.file_entry.insert(0, f"{len(files)} files selected")
                self.preview_btn.configure(state=tk.DISABLED)

    def _parse_area(self):
        area_str = self.area_entry.get().strip()
        if not area_str:
            return None
        parts = area_str.replace(" ", "").split(",")
        if len(parts) == 4:
            try:
                xmin, xmax, ymin, ymax = map(int, parts)
                return SubtitleArea(ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)
            except ValueError:
                pass
        if len(parts) == 2:
            try:
                ymin, ymax = map(int, parts)
                return SubtitleArea(ymin=ymin, ymax=ymax, xmin=0, xmax=99999)
            except ValueError:
                pass
        messagebox.showwarning("Invalid area", "Use format: xmin,xmax,ymin,ymax or ymin,ymax")
        return None

    def _parse_watermark_area(self):
        area_str = self.watermark_entry.get().strip()
        if not area_str:
            return None
        parts = area_str.replace(" ", "").split(",")
        if len(parts) == 4:
            try:
                xmin, xmax, ymin, ymax = map(int, parts)
                return SubtitleArea(ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)
            except ValueError:
                pass
        messagebox.showwarning("Invalid watermark area", "Use format: xmin,xmax,ymin,ymax")
        return None

    def _open_preview(self):
        path = self.file_entry.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Select a single video file first")
            return

        sub_area = self._parse_area()
        wm_area = self._parse_watermark_area()

        try:
            VideoPreviewWindow(
                self.root, path,
                on_confirm=self._on_preview_confirm,
                sub_area=sub_area,
                wm_area=wm_area,
            )
        except Exception as e:
            messagebox.showerror("Preview Error", f"Could not open video preview:\n{e}")

    def _on_preview_confirm(self, sub_area, wm_area):
        if sub_area is None and wm_area is None:
            return
        if sub_area is not None:
            self.area_entry.delete(0, tk.END)
            self.area_entry.insert(0, f"{sub_area.xmin},{sub_area.xmax},{sub_area.ymin},{sub_area.ymax}")
        if wm_area is not None:
            self.watermark_entry.delete(0, tk.END)
            self.watermark_entry.insert(0, f"{wm_area.xmin},{wm_area.xmax},{wm_area.ymin},{wm_area.ymax}")

    def _start_extraction(self):
        if self.running:
            return

        if not self.video_paths:
            path = self.file_entry.get().strip()
            if path and os.path.isfile(path):
                self.video_paths = [path]
            elif path and os.path.isdir(path):
                self.video_paths = [str(p) for p in Path(path).glob("*.mp4")]
            else:
                messagebox.showerror("Error", "Select a video file first")
                return

        area = self._parse_area()
        watermark_area = self._parse_watermark_area()
        lang = self.lang_var.get().split(" ")[0]

        config.language = lang
        config.mode = self.mode_var.get()
        config.use_gpu = False
        config.num_ocr_workers = max(1, int(self.workers_var.get()))
        config.extract_frequency = int(self.fps_var.get())
        config.threshold_text_similarity = int(self.sim_var.get())
        config.drop_score = int(self.drop_var.get())
        config.generate_txt = self.txt_var.get()
        config.debug_no_delete_cache = self.cache_var.get()

        self.running = True
        self._cancel_requested = False
        self._start_time = time.time()
        self.progress.configure(value=0)
        self.progress_label.configure(text="0%")
        self.start_btn.configure(state=tk.DISABLED)
        self.pause_btn.configure(state=tk.NORMAL, text="Pause")
        self.cancel_btn.configure(state=tk.NORMAL)
        self._log(f"\nStarting extraction of {len(self.video_paths)} video(s)...")

        self.extraction_thread = threading.Thread(target=self._run_extraction, args=(area, watermark_area), daemon=True)
        self.extraction_thread.start()
        self._poll_progress()

    def _run_extraction(self, area, watermark_area):
        old_stdout = sys.stdout
        handler = TextHandler(self.log_text)
        sys.stdout = handler
        try:
            for i, video_path in enumerate(self.video_paths, 1):
                if self._cancel_requested:
                    print("Cancelled by user.")
                    break
                if not os.path.isfile(video_path):
                    print(f"[{i}/{len(self.video_paths)}] SKIP: {video_path}")
                    continue
                print(f"\n[{i}/{len(self.video_paths)}] {os.path.basename(video_path)}")

                self.extractor = SubtitleExtractor(video_path, area, watermark_area)
                self.extractor.run()
                if self.extractor.is_cancelled:
                    print("Cancelled.")
                else:
                    print(f"OK: {self.extractor.subtitle_output_path}")
                self.extractor = None
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            self.root.after(0, self._on_finished)

    def _poll_progress(self):
        if self.running and self.extractor:
            frame_pct = self.extractor.progress_frame_extract
            ocr_pct = self.extractor.progress_ocr
            post_pct = self.extractor.progress_post
            total = frame_pct * 0.10 + ocr_pct * 0.85 + post_pct * 0.05
            self.progress.configure(value=total)

            eta = ""
            if self._start_time and total > 0:
                elapsed = time.time() - self._start_time
                if total >= 100:
                    eta = f"  ({elapsed:.0f}s)"
                else:
                    remaining = elapsed * (100 - total) / total
                    eta = f"  ETA {remaining:.0f}s  ({elapsed:.0f}s elapsed)"

            self.progress_label.configure(text=f"{total:.0f}%{eta}")
        if self.running:
            self.root.after(500, self._poll_progress)

    def _toggle_pause(self):
        if not self.extractor:
            return
        if self.extractor.is_paused:
            self.extractor.resume()
            self.pause_btn.configure(text="Pause")
            self._log("Resumed.")
        else:
            self.extractor.pause()
            self.pause_btn.configure(text="Resume")
            self._log("Paused.")

    def _cancel_extraction(self):
        self._cancel_requested = True
        if self.extractor:
            self._log("Cancelling...")
            self.extractor.cancel()
        self.pause_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.DISABLED)

    def _on_finished(self):
        self.running = False
        self.extractor = None
        self.start_btn.configure(state=tk.NORMAL)
        self.pause_btn.configure(state=tk.DISABLED, text="Pause")
        self.cancel_btn.configure(state=tk.DISABLED)
        self.progress.configure(value=100)
        self.progress_label.configure(text="100%")
        self.video_paths = []
        self._log("Done!\n")

    def _on_close(self):
        if self.running:
            if messagebox.askyesno("Confirm", "Extraction in progress. Cancel and exit?"):
                self._cancel_extraction()
                self.root.destroy()
            else:
                return
        self._ed_video_close()
        self.root.destroy()

    def _log(self, text):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = SubtitleExtractorGUI(root)
    root.mainloop()
