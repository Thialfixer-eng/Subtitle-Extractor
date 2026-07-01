#!/usr/bin/env python
import os
import sys
import argparse
import multiprocessing

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import config, VERSION
from backend.bean.subtitle_area import SubtitleArea
from backend.subtitle_extractor import SubtitleExtractor


LANGUAGES = {
    "ch": "Chinese (Simplified)",
    "en": "English",
    "japan": "Japanese",
    "korean": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ar": "Arabic",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "chinese_cht": "Chinese (Traditional)",
}


def parse_subtitle_area(area_str):
    if not area_str:
        return None
    parts = area_str.replace(" ", "").split(",")
    if len(parts) == 4:
        try:
            xmin, xmax, ymin, ymax = map(int, parts)
            return SubtitleArea(ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)
        except ValueError:
            pass
    try:
        ymin, ymax = map(int, parts[:2])
        return SubtitleArea(ymin=ymin, ymax=ymax, xmin=0, xmax=99999)
    except ValueError:
        pass
    return None


def parse_watermark_area(area_str):
    return parse_subtitle_area(area_str)


def main():
    parser = argparse.ArgumentParser(
        description=f"Video Subtitle Extractor v{VERSION} - Extract hard-coded subtitles to SRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4                          Basic extraction
  %(prog)s video.mp4 --lang en                English video
  %(prog)s video.mp4 --area 0,1920,900,1080   Subtitle region (faster)
  %(prog)s video.mp4 --watermark-area 0,200,0,100   Exclude watermark
  %(prog)s video.mp4 --fps 5                  Higher frame rate
  %(prog)s video.mp4 --mode auto              Smart detection mode
  %(prog)s video.mp4 --txt                    Also generate .txt
  %(prog)s video.mp4 --no-word-seg            Skip word segmentation
  %(prog)s "D:\\videos\\movie1.mp4" "D:\\videos\\movie2.mp4" --batch

Output:
  output/<video_name>.srt

Translation (GUI only):
  Open the GUI (gui.py), go to Translation tab, select an SRT file,
  choose service (local/libre/huggingface/deepl/openai/google) and
  languages, then click Translate.

Notes:
  - CPU only (PaddlePaddle CPU build). Use --gpu for CUDA acceleration.
  - Specify --area to crop frames before OCR for faster processing.
  - Use --watermark-area to exclude logos/watermarks from results.
  - --mode fast extracts frames at intervals; accurate processes each frame.
        """,
    )

    parser.add_argument("video", nargs="+", help="Path to video file(s)")
    parser.add_argument("--lang", default="ch", choices=list(LANGUAGES.keys()),
                        help=f"OCR language (default: ch). Options: {', '.join(f'{k}({v})' for k, v in LANGUAGES.items())}")
    parser.add_argument("--mode", default="fast", choices=["fast", "auto", "accurate"],
                        help="Extraction mode: fast (quick, may miss some), auto, accurate (slow, precise)")
    parser.add_argument("--area", default=None,
                        help="Subtitle area: xmin,xmax,ymin,ymax or ymin,ymax (e.g. '0,1920,900,1080')")
    parser.add_argument("--watermark-area", default=None,
                        help="Watermark exclusion area: xmin,xmax,ymin,ymax (detections in this zone are skipped)")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration (requires CUDA/PaddlePaddle GPU)")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID (default: 0)")
    parser.add_argument("--fps", type=int, default=3,
                        help="Frames per second to extract (default: 3)")
    parser.add_argument("--similarity", type=int, default=80,
                        help="Text similarity threshold 0-100 for dedup (default: 80)")
    parser.add_argument("--drop-score", type=int, default=75,
                        help="OCR confidence threshold 0-100 (default: 75)")
    parser.add_argument("--no-word-seg", action="store_true", help="Disable word segmentation")
    parser.add_argument("--txt", action="store_true", help="Also generate TXT file")
    parser.add_argument("--no-cache", action="store_true", help="Delete cache after extraction")
    parser.add_argument("--batch", action="store_true", help="Process multiple videos in batch mode")
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    args = parser.parse_args()

    if args.version:
        print(f"Video Subtitle Extractor v{VERSION}")
        return

    if args.batch:
        videos = args.video
    else:
        videos = args.video

    config.language = args.lang
    config.mode = args.mode
    config.use_gpu = args.gpu
    config.gpu_id = args.gpu_id
    config.extract_frequency = args.fps
    config.threshold_text_similarity = args.similarity
    config.drop_score = args.drop_score
    config.word_segmentation = not args.no_word_seg
    config.generate_txt = args.txt
    config.debug_no_delete_cache = not args.no_cache

    sub_area = parse_subtitle_area(args.area)
    watermark_area = parse_watermark_area(args.watermark_area)

    if sub_area:
        print(f"Subtitle area: x=[{sub_area.xmin}, {sub_area.xmax}] y=[{sub_area.ymin}, {sub_area.ymax}]")
    else:
        print("No subtitle area specified - will attempt full-frame detection")
    if watermark_area:
        print(f"Watermark exclusion: x=[{watermark_area.xmin}, {watermark_area.xmax}] y=[{watermark_area.ymin}, {watermark_area.ymax}]")

    if config.use_gpu:
        print("GPU acceleration enabled")

    print(f"{'Batch processing' if args.batch else 'Processing'}: {len(videos)} video(s)")
    print()

    for i, video_path in enumerate(videos, 1):
        if not os.path.isfile(video_path):
            print(f"[{i}/{len(videos)}] SKIP (not found): {video_path}")
            continue

        print(f"[{i}/{len(videos)}] Processing: {video_path}")
        print("-" * 60)

        extractor = SubtitleExtractor(video_path, sub_area, watermark_area)
        extractor.run()

        print("-" * 60)
        print()

    print("All done!")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
