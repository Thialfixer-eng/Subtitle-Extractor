import os
import json
import re
import sys

from backend.config import BASE_DIR, config


def load_typo_map():
    typo_path = os.path.join(BASE_DIR, "configs", "typoMap.json")
    if os.path.exists(typo_path):
        with open(typo_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def fix_typos(text, typo_map):
    for wrong, correct in typo_map.items():
        text = text.replace(wrong, correct)
    return text


def execute(path, lang="ch"):
    typo_map = load_typo_map()

    try:
        import pysrt
        subs = pysrt.open(path, encoding="utf-8")
    except Exception:
        return

    for sub in subs:
        text = sub.text
        text = fix_typos(text, typo_map)
        text = re.sub(r"\s+", " ", text).strip()
        sub.text = text

    subs.save(path, encoding="utf-8")
