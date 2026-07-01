import os
import re
import json
import time
import threading
from pathlib import Path

DEEPL_LANGUAGES = {
    "BG": "Bulgarian", "CS": "Czech", "DA": "Danish", "DE": "German",
    "EL": "Greek", "EN": "English", "ES": "Spanish", "ET": "Estonian",
    "FI": "Finnish", "FR": "French", "HU": "Hungarian", "ID": "Indonesian",
    "IT": "Italian", "JA": "Japanese", "KO": "Korean", "LT": "Lithuanian",
    "LV": "Latvian", "NB": "Norwegian", "NL": "Dutch", "PL": "Polish",
    "PT": "Portuguese", "RO": "Romanian", "RU": "Russian", "SK": "Slovak",
    "SL": "Slovenian", "SV": "Swedish", "TR": "Turkish", "UK": "Ukrainian",
    "ZH": "Chinese",
}

SRC_LANGUAGES = {"auto": "Auto-detect"} | DEEPL_LANGUAGES

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "translator_config.json")


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"deepl_key": "", "openai_key": "", "google_key": "", "libre_key": "", "hf_key": "", "service": "local"}


def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


class SrtParser:
    @staticmethod
    def parse_blocks(srt_text):
        blocks = []
        lines = srt_text.split("\n")
        current = None
        text_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped and current is not None:
                if text_lines:
                    current["text"] = "\n".join(text_lines)
                    blocks.append(current)
                current = None
                text_lines = []
            elif current is None and stripped.isdigit():
                current = {"index": stripped, "timing": None, "text": None}
            elif current is not None and current["timing"] is None and "-->" in stripped:
                current["timing"] = stripped
            elif current is not None:
                text_lines.append(line.rstrip("\n"))

        if current is not None and text_lines:
            current["text"] = "\n".join(text_lines)
            blocks.append(current)

        return blocks

    @staticmethod
    def blocks_to_srt(blocks):
        lines = []
        for b in blocks:
            lines.append(b["index"])
            lines.append(b["timing"])
            lines.append(b["text"])
            lines.append("")
        return "\n".join(lines)


class AssParser:
    @staticmethod
    def default_style():
        return {
            "name": "Default", "fontname": "Arial", "fontsize": 20,
            "bold": 0, "italic": 0, "underline": 0, "strikeout": 0,
            "scale_x": 100, "scale_y": 100, "spacing": 0, "angle": 0,
            "border_style": 1, "outline": 2, "shadow": 2,
            "alignment": 2, "margin_l": 10, "margin_r": 10, "margin_v": 10,
            "encoding": 1,
            "primary_color": "&H00FFFFFF", "secondary_color": "&H000000FF",
            "outline_color": "&H00000000", "shadow_color": "&H00000000",
        }

    @staticmethod
    def ass_color_to_int(color_str):
        try:
            c = color_str.replace("&H", "").replace("&", "")
            return int(c, 16) if c else 0x00FFFFFF
        except ValueError:
            return 0x00FFFFFF

    @staticmethod
    def int_to_ass_color(val):
        return f"&H{val:08X}&"

    @staticmethod
    def parse_ass(text):
        blocks = []
        styles = {}
        current_section = None
        format_fields = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                current_section = stripped[1:-1]
                format_fields = []
                continue
            if current_section == "V4+ Styles" or current_section == "V4 Styles":
                if stripped.upper().startswith("FORMAT:"):
                    format_fields = [f.strip() for f in stripped[7:].split(",")]
                elif stripped.upper().startswith("STYLE:"):
                    vals = [v.strip() for v in stripped[6:].split(",")]
                    if format_fields:
                        style = dict(zip(format_fields, vals))
                    else:
                        fields = ["Name", "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour", "OutlineColour", "BackColour", "Bold", "Italic", "Underline", "StrikeOut", "ScaleX", "ScaleY", "Spacing", "Angle", "BorderStyle", "Outline", "Shadow", "Alignment", "MarginL", "MarginR", "MarginV", "Encoding"]
                        style = dict(zip(fields, vals + [""] * (len(fields) - len(vals))))
                    name = style.get("Name", "Default")
                    styles[name] = style
            elif current_section == "Events":
                if stripped.upper().startswith("FORMAT:"):
                    format_fields = [f.strip() for f in stripped[7:].split(",")]
                elif stripped.upper().startswith("DIALOGUE:") or stripped.upper().startswith("COMMENT:"):
                    is_comment = stripped.upper().startswith("COMMENT:")
                    prefix_len = 9 if stripped.upper().startswith("DIALOGUE:") else 8
                    vals = [v.strip() for v in stripped[prefix_len:].split(",", 9)]
                    if len(vals) < 10:
                        continue
                    if not format_fields:
                        fields = ["Layer", "Start", "End", "Style", "Name", "MarginL", "MarginR", "MarginV", "Effect", "Text"]
                    else:
                        fields = format_fields
                    d = dict(zip(fields, vals))
                    blocks.append({
                        "index": str(len(blocks) + 1),
                        "timing": f"{d.get('Start', '0:00:00.00')} --> {d.get('End', '0:00:00.00')}",
                        "text": d.get("Text", ""),
                        "layer": d.get("Layer", "0"),
                        "style": d.get("Style", "Default"),
                        "actor": d.get("Name", ""),
                        "margin_l": d.get("MarginL", "0"),
                        "margin_r": d.get("MarginR", "0"),
                        "margin_v": d.get("MarginV", "0"),
                        "effect": d.get("Effect", ""),
                        "comment": is_comment,
                    })
        return blocks, styles

    @staticmethod
    def blocks_to_ass(blocks, styles=None):
        if styles is None:
            styles = {"Default": AssParser.default_style()}
        lines = []
        lines.append("[Script Info]")
        lines.append("Title: Subtitle Extractor")
        lines.append("ScriptType: v4.00+")
        lines.append("WrapStyle: 0")
        lines.append("ScaledBorderAndShadow: yes")
        lines.append("")
        lines.append("[V4+ Styles]")
        sf = ["Name", "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour", "OutlineColour", "BackColour", "Bold", "Italic", "Underline", "StrikeOut", "ScaleX", "ScaleY", "Spacing", "Angle", "BorderStyle", "Outline", "Shadow", "Alignment", "MarginL", "MarginR", "MarginV", "Encoding"]
        lines.append("Format: " + ", ".join(sf))
        for s in styles.values():
            row = [str(s.get(f.lower(), AssParser.default_style().get(f.lower(), ""))) for f in sf]
            lines.append("Style: " + ",".join(row))
        lines.append("")
        lines.append("[Events]")
        ef = ["Layer", "Start", "End", "Style", "Name", "MarginL", "MarginR", "MarginV", "Effect", "Text"]
        lines.append("Format: " + ", ".join(ef))
        for b in blocks:
            raw_text = b.get("text", "")
            raw_text = raw_text.replace("\n", "\\N")
            layer = b.get("layer", "0")
            style = b.get("style", "Default")
            actor = b.get("actor", "")
            ml = b.get("margin_l", "0")
            mr = b.get("margin_r", "0")
            mv = b.get("margin_v", "0")
            effect = b.get("effect", "")
            timing = b.get("timing", "0:00:00.00 --> 0:00:00.00")
            parts = timing.split("-->")
            start = parts[0].strip().replace(",", ".")
            end = parts[1].strip().replace(",", ".")
            prefix = "Comment" if b.get("comment") else "Dialogue"
            lines.append(f"{prefix}: {layer},{start},{end},{style},{actor},{ml},{mr},{mv},{effect},{raw_text}")
        return "\n".join(lines)

    @staticmethod
    def strip_tags(text):
        return re.sub(r"\{[^}]*\}", "", text)

    # ---- ASS Dialogue Block Parsing (Aegisub-style) ----

    @staticmethod
    def parse_dialogue_blocks(text):
        blocks = []
        pos = 0
        for part in re.split(r'(\{[^}]*\})', text):
            if not part:
                continue
            if part.startswith('{') and part.endswith('}'):
                blocks.append({"type": "override", "text": part, "start": pos, "end": pos + len(part)})
            else:
                blocks.append({"type": "plain", "text": part, "start": pos, "end": pos + len(part)})
            pos += len(part)
        return blocks

    @staticmethod
    def block_at_pos(blocks, pos):
        for i, block in enumerate(blocks):
            if block["start"] <= pos < block["end"]:
                return i, block
        if blocks and pos == blocks[-1]["end"]:
            return len(blocks) - 1, blocks[-1]
        return None, None

    @staticmethod
    def normalize_pos(text, raw_pos):
        plain_pos = 0
        in_block = False
        for i in range(min(raw_pos, len(text))):
            if text[i] == '{':
                in_block = True
            elif text[i] == '}':
                in_block = False
            elif not in_block:
                plain_pos += 1
        return plain_pos

    @staticmethod
    def _find_last_tag_in_str(text, tag_name):
        tag_esc = re.escape(tag_name)
        for pat_suffix in [r'(\d+(?:\.\d+)?)', r'(&H[0-9A-Fa-f]+&)', r'([^}\\]+?)(?=\\|\}|$)']:
            matches = list(re.finditer(tag_esc + pat_suffix, text))
            if matches:
                return matches[-1].group(1)
        return None

    @staticmethod
    def _remove_single_tag(inner, tag_name):
        tag_esc = re.escape(tag_name)
        for pat in [tag_esc + r'\d+(?:\.\d+)?',
                     tag_esc + r'&H[0-9A-Fa-f]+&',
                     tag_esc + r'[^}\\]+?(?=\\|\}|$)']:
            inner = re.sub(pat, '', inner)
        return inner

    @staticmethod
    def get_effective_at_pos(full_text, tag_name, style, raw_pos):
        blocks = AssParser.parse_dialogue_blocks(full_text)
        block_idx, block = AssParser.block_at_pos(blocks, raw_pos)

        if block is None:
            style_val = AssParser.get_style_tag_value(style, tag_name)
            return style_val not in (None, "0", "", False, 0)

        for i in range(block_idx, -1, -1):
            if blocks[i]["type"] == "override":
                val = AssParser._find_last_tag_in_str(blocks[i]["text"], tag_name)
                if val is not None:
                    return val not in ("0", "", False, 0)

        style_val = AssParser.get_style_tag_value(style, tag_name)
        return style_val not in (None, "0", "", False, 0)

    @staticmethod
    def get_style_tag_value(style, tag_name):
        tag_map = {
            "\\b": "bold", "\\i": "italic", "\\u": "underline", "\\s": "strikeout",
            "\\fn": "fontname", "\\fs": "fontsize",
            "\\c": "primary_color", "\\2c": "secondary_color",
            "\\3c": "outline_color", "\\4c": "shadow_color",
        }
        key = tag_map.get(tag_name)
        if key and style and key in style:
            return str(style[key])
        return None

    @staticmethod
    def insert_tag_at_pos(full_text, tag_name, tag_value, raw_pos):
        tag_str = tag_name + tag_value
        blocks = AssParser.parse_dialogue_blocks(full_text)
        block_idx, block = AssParser.block_at_pos(blocks, raw_pos)

        if block is None:
            return full_text, 0

        if block["type"] == "override":
            inner = block["text"][1:-1]
            inner = AssParser._remove_single_tag(inner, tag_name)
            if inner:
                inner += tag_str
            else:
                inner = tag_str
            new_block = "{" + inner + "}"
            result = full_text[:block["start"]] + new_block + full_text[block["end"]:]
            shift = len(result) - len(full_text)
            return result, shift
        else:
            result = full_text[:raw_pos] + "{" + tag_str + "}" + full_text[raw_pos:]
            shift = len(tag_str) + 2
            return result, shift

    @staticmethod
    def toggle_binary_tag(full_text, tag_name, style, raw_pos=0):
        current_is_on = AssParser.get_effective_at_pos(full_text, tag_name, style, raw_pos)
        new_val = "0" if current_is_on else "1"
        result, _ = AssParser.insert_tag_at_pos(full_text, tag_name, new_val, raw_pos)
        return result

    @staticmethod
    def toggle_with_selection(full_text, tag_name, style, sel_start, sel_end):
        current_is_on = AssParser.get_effective_at_pos(full_text, tag_name, style, sel_start)
        open_val = "0" if current_is_on else "1"
        close_val = "1" if current_is_on else "0"

        result, shift = AssParser.insert_tag_at_pos(full_text, tag_name, open_val, sel_start)
        result, _ = AssParser.insert_tag_at_pos(result, tag_name, close_val, sel_end + shift)
        return result

    @staticmethod
    def get_closing_tag(tag_str):
        pairs = {
            "{\\b1}": "{\\b0}", "{\\i1}": "{\\i0}", "{\\u1}": "{\\u0}", "{\\s1}": "{\\s0}",
            "{\\b0}": "{\\b1}", "{\\i0}": "{\\i1}", "{\\u0}": "{\\u1}", "{\\s0}": "{\\s1}",
            "{\\fnArial}": "", "{\\fnSimHei}": "",
            "{\\c&HFFFFFF&}": "", "{\\2c&HFFFFFF&}": "",
        }
        return pairs.get(tag_str, "")


class TranslationError(Exception):
    pass


class DeepLTranslator:
    def __init__(self, api_key):
        self.api_key = api_key

    def translate(self, text, target_lang, source_lang=None):
        import requests
        url = "https://api-free.deepl.com/v2/translate"
        params = {"auth_key": self.api_key, "text": text, "target_lang": target_lang}
        if source_lang and source_lang != "auto":
            params["source_lang"] = source_lang

        resp = requests.post(url, data=params, timeout=60)
        if resp.status_code == 403:
            raise TranslationError("DeepL: invalid API key or insufficient quota (403). Get a free key at https://deepl.com/pro-api")
        if resp.status_code == 429:
            raise TranslationError("DeepL: too many requests — quota exceeded (429). Wait or upgrade your plan.")
        resp.raise_for_status()
        return resp.json()["translations"][0]["text"]


class OpenAITranslator:
    def __init__(self, api_key):
        self.api_key = api_key

    def translate(self, text, target_lang, source_lang=None):
        from openai import OpenAI, APIError, RateLimitError, AuthenticationError
        client = OpenAI(api_key=self.api_key)
        src = "the source language" if source_lang == "auto" or not source_lang else DEEPL_LANGUAGES.get(source_lang, source_lang)
        tgt = DEEPL_LANGUAGES.get(target_lang, target_lang)

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Translate the following subtitle text from {src} to {tgt}. Preserve all line breaks and formatting. Return only the translation, no explanations."},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
                timeout=60,
            )
            return resp.choices[0].message.content.strip()
        except AuthenticationError:
            raise TranslationError("OpenAI: invalid API key. Check your key at https://platform.openai.com/api-keys")
        except RateLimitError:
            raise TranslationError("OpenAI: rate limit or quota exceeded. Check your usage at https://platform.openai.com/account/usage")


class GoogleTranslator:
    SEP = "|||SEP|||"

    def __init__(self, api_key):
        self.api_key = api_key

    def translate_batch(self, texts, target_lang, source_lang=None):
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        src = "the source language" if source_lang == "auto" or not source_lang else DEEPL_LANGUAGES.get(source_lang, source_lang)
        tgt = DEEPL_LANGUAGES.get(target_lang, target_lang)

        joined = f"\n{self.SEP}\n".join(texts)
        prompt = (
            f"Translate each text below from {src} to {tgt}. "
            f"Return them in the SAME ORDER, separated by exactly '{self.SEP}'. "
            f"Preserve all internal line breaks. Return ONLY the translations, nothing else.\n\n"
            f"{joined}"
        )

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        }

        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code == 403:
            raise TranslationError("Google: invalid API key. Get one at https://aistudio.google.com/apikey")
        if resp.status_code == 429:
            raise TranslationError("Google: rate limit (60 req/min free tier). Try again in a moment.")
        resp.raise_for_status()

        data = resp.json()
        try:
            raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
            raise TranslationError(f"Google: request blocked ({reason}). Try a different source language.")

        parts = raw.split(self.SEP)
        if len(parts) != len(texts):
            return texts
        return [p.strip() for p in parts]


LANG_TO_ISO = {
    "auto": "auto", "BG": "bg", "CS": "cs", "DA": "da", "DE": "de",
    "EL": "el", "EN": "en", "ES": "es", "ET": "et", "FI": "fi",
    "FR": "fr", "HU": "hu", "ID": "id", "IT": "it", "JA": "ja",
    "KO": "ko", "LT": "lt", "LV": "lv", "NB": "nb", "NL": "nl",
    "PL": "pl", "PT": "pt", "RO": "ro", "RU": "ru", "SK": "sk",
    "SL": "sl", "SV": "sv", "TR": "tr", "UK": "uk", "ZH": "zh",
}


class MyMemoryTranslator:
    """MyMemory API - free tier: 5000 chars/day without key, unlimited with free key (signup at mymemory.translated.net)."""

    def __init__(self, api_key=""):
        self.api_key = api_key

    def translate(self, text, target_lang, source_lang=None):
        import requests
        from urllib.parse import quote
        src = LANG_TO_ISO.get(source_lang, "zh") if source_lang and source_lang != "auto" else "zh"
        tgt = LANG_TO_ISO.get(target_lang, "pl")
        langpair = f"{src}|{tgt}"
        url = f"https://api.mymemory.translated.net/get?q={quote(text)}&langpair={langpair}"
        if self.api_key:
            url += f"&key={self.api_key}"

        resp = requests.get(url, timeout=30)
        if resp.status_code == 429:
            raise TranslationError("MyMemory: rate limited. Free tier: 5000 chars/day without key. Sign up for free key at mymemory.translated.net")
        resp.raise_for_status()
        data = resp.json()
        if data.get("responseStatus") != 200:
            raise TranslationError(f"MyMemory: {data.get('responseDetails', 'unknown error')}")
        return data["responseData"]["translatedText"]


class HuggingFaceTranslator:
    """Hugging Face Inference API - free with token (sign up at huggingface.co)."""

    def __init__(self, api_key):
        self.api_key = api_key

    def _model_for(self, source_lang, target_lang):
        src = LANG_TO_ISO.get(source_lang, "")
        tgt = LANG_TO_ISO.get(target_lang, "")
        if src == "zh" and tgt == "pl":
            return "Helsinki-NLP/opus-mt-zh-pl"
        return f"Helsinki-NLP/opus-mt-{src}-{tgt}"

    def translate(self, text, target_lang, source_lang=None):
        import requests
        src = LANG_TO_ISO.get(source_lang, "zh") if source_lang and source_lang != "auto" else "zh"
        tgt = LANG_TO_ISO.get(target_lang, "pl")
        model = self._model_for(src, tgt)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json={"inputs": text},
            timeout=120,
        )
        if resp.status_code == 403:
            raise TranslationError("HuggingFace: invalid token. Get a free one at https://huggingface.co/settings/tokens")
        if resp.status_code == 503:
            raise TranslationError(f"HuggingFace: model {model} is loading (cold start) — try again in ~30s.")
        resp.raise_for_status()

        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("translation_text", data[0].get("generated_text", text))
        return text


class LocalTranslator:
    def __init__(self):
        self._models = {}

    def _install_package(self, from_code, to_code):
        import argostranslate.package
        import argostranslate.translate
        try:
            return argostranslate.translate.get_translation_from_codes(from_code, to_code)
        except Exception:
            pass
        available = argostranslate.package.get_available_packages()
        pkg = next((p for p in available if p.from_code == from_code and p.to_code == to_code), None)
        if pkg is None:
            return None
        pkg.download()
        pkg.install()
        return argostranslate.translate.get_translation_from_codes(from_code, to_code)

    def _ensure_models(self, source_lang, target_lang):
        import argostranslate.package
        src = LANG_TO_ISO.get(source_lang, "zh")
        tgt = LANG_TO_ISO.get(target_lang, "pl")

        key = f"{src}->{tgt}"
        if key in self._models:
            return

        argostranslate.package.update_package_index()
        model = self._install_package(src, tgt)
        if model is None:
            if src == "en" or tgt == "en":
                raise TranslationError(f"No local model for {source_lang} -> {target_lang}")
            en_src = self._install_package(src, "en")
            en_tgt = self._install_package("en", tgt)
            if en_src is None:
                raise TranslationError(f"No model for {source_lang} -> English (pivot required)")
            if en_tgt is None:
                raise TranslationError(f"No model for English -> {target_lang} (pivot required)")
            self._models[key] = (en_src, en_tgt)
        else:
            self._models[key] = model

    def translate(self, text, target_lang, source_lang=None):
        self._ensure_models(source_lang or "auto", target_lang)
        key = f"{LANG_TO_ISO.get(source_lang or 'auto', 'zh')}->{LANG_TO_ISO.get(target_lang, 'pl')}"
        model = self._models[key]
        if isinstance(model, tuple):
            # Pivot through English
            intermediate = model[0].translate(text)
            return model[1].translate(intermediate)
        return model.translate(text)


def get_translator(service, config):
    key_map = {
        "openai": "openai_key",
        "google": "google_key",
        "deepl": "deepl_key",
        "libre": "libre_key",
        "huggingface": "hf_key",
    }
    key = config.get(key_map.get(service, ""), "")
    if service == "dictionary":
        return DictionaryTranslator()
    if service == "dictionary2":
        return Dictionary2Translator()
    if service == "dictionary3":
        return Dictionary3Translator()
    if service == "baidu":
        return BaiduTranslator(key)
    if service == "openai":
        return OpenAITranslator(key)
    if service == "google":
        return GoogleTranslator(key)
    if service == "libre":
        return MyMemoryTranslator(key)
    if service == "huggingface":
        return HuggingFaceTranslator(key)
    return DeepLTranslator(key)


class DictionaryTranslator:
    """Word-by-word gloss using CC-CEDICT (no API key needed)."""

    def __init__(self):
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        import backend.dictionary as d
        d._ensure_loaded()
        self._entries = d._cedict_entries
        self._index = d._cedict_index
        self._max_word_len = max((len(e["simp"]) for e in self._entries), default=1)
        self._loaded = True

    def _segment(self, text):
        """Greedy longest-match segmentation using CEDICT."""
        self._ensure_loaded()
        words = []
        i = 0
        while i < len(text):
            # Group consecutive non-CJK characters
            if not re.match(r"[\u4e00-\u9fff\u3400-\u4dbf]", text[i]):
                j = i
                while j < len(text) and not re.match(r"[\u4e00-\u9fff\u3400-\u4dbf]", text[j]):
                    j += 1
                words.append(text[i:j])
                i = j
                continue
            matched = False
            for end in range(min(i + self._max_word_len, len(text)), i, -1):
                chunk = text[i:end]
                if chunk in self._index:
                    words.append(chunk)
                    i = end
                    matched = True
                    break
            if not matched:
                words.append(text[i])
                i += 1
        return words

    def translate(self, text, target_lang=None, source_lang=None):
        self._ensure_loaded()
        # Only process Chinese text
        has_chinese = bool(re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))
        if not has_chinese:
            return text
        words = self._segment(text)
        gloss_parts = []
        for w in words:
            if not re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", w):
                gloss_parts.append(w)
                continue
            entries = self._index.get(w, [])
            if entries:
                e = entries[0]
                pinyin = e.get("pinyin", "")
                defs = e.get("defs", [])
                if defs:
                    gloss_parts.append(f"{w} ({pinyin}) [{defs[0]}]")
                else:
                    gloss_parts.append(f"{w} ({pinyin})")
            else:
                gloss_parts.append(w)
        gloss = "  ".join(gloss_parts)
        return f"{text}\nGloss: {gloss}"


class Dictionary2Translator:
    """Full-definition gloss: original text + all definitions for each segment."""

    def __init__(self):
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        import backend.dictionary as d
        d._ensure_loaded()
        self._entries = d._cedict_entries
        self._index = d._cedict_index
        self._max_word_len = max((len(e["simp"]) for e in self._entries), default=1)
        self._loaded = True

    def _segment(self, text):
        self._ensure_loaded()
        words = []
        i = 0
        while i < len(text):
            if not re.match(r"[\u4e00-\u9fff\u3400-\u4dbf]", text[i]):
                j = i
                while j < len(text) and not re.match(r"[\u4e00-\u9fff\u3400-\u4dbf]", text[j]):
                    j += 1
                words.append(text[i:j])
                i = j
                continue
            matched = False
            for end in range(min(i + self._max_word_len, len(text)), i, -1):
                chunk = text[i:end]
                if chunk in self._index:
                    words.append(chunk)
                    i = end
                    matched = True
                    break
            if not matched:
                words.append(text[i])
                i += 1
        return words

    def translate(self, text, target_lang=None, source_lang=None):
        self._ensure_loaded()
        has_chinese = bool(re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))
        if not has_chinese:
            return text
        words = self._segment(text)
        lines = [text, ""]
        for w in words:
            if not re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", w):
                lines.append(w)
                continue
            entries = self._index.get(w, [])
            if entries:
                seen = set()
                for e in entries:
                    pinyin = e.get("pinyin", "")
                    key = (w, pinyin)
                    if key in seen:
                        continue
                    seen.add(key)
                    defs = e.get("defs", [])
                    if defs:
                        lines.append(f"  {w} ({pinyin})")
                        for d in defs:
                            lines.append(f"    - {d}")
                    else:
                        lines.append(f"  {w} ({pinyin})")
            else:
                lines.append(f"  {w} — (no entry)")
        return "\n".join(lines)


class Dictionary3Translator:
    """Full-definition gloss + rough English translation."""

    def __init__(self):
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        import backend.dictionary as d
        d._ensure_loaded()
        self._entries = d._cedict_entries
        self._index = d._cedict_index
        self._max_word_len = max((len(e["simp"]) for e in self._entries), default=1)
        self._loaded = True

    def _segment(self, text):
        self._ensure_loaded()
        words = []
        i = 0
        while i < len(text):
            if not re.match(r"[\u4e00-\u9fff\u3400-\u4dbf]", text[i]):
                j = i
                while j < len(text) and not re.match(r"[\u4e00-\u9fff\u3400-\u4dbf]", text[j]):
                    j += 1
                words.append(text[i:j])
                i = j
                continue
            matched = False
            for end in range(min(i + self._max_word_len, len(text)), i, -1):
                chunk = text[i:end]
                if chunk in self._index:
                    words.append(chunk)
                    i = end
                    matched = True
                    break
            if not matched:
                words.append(text[i])
                i += 1
        return words

    def translate(self, text, target_lang=None, source_lang=None):
        self._ensure_loaded()
        has_chinese = bool(re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))
        if not has_chinese:
            return text
        words = self._segment(text)
        # Build rough English translation from first definitions
        trans_parts = []
        for w in words:
            if not re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", w):
                trans_parts.append(w)
                continue
            entries = self._index.get(w, [])
            if entries:
                defs = entries[0].get("defs", [])
                trans_parts.append(defs[0] if defs else w)
            else:
                trans_parts.append(w)
        rough_trans = " ".join(trans_parts)
        lines = [text, rough_trans, ""]
        for w in words:
            if not re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", w):
                lines.append(w)
                continue
            entries = self._index.get(w, [])
            if entries:
                seen = set()
                for e in entries:
                    pinyin = e.get("pinyin", "")
                    key = (w, pinyin)
                    if key in seen:
                        continue
                    seen.add(key)
                    defs = e.get("defs", [])
                    if defs:
                        lines.append(f"  {w} ({pinyin})")
                        for d in defs:
                            lines.append(f"    - {d}")
                    else:
                        lines.append(f"  {w} ({pinyin})")
            else:
                lines.append(f"  {w} — (no entry)")
        return "\n".join(lines)


class BaiduTranslator:
    """Word-by-word gloss + example phrases from Baidu fanyi public API."""

    def __init__(self, api_key=""):
        self._api_key = api_key  # unused, public API
        self._loaded = False
        self._cache = {}

    def _ensure_loaded(self):
        if self._loaded:
            return
        import backend.dictionary as d
        d._ensure_loaded()
        self._entries = d._cedict_entries
        self._index = d._cedict_index
        self._max_word_len = max((len(e["simp"]) for e in self._entries), default=1)
        self._loaded = True

    def _segment(self, text):
        self._ensure_loaded()
        words = []
        i = 0
        while i < len(text):
            if not re.match(r"[\u4e00-\u9fff\u3400-\u4dbf]", text[i]):
                j = i
                while j < len(text) and not re.match(r"[\u4e00-\u9fff\u3400-\u4dbf]", text[j]):
                    j += 1
                words.append(text[i:j])
                i = j
                continue
            matched = False
            for end in range(min(i + self._max_word_len, len(text)), i, -1):
                chunk = text[i:end]
                if chunk in self._index:
                    words.append(chunk)
                    i = end
                    matched = True
                    break
            if not matched:
                words.append(text[i])
                i += 1
        return words

    def _baidu_lookup(self, word):
        if word in self._cache:
            return self._cache[word]
        import requests
        try:
            r = requests.post("https://fanyi.baidu.com/sug", data={"kw": word}, timeout=10)
            data = r.json()
            results = data.get("data", [])
            self._cache[word] = results
            return results
        except Exception:
            self._cache[word] = []
            return []

    def translate(self, text, target_lang=None, source_lang=None):
        self._ensure_loaded()
        has_chinese = bool(re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))
        if not has_chinese:
            return text
        words = self._segment(text)
        lines = [text, ""]
        for w in words:
            if not re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", w):
                lines.append(w)
                continue
            # CEDICT entry
            entries = self._index.get(w, [])
            pinyin = entries[0].get("pinyin", "") if entries else ""
            first_def = entries[0].get("defs", [""])[0] if entries else ""
            lines.append(f"  {w} ({pinyin}) — {first_def}")
            # Baidu phrases
            phrases = self._baidu_lookup(w)
            if phrases:
                has_extra = False
                for p in phrases:
                    pw = p.get("k", "")
                    pv = p.get("v", "")
                    if pw == w:
                        continue
                    if not has_extra:
                        lines.append("    Phrases:")
                        has_extra = True
                    lines.append(f"      {pw} — {pv}")
        return "\n".join(lines)


def translate_srt_file(srt_path, output_path, source_lang, target_lang, service, api_key, progress_callback=None, cancel_flag=None):
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = SrtParser.parse_blocks(content)
    total = len(blocks)

    if service == "dictionary":
        translator = DictionaryTranslator()
    elif service == "dictionary2":
        translator = Dictionary2Translator()
    elif service == "dictionary3":
        translator = Dictionary3Translator()
    elif service == "baidu":
        translator = BaiduTranslator(api_key)
    elif service == "openai":
        translator = OpenAITranslator(api_key)
    elif service == "google":
        translator = GoogleTranslator(api_key)
    elif service == "local":
        translator = LocalTranslator()
    elif service == "libre":
        translator = MyMemoryTranslator(api_key)
    elif service == "huggingface":
        translator = HuggingFaceTranslator(api_key)
    else:
        translator = DeepLTranslator(api_key)

    if service == "google":
        non_empty = [(i, b) for i, b in enumerate(blocks) if b["text"].strip()]
        if not non_empty:
            return output_path

        idxs, valid_blocks = zip(*non_empty)
        texts = [b["text"].strip() for b in valid_blocks]

        try:
            translated_texts = translator.translate_batch(list(texts), target_lang, source_lang)
        except TranslationError:
            raise

        for idx, new_text in zip(idxs, translated_texts):
            blocks[idx]["text"] = new_text
            if progress_callback:
                progress_callback((idx + 1) / total * 100)

        if cancel_flag and cancel_flag():
            return None
    else:
        for i, block in enumerate(blocks):
            if cancel_flag and cancel_flag():
                return None

            text = block["text"].strip()
            if not text:
                if progress_callback:
                    progress_callback((i + 1) / total * 100)
                continue

            try:
                translated_text = translator.translate(text, target_lang, source_lang)
                block["text"] = translated_text
            except TranslationError:
                raise
            except Exception as e:
                print(f"[TRANSLATE ERROR] block {block['index']}: {e}")
                block["text"] = text

            if progress_callback:
                progress_callback((i + 1) / total * 100)

    result = SrtParser.blocks_to_srt(blocks)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    return output_path
