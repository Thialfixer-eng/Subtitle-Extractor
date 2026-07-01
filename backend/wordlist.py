import os
import json

WORDLISTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wordlists.json")


def load_lists():
    if not os.path.exists(WORDLISTS_FILE):
        return []
    try:
        with open(WORDLISTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else data.get("lists", [])
    except Exception:
        return []


def save_lists(lists):
    with open(WORDLISTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"lists": lists}, f, indent=2, ensure_ascii=False)


def add_list(name):
    lists = load_lists()
    if any(lst["name"] == name for lst in lists):
        return False
    lists.append({"name": name, "words": []})
    save_lists(lists)
    return True


def delete_list(name):
    lists = load_lists()
    lists = [lst for lst in lists if lst["name"] != name]
    save_lists(lists)
    return lists


def rename_list(old_name, new_name):
    lists = load_lists()
    for lst in lists:
        if lst["name"] == old_name:
            lst["name"] = new_name
            save_lists(lists)
            return True
    return False


def add_word(list_name, word_entry):
    lists = load_lists()
    for lst in lists:
        if lst["name"] == list_name:
            entry = {
                "simp": word_entry.get("simp", ""),
                "trad": word_entry.get("trad", ""),
                "pinyin": word_entry.get("pinyin", ""),
                "defs": word_entry.get("defs", []),
                "hsk": word_entry.get("hsk", ""),
                "strokes": word_entry.get("strokes", ""),
            }
            lst["words"].append(entry)
            save_lists(lists)
            return True
    return False


def remove_word(list_name, word_idx):
    lists = load_lists()
    for lst in lists:
        if lst["name"] == list_name:
            if 0 <= word_idx < len(lst["words"]):
                lst["words"].pop(word_idx)
                save_lists(lists)
                return True
    return False


def export_anki(list_name, filepath):
    lists = load_lists()
    for lst in lists:
        if lst["name"] == list_name:
            lines = []
            for w in lst["words"]:
                simp = w.get("simp", "")
                trad = w.get("trad", "")
                pinyin = w.get("pinyin", "")
                defs = " / ".join(w.get("defs", []))
                hsk = w.get("hsk", "")
                strokes = w.get("strokes", "")
                lines.append(f"{simp}\t{trad}\t{pinyin}\t{defs}\tHSK:{hsk}\tStrokes:{strokes}")
            header = "Simplified\tTraditional\tPinyin\tDefinitions\tHSK\tStrokes"
            with open(filepath, "w", encoding="utf-8-sig") as f:
                f.write(header + "\n")
                for line in lines:
                    f.write(line + "\n")
            return len(lines)
    return -1
