"""Lập data/processed/index.csv: mỗi video dùng được 1 dòng, kèm cách chia train/val/test.

Chạy (từ thư mục gốc repo):
  python -m src.dataset.index_kaggle             giữ cách chia đã có trong index.csv
  python -m src.dataset.index_kaggle --resplit   chia lại từ đầu
"""
import sys
from collections import Counter
from unicodedata import normalize

import cv2
import numpy as np
import pandas as pd

from src.core.paths import EXCLUDE_FILE, INDEX_FILE, PROCESSED_DIR, RAW_DIR, VIDEO_DIR, VOCAB_FILE, WORDS_FILE

MIN_VIDEOS = 20         # từ còn dưới 20 video thì bỏ cả từ
SOURCE_B_FRAMES = 30    # nguồn B: không quá 30 frame. Nguồn A: dài hơn

def nfc(s):
    """Đưa chuỗi về dạng Unicode NFC để so sánh được với nhau."""
    return normalize("NFC", s.strip())

def main_words():
    """Các từ trong danh sách chính của words.txt (phần đứng trước dòng '# Dự phòng')."""
    text = WORDS_FILE.read_text(encoding="utf-8").split("# Dự phòng")[0]
    return [nfc(line) for line in text.splitlines() if line.strip() and not line.startswith("#")]

def load_exclude():
    """Tập video bị loại, mỗi phần tử dạng 'train/Bố/123456'."""
    lines = EXCLUDE_FILE.read_text(encoding="utf-8").splitlines()
    return {nfc(line) for line in lines if line.strip() and not line.startswith("#")}

def video_header(key):
    """(fps, rộng, cao) đọc từ header của file mp4, không giải mã hình."""
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{key}.mp4"))
    fps, width, height = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Không đọc được video {key}.mp4 (thiếu file?)")
    return round(fps, 2), int(width), int(height)

def list_videos():
    """-> (bảng video dùng được, các từ bị bỏ kèm số video còn lại)."""
    files = sorted(RAW_DIR.glob("*/*/*.npy"))                     # raw/<train|test>/<từ>/<id>.npy
    df = pd.DataFrame({"key": [f.relative_to(RAW_DIR).with_suffix("").as_posix() for f in files],
                       "word": [nfc(f.parent.name) for f in files],
                       "orig_split": [f.parts[-3] for f in files]})
    df = df[df.word.isin(main_words())]                           # 1. chỉ lấy danh sách chính
    df = df[~df.key.map(nfc).isin(load_exclude())]                # 2. trừ video bị loại
    counts = df.word.value_counts()
    dropped = counts[counts < MIN_VIDEOS]                         # 3. bỏ từ còn quá ít video
    df = df[~df.word.isin(dropped.index)].reset_index(drop=True)

    words = sorted(df.word.unique())
    df["label"] = df.word.map({w: i for i, w in enumerate(words)})
    df = df.sort_values("label", kind="stable").reset_index(drop=True)   # xếp theo từ; trong mỗi từ giữ thứ tự file
    df["signer"] = ""                                             # Kaggle không có thông tin người ký
    df["frames"] = [len(np.load(RAW_DIR / f"{k}.npy", mmap_mode="r")) for k in df.key]
    df["source"] = np.where(df.frames > SOURCE_B_FRAMES, "A", "B")
    header = pd.DataFrame([video_header(k) for k in df.key], columns=["fps", "width", "height"])
    return df.join(header), dropped

SPLIT_SEED = 42
HELD_OUT_RATE = 0.15    # val 15%, test 15%
MIN_HELD_OUT = 2        # mỗi từ có ít nhất 2 video val và 2 video test
FIELDS = ["key", "word", "label", "signer", "source", "orig_split", "split", "frames", "fps", "width", "height"]


def held_out_count(n):
    """Số video val (cũng là số video test) của 1 từ có n video."""
    return max(MIN_HELD_OUT, round(HELD_OUT_RATE * n))

def split_word(keys_by_source, rng):
    """Chia các video của 1 từ. keys_by_source = {"A": [key, ...], "B": [...]} -> {key: "train" | "val" | "test"}."""
    queue = []
    for source in sorted(keys_by_source):
        keys = list(keys_by_source[source])
        rng.shuffle(keys)                                          # xáo riêng từng nguồn
        queue += [((k + 0.5) / len(keys), source, key) for k, key in enumerate(keys)]
    queue.sort()                                                   # rải xen kẽ: sắp theo vị trí 0..1
    n = held_out_count(len(queue))
    return {key: "test" if i < n else "val" if i < 2 * n else "train" for i, (_, _, key) in enumerate(queue)}

def fill_new(keys, old):
    """Từ đã có cách chia cũ: video cũ giữ nguyên tập, video mới vào tập đang thiếu nhiều nhất."""
    n = held_out_count(len(keys))
    target = {"val": n, "test": n, "train": len(keys) - 2 * n}
    split = {k: old[k] for k in keys if k in old}
    have = Counter(split.values())
    for key in sorted(set(keys) - set(split)):
        s = max(target, key=lambda s: (target[s] - have[s], s == "train"))   # thiếu nhiều nhất; bằng nhau thì train
        split[key] = s
        have[s] += 1
    return split

def build_index(resplit=False):
    df, dropped = list_videos()
    old = {}
    if INDEX_FILE.exists() and not resplit:                        # đóng băng: đọc lại cách chia đã có
        old = pd.read_csv(INDEX_FILE, encoding="utf-8-sig").set_index("key")["split"].to_dict()

    rng = np.random.default_rng(SPLIT_SEED)
    split = {}
    for word in sorted(df.word.unique()):                          # thứ tự cố định -> cùng seed cho cùng kết quả
        rows = df[df.word == word]
        keys = rows.key.tolist()
        if any(k in old for k in keys):
            split.update(fill_new(keys, old))
        else:
            split.update(split_word({s: rows.key[rows.source == s].tolist() for s in rows.source.unique()}, rng))
    df["split"] = df.key.map(split)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df[FIELDS].to_csv(INDEX_FILE, index=False, encoding="utf-8-sig")
    pd.Series(sorted(df.word.unique())).to_csv(VOCAB_FILE, header=False, encoding="utf-8-sig")

    print(f"{len(df)} video, {df.word.nunique()} từ -> {INDEX_FILE}")
    if old:
        print(f"Giữ cách chia cũ: {sum(k not in old for k in df.key)} video mới, "
              f"{len(set(old) - set(df.key))} video đã bị bớt (dùng --resplit để chia lại từ đầu)")
    order = ["train", "val", "test"]
    print(pd.crosstab(df.word, df.split, margins=True, margins_name="Tổng")[order + ["Tổng"]].to_string())
    print("\nNguồn trong từng tập:")
    print(pd.crosstab(df.source, df.split)[order].to_string())
    print("\nTừ bị bỏ (dưới 20 video):", dropped.to_dict())


if __name__ == "__main__":
    build_index(resplit="--resplit" in sys.argv)
