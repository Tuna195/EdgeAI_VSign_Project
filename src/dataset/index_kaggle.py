"""Adapter cho bộ Kaggle VSL v2: tạo data/processed/index.csv (mỗi video dùng được 1 dòng) và chia train/val/test mới.

Chạy: python -m src.dataset.index_kaggle             (giữ cách chia đã có trong index.csv, chỉ thêm/bớt video)
      python -m src.dataset.index_kaggle --resplit   (xáo và chia lại từ đầu: kết quả train cũ không còn so được)

Mọi bước sau (build_sequences, train) chỉ đọc index.csv. Sau này Multi-VSL có adapter riêng ghi cùng định dạng.

Cách chia: trong TỪNG TỪ, val và test mỗi phần max(2, round(15% số video)), còn lại là train.
Video nguồn A (> 30 frame) và B (<= 30 frame) được chia theo tỉ lệ, để val/test có cả hai loại như train.
Split bị "đóng băng": chạy lại (vd. sau khi loại thêm video) thì video cũ giữ nguyên phía, video mới được xếp
vào phía đang thiếu nhất. Nếu không đóng băng, loại 1 video sẽ làm thứ tự xáo đổi hết -> kết quả cũ không so được.
Lưu ý: bộ Kaggle không có thông tin người ký -> cùng 1 người có thể ở cả train lẫn test (điểm test lạc quan).
"""
import argparse
import csv
import unicodedata
from collections import Counter, defaultdict

import cv2
import numpy as np

from src.core.paths import INDEX_FILE, PROCESSED_DIR, RAW_DIR, VIDEO_DIR, VOCAB_FILE
from src.core.preprocess import usable_videos

FIELDS = ["key", "word", "label", "signer", "source", "orig_split", "split", "frames", "fps", "width", "height"]
SPLIT_SEED = 42
VAL_RATE = TEST_RATE = 0.15
MIN_HELD_OUT = 2       # mỗi từ có ít nhất 2 video val và 2 video test
SOURCE_B_FRAMES = 30   # nguồn B: video <= 30 frame (ID ngẫu nhiên); nguồn A: > 30 frame (phông xanh/trắng)


def held_out_counts(n):
    """Số video (val, test) cho 1 từ có n video."""
    return max(MIN_HELD_OUT, round(VAL_RATE * n)), max(MIN_HELD_OUT, round(TEST_RATE * n))


def video_header(split, word, vid):
    """(fps, width, height) đọc từ header video, không giải mã frame nào."""
    cap = cv2.VideoCapture(str(VIDEO_DIR / split / word / f"{vid}.mp4"))
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Không đọc được kích thước video {split}/{word}/{vid} (thiếu file mp4?)")
    return round(fps, 2), w, h


def split_word(rows, rng):
    """Chia các video của 1 từ: gán r["split"] cho từng dòng.

    Mỗi nguồn được xáo riêng, rồi "rải" các nguồn xen kẽ theo tỉ lệ thành 1 hàng: phần đầu hàng -> test,
    tiếp theo -> val, còn lại -> train. Rải theo tỉ lệ nên nguồn nào cũng có mặt ở val/test đúng tỉ lệ
    (không làm tròn lên cho từng nguồn, nên tổng val/test của từ không bị phình ra).
    """
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    queue = []
    for src in sorted(by_source):
        group = by_source[src]
        rng.shuffle(group)
        # vị trí (0..1) của từng video trong hàng chung: rải đều theo nguồn
        queue += [((k + 0.5) / len(group), src, r) for k, r in enumerate(group)]
    queue.sort(key=lambda t: (t[0], t[1]))
    n_val, n_test = held_out_counts(len(rows))
    for k, (_, _, r) in enumerate(queue):
        r["split"] = "test" if k < n_test else ("val" if k < n_test + n_val else "train")


def fill_new(rows):
    """Video mới (chưa có split) được xếp vào phía đang thiếu nhiều nhất so với mục tiêu."""
    n_val, n_test = held_out_counts(len(rows))
    target = {"val": n_val, "test": n_test, "train": len(rows) - n_val - n_test}
    have = Counter(r["split"] for r in rows if r["split"])
    for r in sorted((r for r in rows if not r["split"]), key=lambda r: r["key"]):
        r["split"] = max(target, key=lambda s: (target[s] - have[s], s == "train"))
        have[r["split"]] += 1


def build_index(resplit=False):
    videos, dropped = usable_videos()
    words = sorted(videos)
    old = {}
    if INDEX_FILE.exists() and not resplit:
        with open(INDEX_FILE, encoding="utf-8-sig") as f:
            old = {r["key"]: r["split"] for r in csv.DictReader(f)}

    rows = []
    for label, word in enumerate(words):
        for key in videos[word]:
            orig_split, folder, vid = key.split("/")
            frames = len(np.load(RAW_DIR / orig_split / folder / f"{vid}.npy", mmap_mode="r"))
            fps, w, h = video_header(orig_split, folder, vid)
            rows.append({"key": key, "word": unicodedata.normalize("NFC", word), "label": label, "signer": "",
                         "source": "B" if frames <= SOURCE_B_FRAMES else "A", "orig_split": orig_split,
                         "split": old.get(key, ""), "frames": frames, "fps": fps, "width": w, "height": h})

    rng = np.random.default_rng(SPLIT_SEED)
    by_word = defaultdict(list)
    for r in rows:
        by_word[r["word"]].append(r)
    n_new = sum(not r["split"] for r in rows)
    for word in words:   # thứ tự cố định -> cùng seed cho cùng kết quả
        rs = by_word[word]
        if all(not r["split"] for r in rs):
            split_word(rs, rng)
        else:
            fill_new(rs)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(VOCAB_FILE, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([[i, w] for i, w in enumerate(words)])

    report(rows, words, n_new if old else None, len(set(old) - {r["key"] for r in rows}) if old else None)
    if dropped:
        print(f"Bỏ {len(dropped)} từ còn < 20 video:", ", ".join(dropped))


def report(rows, words, n_new, n_removed):
    print(f"{len(rows)} video, {len(words)} từ -> {INDEX_FILE}")
    if n_new is not None:
        print(f"Giữ cách chia cũ: thêm {n_new} video mới, bớt {n_removed} video (dùng --resplit để chia lại từ đầu)")
    print(f"\n{'Từ':16s} {'train':>5s} {'val':>4s} {'test':>4s}   nguồn A/B (train | val | test)")
    total = Counter()
    for word in words:
        rs = [r for r in rows if r["word"] == word]
        c = Counter(r["split"] for r in rs)
        src = " | ".join(f'{sum(r["split"] == s and r["source"] == "A" for r in rs)}/'
                         f'{sum(r["split"] == s and r["source"] == "B" for r in rs)}' for s in ("train", "val", "test"))
        print(f"{word:16s} {c['train']:5d} {c['val']:4d} {c['test']:4d}   {src}")
        total.update(c)
    print(f"{'Tổng':16s} {total['train']:5d} {total['val']:4d} {total['test']:4d}")
    sizes = Counter((r["source"], r["width"], r["height"]) for r in rows)
    print("\nKích thước video theo nguồn:", ", ".join(f"{s} {w}x{h}: {n}" for (s, w, h), n in sorted(sizes.items())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resplit", action="store_true", help="xáo và chia lại từ đầu")
    build_index(parser.parse_args().resplit)
