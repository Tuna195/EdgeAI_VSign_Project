"""index.csv + data/raw -> data/processed/sequences.npz: mỗi video 1 chuỗi (T, 225) độ dài thay đổi, chưa chuẩn hoá.

Chạy: python -m src.dataset.build_sequences   (sau index_kaggle)

Mỗi video qua prepare_sequence (src/core/features.py): sửa tỉ lệ khung hình, fill_gaps, bỏ frame trống 2 đầu.
Chưa lấy N frame, chưa chuẩn hoá: các bước đó làm lúc train (augmentation đổi tốc độ ký cần chuỗi đầy đủ,
và thử 16/20/30 frame hay chuẩn hoá a/b/c không phải chạy lại file này).

sequences.npz: keys (N,), offsets (N+1,), data (tổng T, 225). Chuỗi thứ i = data[offsets[i]:offsets[i+1]].
(Ghép nối tiếp thay vì lưu mảng 3 chiều: các video dài khác nhau, đệm 0 cho bằng nhau sẽ tốn chỗ và dễ nhầm.)
Video bị bỏ ghi vào data/reports/sequences_dropped.csv.
"""
import csv
from collections import Counter, defaultdict

import numpy as np

from src.core.features import prepare_sequence
from src.core.paths import INDEX_FILE, RAW_DIR, REPORT_DIR, SEQ_FILE

DROPPED_FILE = REPORT_DIR / "sequences_dropped.csv"


def load_sequences(path=SEQ_FILE):
    """-> {key: (T, 225) float32}"""
    z = np.load(path)
    data, off = z["data"], z["offsets"]
    return {str(k): data[off[i]:off[i + 1]] for i, k in enumerate(z["keys"])}


def main():
    with open(INDEX_FILE, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    blank_report = {}
    videos_csv = REPORT_DIR / "videos.csv"
    if videos_csv.exists():
        with open(videos_csv, encoding="utf-8-sig") as f:
            blank_report = {f'{r["split"]}/{r["word"]}/{r["id"]}': int(r["blank_frames"]) for r in csv.DictReader(f)}

    keys, seqs, dropped = [], [], []
    trimmed = {}   # key -> số frame trống đã cắt ở 2 đầu
    for r in rows:
        split, word, vid = r["key"].split("/")
        raw = np.load(RAW_DIR / split / word / f"{vid}.npy")
        seq, reason = prepare_sequence(raw, int(r["width"]), int(r["height"]))
        if seq is None:
            dropped.append({"key": r["key"], "word": r["word"], "split": r["split"], "reason": reason})
            continue
        keys.append(r["key"])
        seqs.append(seq)
        trimmed[r["key"]] = len(raw) - len(seq)

    offsets = np.concatenate([[0], np.cumsum([len(s) for s in seqs])]).astype(np.int64)
    np.savez_compressed(SEQ_FILE, keys=np.array(keys), offsets=offsets, data=np.concatenate(seqs))
    with open(DROPPED_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["key", "word", "split", "reason"])
        w.writeheader()
        w.writerows(dropped)

    # ---- báo cáo ----
    print(f"Giữ {len(keys)}/{len(rows)} video -> {SEQ_FILE}")
    if dropped:
        print(f"Bỏ {len(dropped)} video (chi tiết: {DROPPED_FILE}):")
        for (word, reason), n in sorted(Counter((d["word"], d["reason"]) for d in dropped).items()):
            print(f"  {word:16s} {n:3d}  {reason}")
    source = {r["key"]: r["source"] for r in rows}
    lengths = defaultdict(list)
    for k, s in zip(keys, seqs):
        lengths[source[k]].append(len(s))
    for src, ls in sorted(lengths.items()):
        print(f"Nguồn {src}: {len(ls)} chuỗi, độ dài min {min(ls)} / trung vị {int(np.median(ls))} / max {max(ls)} frame")
    n_cut = sum(v > 0 for v in trimmed.values())
    print(f"Cắt frame trống ở 2 đầu: {n_cut} video, tổng {sum(trimmed.values())} frame")
    if blank_report:
        both = [k for k in trimmed if k in blank_report]
        agree = sum((trimmed[k] > 0) == (blank_report[k] > 0) for k in both)
        print(f"  So với cột blank_frames (frame ảnh 1 màu) của videos.csv: khớp có/không ở {agree}/{len(both)} video; "
              f"tổng blank_frames = {sum(blank_report[k] for k in both)}")
        diff = [(k, trimmed[k], blank_report[k]) for k in both if trimmed[k] != blank_report[k]]
        for k, t, b in diff[:10]:
            print(f"    {k}: cắt {t}, blank_frames {b}")
        if len(diff) > 10:
            print(f"    ... và {len(diff) - 10} video khác lệch số frame")


if __name__ == "__main__":
    main()
