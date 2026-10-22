"""Tạo ảnh tổng hợp để duyệt video bằng mắt: mỗi từ 1 ảnh, mỗi video 1 hàng gồm 8 frame cách đều.

Chạy: python src/make_review_sheets.py            (tất cả các từ)
      python src/make_review_sheets.py "Chào"     (1 từ)

Kết quả: data/reports/sheets/<từ>.jpg. Đầu mỗi hàng ghi train/test + id video, hàng nghi trùng
(theo data/reports/duplicates.csv) có viền đỏ -> mở ảnh lên là thấy ngay video nào giống nhau.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "videos"
REPORT_DIR = ROOT / "data" / "reports"
OUT_DIR = REPORT_DIR / "sheets"
N_FRAMES, SIZE, LABEL_W = 8, 96, 110


def video_row(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    idx = np.linspace(0, len(frames) - 1, N_FRAMES).round().astype(int)
    return np.hstack([cv2.resize(frames[k], (SIZE, SIZE)) for k in idx]), len(frames)


def main():
    flagged = set()
    dup_file = REPORT_DIR / "duplicates.csv"
    if dup_file.exists():
        for r in csv.DictReader(open(dup_file, encoding="utf-8-sig")):
            flagged.update([(r["word"], r["video_a"]), (r["word"], r["video_b"])])

    words = sys.argv[1:] or sorted({p.name for p in VIDEO_DIR.glob("*/*") if p.is_dir()})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for word in words:
        rows = []
        for split in ("train", "test"):
            for video in sorted((VIDEO_DIR / split / word).glob("*.mp4")):
                strip, n = video_row(video)
                label = np.full((SIZE, LABEL_W, 3), 255, np.uint8)
                # cv2.putText không vẽ được chữ có dấu -> chỉ ghi split, id, số frame
                cv2.putText(label, split, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(label, video.stem, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(label, f"{n} frame", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
                row = np.hstack([label, strip])
                if (word, f"{split}/{video.stem}") in flagged:
                    cv2.rectangle(row, (0, 0), (row.shape[1] - 1, SIZE - 1), (0, 0, 255), 4)
                rows.append(row)
        if rows:
            # imencode + tofile thay cho imwrite: imwrite trên Windows lỗi với tên file có dấu
            ok, buf = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
            buf.tofile(str(OUT_DIR / f"{word}.jpg"))
            print(f"{word}: {len(rows)} video")


if __name__ == "__main__":
    main()
