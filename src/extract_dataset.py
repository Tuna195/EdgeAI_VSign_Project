"""Trích landmark thô cho mọi video trong data/videos -> data/raw (mỗi video 1 file .npy).

Chạy: python src/extract_dataset.py            (8 tiến trình song song)
      python src/extract_dataset.py --workers 4

data/videos/<train|test>/<từ>/<id>.mp4  ->  data/raw/<train|test>/<từ>/<id>.npy  (T, 225) float32
Lưu dữ liệu THÔ (chưa chuẩn hoá): MediaPipe là bước chậm nhất, sau này đổi cách chuẩn hoá
hay số frame thì chỉ cần chạy lại phần NumPy, không phải chạy lại MediaPipe.
File .npy đã có thì bỏ qua -> bị ngắt giữa chừng thì chạy lại là tiếp tục.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from landmarks import video_to_raw

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "videos"
RAW_DIR = ROOT / "data" / "raw"


def extract_one(paths):
    video_path, out_path = paths
    try:
        arr = video_to_raw(video_path)
    except Exception as e:
        return f"{video_path}: {e}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    jobs = []
    for video in sorted(VIDEO_DIR.glob("*/*/*.mp4")):
        out = RAW_DIR / video.relative_to(VIDEO_DIR).with_suffix(".npy")
        if not out.exists():
            jobs.append((video, out))
    print(f"Cần trích {len(jobs)} video")

    start = time.time()
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, err in enumerate(pool.map(extract_one, jobs), 1):
            if err:
                errors.append(err)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({time.time() - start:.0f}s)", flush=True)

    print("Xong." if not errors else f"Lỗi {len(errors)} video:")
    for e in errors:
        print("  ", e)


if __name__ == "__main__":
    main()
