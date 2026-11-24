"""Kiểm tra video dataset có bị ÉP MÉO (vd. 16:9 ép thành 224x224) hay không, bằng tỉ lệ cơ thể.

Chạy: python -m src.review.check_aspect            đo trên dataset (data/raw, theo index.csv)
      python -m src.review.check_aspect --webcam   đo trên webcam của bạn (mốc chuẩn, không méo), nhấn q để dừng

Ý tưởng: ép ảnh theo chiều ngang làm mọi khoảng cách NGANG co lại, khoảng cách DỌC giữ nguyên.
Nên tỉ lệ ngang / dọc của cùng một bộ phận cơ thể sẽ nhỏ hơn bình thường. Hai tỉ lệ dùng ở đây:
  mắt / mắt-miệng   khoảng cách 2 mắt (ngang) / từ mắt xuống miệng (dọc)     (Pose 2, 5, 9, 10)
  vai / thân        độ rộng 2 vai (ngang) / từ vai xuống hông (dọc)          (Pose 11, 12, 23, 24)
Nếu dataset bị ép từ 16:9 thì các tỉ lệ này chỉ còn khoảng 0,56 lần so với webcam (4:3 ép vuông: 0,75 lần).
Nhìn thẳng vào camera khi đo webcam (quay nghiêng thì khoảng cách ngang co lại).
"""
import argparse
import csv
import time

import numpy as np

from src.core.features import fix_aspect
from src.core.paths import INDEX_FILE, RAW_DIR


def frame_ratios(raw):
    """(T, 225) -> (mắt / mắt-miệng, vai / thân) mỗi frame thấy thân; nan nếu không đo được."""
    p = raw[:, 126:].reshape(len(raw), 33, 3)
    p = p[p.any(axis=(1, 2))]
    with np.errstate(divide="ignore", invalid="ignore"):
        eye = np.abs(p[:, 2, 0] - p[:, 5, 0]) / np.abs((p[:, 9, 1] + p[:, 10, 1] - p[:, 2, 1] - p[:, 5, 1]) / 2)
        body = np.abs(p[:, 11, 0] - p[:, 12, 0]) / np.abs((p[:, 23, 1] + p[:, 24, 1] - p[:, 11, 1] - p[:, 12, 1]) / 2)
    return eye, body


def describe(name, values):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    print(f"  {name:16s} trung vị {np.median(v):.2f}   (50% ở giữa: {np.percentile(v, 25):.2f} - {np.percentile(v, 75):.2f})")


def check_dataset():
    with open(INDEX_FILE, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for source in ("A", "B"):
        eyes, bodies = [], []
        for r in (r for r in rows if r["source"] == source):
            split, word, vid = r["key"].split("/")
            raw = fix_aspect(np.load(RAW_DIR / split / word / f"{vid}.npy"), int(r["width"]), int(r["height"]))
            e, b = frame_ratios(raw)
            eyes.append(np.nanmedian(e))
            bodies.append(np.nanmedian(b))
        print(f"Nguồn {source} ({len(eyes)} video, mỗi video lấy trung vị các frame):")
        describe("mắt / mắt-miệng", eyes)
        describe("vai / thân", bodies)


def check_webcam():
    import cv2   # chỉ cần khi đo webcam
    from src.core.landmarks import create_landmarkers, frame_to_vector

    hand, pose = create_landmarkers(use_pose=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam")
    vectors, start = [], time.monotonic()
    w = h = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        vectors.append(frame_to_vector(frame, int((time.monotonic() - start) * 1000), hand, pose))
        cv2.putText(frame, f"{len(vectors)} frame - nhin thang camera, q de dung", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("check_aspect", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    hand.close()
    pose.close()
    e, b = frame_ratios(fix_aspect(np.stack(vectors), w, h))
    print(f"Webcam {w}x{h}, {len(vectors)} frame (đã sửa tỉ lệ x *= {w}/{h}):")
    describe("mắt / mắt-miệng", e)
    describe("vai / thân", b)
    print("  (vai / thân chỉ đúng khi thấy được hông trong khung hình: ngồi sát webcam thì bỏ qua dòng này)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam", action="store_true")
    check_webcam() if parser.parse_args().webcam else check_dataset()
