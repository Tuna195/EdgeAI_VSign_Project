"""Tiền xử lý: landmark thô (T, 225) -> mẫu cho model (N_FRAMES, 227), và tạo dataset.npz.

Chạy: python src/preprocess.py   (đọc data/raw, ghi data/processed/dataset.npz)

Pipeline cho mỗi video:
  1. fill_gaps   lấp chỗ MediaPipe mất tay vài frame giữa chừng (tay chuyển động nhanh bị nhoè)
  2. normalize   dời gốc về giữa hai vai, đơn vị = độ rộng vai (trung vị cả video),
                 thêm 2 cờ có tay trái/phải -> (T, 227)
  3. trim_idle   cắt đoạn tay buông ở đầu và cuối
  4. resample    chọn N_FRAMES frame cách đều -> (N_FRAMES, 227)
"""
import csv
from pathlib import Path

import numpy as np

from landmarks import HAND_DIM

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = ROOT / "data" / "processed" / "dataset.npz"
EXCLUDE_FILE = ROOT / "data" / "exclude.txt"   # video loại sau khi xem bằng mắt, mỗi dòng: train/<từ>/<id>

N_FRAMES = 30        # một số video trong dataset chỉ có đúng 30 frame -> lấy 30, không lấy 60
NUM_POINTS = 75      # 21 tay trái + 21 tay phải + 33 thân
L_SH, R_SH = 11, 12  # vai trái, vai phải (chỉ số trong 33 điểm Pose)
L_WRIST, R_WRIST = 15, 16
MAX_GAP = 4          # chỉ lấp khoảng mất tay <= 4 frame; dài hơn coi như tay thật sự vắng mặt
REST_Y = 1.2         # cổ tay thấp hơn vai > 1.2 lần độ rộng vai = tay đang buông (gần hông)
MARGIN = 2           # giữ thêm 2 frame trước và sau đoạn ký hiệu


def fill_gaps(raw, max_gap=MAX_GAP):
    """(T, 225) -> (T, 225). Nội suy tuyến tính tay bị mất <= max_gap frame, nếu 2 đầu đều thấy tay."""
    out = raw.copy()
    for start in (0, HAND_DIM):   # tay trái, tay phải
        part = out[:, start:start + HAND_DIM]
        seen = np.flatnonzero(part.any(axis=1))
        for a, b in zip(seen[:-1], seen[1:]):
            if 1 < b - a <= max_gap + 1:
                w = np.linspace(0, 1, b - a + 1)[1:-1, None]   # trọng số cho các frame ở giữa
                part[a + 1:b] = (1 - w) * part[a] + w * part[b]
    return out


def normalize(raw):
    """(T, 225) -> (T, 227). Trả về None nếu không thấy người ở frame nào."""
    T = len(raw)
    pts = raw.reshape(T, NUM_POINTS, 3).copy()
    left_ok = pts[:, :21].any(axis=(1, 2))
    right_ok = pts[:, 21:42].any(axis=(1, 2))
    body = pts[:, 42:]
    body_ok = body.any(axis=(1, 2))
    if not body_ok.any():
        return None

    # Dùng 1 tâm và 1 độ rộng vai cho cả video (trung vị các frame thấy thân), không tính lại
    # từng frame: người ký đứng yên nên vai gần như không đổi, còn 1 frame MediaPipe đặt sai vai
    # (đã gặp: độ rộng vai 0.013 thay vì 0.28) sẽ làm toạ độ frame đó phóng to gấp 20 lần.
    shoulders = body[body_ok][:, [L_SH, R_SH], :2]                                  # (T_ok, 2, 2)
    center = np.median(shoulders.mean(axis=1), axis=0)                              # (2,)
    scale = np.median(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1))   # số
    scale = max(scale, 1e-6)

    pts[:, :, :2] = (pts[:, :, :2] - center) / scale   # chỉ x, y; giữ z
    pts[~left_ok, :21] = 0      # điểm vắng mặt trở lại 0
    pts[~right_ok, 21:42] = 0
    pts[~body_ok, 42:] = 0
    flags = np.stack([left_ok, right_ok], axis=1).astype(np.float32)
    return np.concatenate([pts.reshape(T, -1), flags], axis=1)


def active_range(seq, margin=MARGIN):
    """Trả về (a, b): đoạn seq[a:b] có ít nhất 1 cổ tay giơ cao hơn mức buông tay, hoặc None.

    Không dùng "có thấy tay hay không" như kế hoạch ban đầu: video dataset quay cả người,
    tay buông vẫn nằm trong khung. Độ cao cổ tay (theo Pose) dùng được cho cả dataset lẫn
    camera kiosk (khung đến ngang eo: tay buông thì cổ tay ở dưới mép ảnh).
    """
    body = seq[:, 2 * HAND_DIM:2 * HAND_DIM + 99].reshape(len(seq), 33, 3)
    wrist_y = np.minimum(body[:, L_WRIST, 1], body[:, R_WRIST, 1])   # y hướng xuống: nhỏ = cao
    # Frame không thấy thân (vd. frame trống 1 màu ở đầu/cuối vài video) có toạ độ = 0,
    # nếu không loại sẽ bị tính nhầm là "cổ tay giơ cao"
    wrist_y[~body.any(axis=(1, 2))] = np.inf
    active = np.flatnonzero(wrist_y < REST_Y)
    if len(active) == 0:
        return None
    return max(active[0] - margin, 0), min(active[-1] + margin + 1, len(seq))


def trim_idle(seq, margin=MARGIN):
    """Cắt đoạn tay buông ở đầu và cuối. seq là đầu ra của normalize."""
    rng = active_range(seq, margin)
    return None if rng is None else seq[rng[0]:rng[1]]


def sample_indices(length, n=N_FRAMES):
    """Chỉ số n frame cách đều trong đoạn dài length (có thể lặp nếu length < n)."""
    return np.linspace(0, length - 1, n).round().astype(int)


def resample(seq, n=N_FRAMES):
    """Chọn n frame cách đều (frame gần nhất, không nội suy: tránh toạ độ giả giữa có tay và không tay)."""
    return seq[sample_indices(len(seq), n)]


def raw_to_sample(raw):
    """Toàn bộ pipeline cho 1 video. Trả về (N_FRAMES, 227) hoặc None nếu video không dùng được."""
    seq = normalize(fill_gaps(raw)) if len(raw) else None
    seq = trim_idle(seq) if seq is not None else None
    if seq is None or len(seq) < 10:
        return None
    return resample(seq)


def build_dataset():
    exclude = set()
    if EXCLUDE_FILE.exists():
        exclude = {line.strip() for line in EXCLUDE_FILE.read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.startswith("#")}

    words = sorted({p.name for p in RAW_DIR.glob("*/*") if p.is_dir()})
    X, y, splits, ids, lengths = [], [], [], [], []
    skipped = []
    for npy in sorted(RAW_DIR.glob("*/*/*.npy")):
        split, word = npy.parent.parent.name, npy.parent.name
        key = f"{split}/{word}/{npy.stem}"
        if key in exclude:
            continue
        raw = np.load(npy)
        sample = raw_to_sample(raw)
        if sample is None:
            skipped.append(key)
            continue
        X.append(sample)
        y.append(words.index(word))
        splits.append(split)
        ids.append(npy.stem)
        lengths.append(len(raw))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_FILE, X=np.stack(X).astype(np.float32), y=np.array(y), words=np.array(words),
                        splits=np.array(splits), ids=np.array(ids), lengths=np.array(lengths))
    with open(OUT_FILE.with_name("vocab.csv"), "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([[i, w] for i, w in enumerate(words)])

    print(f"Xong: {len(X)} mẫu, {len(words)} từ, X.shape = {np.stack(X).shape}")
    print(f"  train {splits.count('train')}, test {splits.count('test')}, loại {len(exclude)}, bỏ qua {len(skipped)}")
    for k in skipped[:20]:
        print("  bỏ qua:", k)


if __name__ == "__main__":
    build_dataset()
