"""Kiểm tra make_features bằng số. Chạy: python -m src.explore.check_features"""
import numpy as np

from src.core.features import NORMS, feature_dim, make_features, points, sample_positions
from src.dataset.build_sequences import load_sequences

seq = load_sequences()["test/Bố/003266"]
idx = sample_positions(len(seq), 30)


def config(norm, use_z=True):
    return {"norm": norm, "use_z": use_z, "n_frames": 30}


print("1. Số cột của từng cấu hình")
for norm in NORMS:
    for use_z in (True, False):
        feat = make_features(seq, idx, config(norm, use_z))
        assert feat.shape == (30, feature_dim(config(norm, use_z)))
        print(f"   {norm:15s} z={use_z!s:5s} -> {feat.shape}")

print("2. Cùng đầu vào -> cùng kết quả")
for norm in NORMS:
    same = np.array_equal(make_features(seq, idx, config(norm)), make_features(seq, idx, config(norm)))
    print(f"   {norm:15s} {same}")


def move(seq, scale, shift):
    """Giả lập người đứng gần hơn (phóng to) và lệch đi (dịch). Điểm vắng mặt vẫn phải bằng 0."""
    pts = points(seq).copy()
    present = pts.any(axis=2)                    # (T, 75): điểm nào có dữ liệu
    pts[:, :, :2] = pts[:, :, :2] * scale + shift
    pts[:, :, 2] *= scale
    pts[~present] = 0
    return pts.reshape(len(seq), -1)


print("3. Phóng to 1,3 lần và dịch (0,10; -0,05): đặc trưng lệch nhiều nhất bao nhiêu?")
moved = move(seq, 1.3, np.array([0.10, -0.05], dtype=np.float32))
for norm in NORMS:
    diff = np.abs(make_features(seq, idx, config(norm)) - make_features(moved, idx, config(norm))).max()
    print(f"   {norm:15s} {diff:.7f}")