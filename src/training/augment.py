"""Augmentation cho mẫu train: mỗi lần lấy mẫu 1 biến đổi ngẫu nhiên khác nhau (không lưu bản sao).

Làm trên chuỗi đầy đủ (T, 225) đã sửa tỉ lệ khung hình, TRƯỚC khi chuẩn hoá: xoay phải làm trên toạ độ thật,
tâm vai và cỡ lòng bàn tay tính SAU khi xoay. Đổi tốc độ ký nằm ở sample_positions (src/core/features.py).
Với chuẩn hoá theo vai (shoulder, shoulder_wrist), phóng to và dịch chuyển bị triệt tiêu (đúng mục đích),
chỉ có tác dụng với norm = none. Xoay và nhiễu thì không bị triệt tiêu.
Không dùng: bỏ frame, lật trái-phải (lật làm tay trái thành tay phải -> sai nghĩa ký hiệu).
"""
import numpy as np

from src.core.features import BODY, NUM_POINTS, shoulder_frame

ROTATE_DEG = 10      # xoay quanh tâm vai, mặt phẳng x-y, trong khoảng ±10°
SCALE = 0.10         # phóng to / thu nhỏ ±10%
SHIFT = 0.05         # dịch chuyển ±0.05 (đơn vị chiều cao ảnh)
NOISE = 0.004        # nhiễu Gauss, độ lệch chuẩn


def augment(seq, rng):
    """(T, 225) -> (T, 225) đã biến đổi. Điểm vắng mặt (= 0) giữ nguyên = 0."""
    pts = seq.reshape(len(seq), NUM_POINTS, 3).copy()
    present = pts.any(axis=2)                                    # (T, 75): điểm nào có dữ liệu
    center, _ = shoulder_frame(pts, pts[:, BODY].any(axis=(1, 2)))

    a = np.deg2rad(rng.uniform(-ROTATE_DEG, ROTATE_DEG))
    s = rng.uniform(1 - SCALE, 1 + SCALE)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * s
    shift = rng.uniform(-SHIFT, SHIFT, size=2)

    pts[:, :, :2] = (pts[:, :, :2] - center) @ rot.T + center + shift
    pts[:, :, 2] *= s
    pts += rng.normal(0, NOISE, size=pts.shape)
    pts[~present] = 0
    return pts.reshape(len(seq), -1).astype(np.float32)
