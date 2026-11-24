"""Augmentation cho video train: mỗi lần lấy mẫu một biến đổi ngẫu nhiên khác nhau (không lưu bản sao nào).

Làm trên chuỗi đầy đủ (T, 225) đã làm sạch (Phần 3), TRƯỚC khi chuẩn hoá: xoay phải làm trên toạ độ thật.
Đổi tốc độ ký nằm ở sample_positions (src/core/features.py).
Không dùng: bỏ frame, lật trái-phải (lật làm tay trái thành tay phải -> sai nghĩa ký hiệu).
"""
import numpy as np

from src.core.features import points, shoulder_frame

ROTATE_DEG = 10      # xoay quanh tâm vai, trong mặt phẳng x-y, góc ngẫu nhiên trong khoảng ±10°
SCALE = 0.10         # phóng to / thu nhỏ ±10%
SHIFT = 0.05         # dịch chuyển ±0,05 (đơn vị: chiều cao ảnh)
NOISE = 0.004        # nhiễu ngẫu nhiên cộng vào mọi toạ độ, độ lệch chuẩn 0,004


def augment(seq, rng):
    """(T, 225) -> (T, 225) đã biến đổi. rng: bộ sinh số ngẫu nhiên có seed. Điểm vắng mặt (= 0) vẫn = 0."""
    pts = points(seq).copy()
    present = pts.any(axis=2)                                    # (T, 75): điểm nào có dữ liệu
    center, _ = shoulder_frame(pts)                              # tâm vai: xoay và phóng quanh điểm này

    a = np.deg2rad(rng.uniform(-ROTATE_DEG, ROTATE_DEG))         # góc xoay (radian)
    s = rng.uniform(1 - SCALE, 1 + SCALE)                        # hệ số phóng
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * s    # ma trận xoay + phóng
    shift = rng.uniform(-SHIFT, SHIFT, size=2)

    pts[:, :, :2] = (pts[:, :, :2] - center) @ rot.T + center + shift
    pts[:, :, 2] *= s
    pts += rng.normal(0, NOISE, size=pts.shape)
    pts[~present] = 0                                            # điểm vắng mặt trở lại 0
    return pts.reshape(len(seq), -1).astype(np.float32)
