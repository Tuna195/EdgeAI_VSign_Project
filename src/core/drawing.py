"""Vẽ dùng chung: khung xương, màu, viết chữ có dấu lên ảnh OpenCV (tách từ live_demo.py, view_video.py)."""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Các cặp điểm nối thành khung xương bàn tay (21 điểm)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),          # ngón trỏ
    (5, 9), (9, 10), (10, 11), (11, 12),     # ngón giữa
    (9, 13), (13, 14), (14, 15), (15, 16),   # ngón áp út
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # ngón út + lòng bàn tay
]

# Phần thân trên của Pose (33 điểm) - đủ dùng cho ngôn ngữ ký hiệu
POSE_CONNECTIONS = [
    (11, 12),                      # hai vai
    (11, 13), (13, 15),            # tay trái: vai - khuỷu - cổ tay
    (12, 14), (14, 16),            # tay phải
    (11, 23), (12, 24), (23, 24),  # thân
    (0, 9), (0, 10), (9, 10),      # mũi - miệng
]

LEFT_COLOR = (255, 128, 0)    # xanh dương (BGR)
RIGHT_COLOR = (0, 128, 255)   # cam
POSE_COLOR = (0, 255, 0)      # xanh lá


@lru_cache(maxsize=None)
def font(size):
    """arial.ttf có sẵn trên Windows; máy khác (Mac, Linux, Colab) thì thử DejaVuSans, cuối cùng font mặc định."""
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def put_text(img, xy, text, size=16, color=(0, 0, 0)):
    """Viết chữ có dấu lên ảnh OpenCV (qua PIL; cv2.putText không vẽ được chữ có dấu). color theo thứ tự RGB."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, font=font(size), fill=color)
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
