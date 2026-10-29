"""Demo trực tiếp: mở webcam, trích vector 225 chiều mỗi frame và vẽ lại các điểm lên hình.

Chạy: python -m src.deploy.live_demo   (nhấn q để thoát)
"""
import time

import cv2

from src.core.drawing import HAND_CONNECTIONS, LEFT_COLOR, POSE_COLOR, POSE_CONNECTIONS, RIGHT_COLOR
from src.core.landmarks import HAND_DIM, create_landmarkers, frame_to_vector

def draw_part(frame, part, connections, color):
    """Vẽ một phần của vector (tay hoặc thân). part là mảng phẳng (N*3,), bỏ qua nếu toàn 0."""
    if not part.any():
        return
    h, w = frame.shape[:2]
    xyz = part.reshape(-1, 3)   # (N*3,) -> (N, 3)
    # x, y là tỉ lệ 0..1 -> nhân kích thước ảnh để ra pixel
    points = [(int(x * w), int(y * h)) for x, y, _ in xyz]
    for a, b in connections:
        cv2.line(frame, points[a], points[b], color, 2)
    for p in points:
        cv2.circle(frame, p, 3, color, -1)


def draw_vector(frame, vec):
    """Vẽ lại vector 225 chiều: [trái 63 | phải 63 | thân 99]."""
    draw_part(frame, vec[2 * HAND_DIM:], POSE_CONNECTIONS, POSE_COLOR)
    draw_part(frame, vec[:HAND_DIM], HAND_CONNECTIONS, LEFT_COLOR)
    draw_part(frame, vec[HAND_DIM:2 * HAND_DIM], HAND_CONNECTIONS, RIGHT_COLOR)


def main():
    hand, pose = create_landmarkers(use_pose=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam")

    start = time.monotonic()
    prev = start
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Không lật ảnh: ảnh gốc -> nhãn Left/Right đúng tay thật (lật sẽ đảo trái-phải).
        # Vì vậy hình hiển thị giống nhìn người đối diện, không giống soi gương.

        # Mốc thời gian thật (ms), luôn tăng dần
        timestamp_ms = int((time.monotonic() - start) * 1000)
        vec = frame_to_vector(frame, timestamp_ms, hand, pose)
        draw_vector(frame, vec)

        # FPS làm mượt: trộn 90% giá trị cũ + 10% giá trị mới để số không nhảy loạn
        now = time.monotonic()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now

        has_left = vec[:HAND_DIM].any()
        has_right = vec[HAND_DIM:2 * HAND_DIM].any()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Left: {'yes' if has_left else '-'}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, LEFT_COLOR, 2)
        cv2.putText(frame, f"Right: {'yes' if has_right else '-'}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RIGHT_COLOR, 2)

        cv2.imshow("VSign live demo - nhan q de thoat", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand.close()
    if pose is not None:
        pose.close()


if __name__ == "__main__":
    main()
