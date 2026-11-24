"""MediaPipe: ảnh -> vector 225 số mỗi frame [tay trái 63 | tay phải 63 | thân 99]. Phần vắng mặt = 0.

Cần: pip install mediapipe opencv-python, và 2 file models/hand_landmarker.task, models/pose_landmarker_lite.task.
Đây chính là cách data/raw đã được tạo (giai đoạn trước). Dùng lại cho: đo FPS trên Pi (8b), video tự quay (8c),
nhận diện trực tiếp (sau này).
"""
import cv2
import mediapipe as mp
import numpy as np

from src.core.paths import MODELS_DIR

vision = mp.tasks.vision
BaseOptions = mp.tasks.BaseOptions
HAND_DIM, POSE_DIM = 21 * 3, 33 * 3      # 63, 99
VECTOR_DIM = 2 * HAND_DIM + POSE_DIM     # 225


def create_hand():
    """Model Hand (tìm tối đa 2 tay), chế độ VIDEO: nhớ vị trí tay của frame trước để tìm nhanh hơn."""
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))


def create_pose():
    """Model Pose (bản lite, nhẹ nhất), chế độ VIDEO."""
    return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.VIDEO))


def to_array(landmarks):
    """Danh sách N điểm landmark -> mảng phẳng (N*3,) gồm x, y, z của từng điểm."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()


def frame_to_vector(frame_bgr, timestamp_ms, hand, pose):
    """1 ảnh (OpenCV, BGR) -> mảng (225,) float32. hand hoặc pose = None thì phần đó để 0.

    timestamp_ms phải tăng dần giữa các lần gọi (yêu cầu của chế độ VIDEO).
    KHÔNG lật ảnh: với ảnh gốc, nhãn Left/Right khớp đúng tay thật của người ký.
    """
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    left, right = np.zeros(HAND_DIM, np.float32), np.zeros(HAND_DIM, np.float32)
    if hand is not None:
        result = hand.detect_for_video(image, timestamp_ms)
        for marks, handedness in zip(result.hand_landmarks, result.handedness):
            if handedness[0].category_name == "Left":
                left = to_array(marks)
            else:
                right = to_array(marks)
    body = np.zeros(POSE_DIM, np.float32)
    if pose is not None:
        result = pose.detect_for_video(image, timestamp_ms)
        if result.pose_landmarks:
            body = to_array(result.pose_landmarks[0])       # chỉ lấy người đầu tiên
    return np.concatenate([left, right, body])


def video_to_raw(path):
    """File video -> (mảng (T, 225) float32, rộng, cao). Tạo model MỚI cho mỗi video (chế độ VIDEO nhớ frame trước)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hand, pose = create_hand(), create_pose()
    vectors = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vectors.append(frame_to_vector(frame, int(len(vectors) * 1000 / fps), hand, pose))
    finally:
        cap.release()
        hand.close()
        pose.close()
    raw = np.stack(vectors) if vectors else np.zeros((0, VECTOR_DIM), np.float32)
    return raw, width, height

# Các cặp điểm nối với nhau khi vẽ bộ xương (số thứ tự điểm theo MediaPipe)
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
POSE_LINKS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 2), (0, 5), (9, 10)]


def draw_landmarks(frame, vec):
    """Vẽ bộ xương từ vector 225 số (toạ độ 0..1 theo ảnh) lên frame. Tay trái xanh dương, tay phải cam, thân xám."""
    h, w = frame.shape[:2]
    pts = vec.reshape(75, 3)
    px = lambda p: (int(p[0] * w), int(p[1] * h))
    for start, color in ((0, (255, 128, 0)), (21, (0, 128, 255))):          # màu theo thứ tự BGR của OpenCV
        hand = pts[start:start + 21]
        if hand.any():                                                     # tay vắng mặt (toàn 0) -> không vẽ
            for a, b in HAND_LINKS:
                cv2.line(frame, px(hand[a]), px(hand[b]), color, 2)
    body = pts[42:]
    if body.any():
        for a, b in POSE_LINKS:
            cv2.line(frame, px(body[a]), px(body[b]), (180, 180, 180), 2)
