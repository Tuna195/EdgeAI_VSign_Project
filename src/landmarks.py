from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

vision = mp.tasks.vision
BaseOptions = mp.tasks.BaseOptions
# Thư mục models/ nằm cạnh src/ -> tính theo vị trí file này, chạy từ đâu cũng đúng
MODELS = Path(__file__).resolve().parent.parent / "models"

NUM_HAND_POINTS = 21   # mỗi bàn tay 21 điểm
NUM_POSE_POINTS = 33   # thân người 33 điểm
HAND_DIM = NUM_HAND_POINTS * 3   # 63  (x, y, z)
POSE_DIM = NUM_POSE_POINTS * 3   # 99
VECTOR_DIM = HAND_DIM * 2 + POSE_DIM   # 225


def create_landmarkers(use_pose=True):
    """Tạo Hand (tối đa 2 tay) và Pose ở chế độ VIDEO. Trả về (hand, pose); pose = None nếu use_pose=False."""
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    ))
    if use_pose:
        pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_lite.task")),
            running_mode=vision.RunningMode.VIDEO,
        ))
    else:
        pose = None
    return (hand, pose)


def landmarks_to_array(landmarks):
    """Danh sách N điểm landmark -> mảng phẳng (N*3,) gồm x, y, z của từng điểm."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()


def frame_to_vector(frame_bgr, timestamp_ms, hand, pose):
    """Trả về mảng (225,) float32: tay trái 63 + tay phải 63 + thân 99. Phần vắng mặt = 0.

    timestamp_ms phải tăng dần giữa các lần gọi (yêu cầu của chế độ VIDEO).
    Đưa ảnh gốc vào, KHÔNG lật (cv2.flip): với ảnh gốc, nhãn Left/Right khớp đúng tay thật
    của người ký hiệu (đã kiểm tra bằng thí nghiệm). Lật trái-phải hay trên-dưới đều làm đảo nhãn.
    """
    # OpenCV đọc ảnh theo thứ tự BGR, MediaPipe cần RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Tay: mặc định là 0 (không thấy tay), có tay nào thì điền vào đúng ô theo nhãn
    left = np.zeros(HAND_DIM, dtype=np.float32)
    right = np.zeros(HAND_DIM, dtype=np.float32)
    hand_result = hand.detect_for_video(mp_image, timestamp_ms)
    for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        label = handedness[0].category_name   # "Left" hoặc "Right"
        if label == "Left":
            left = landmarks_to_array(hand_landmarks)
        else:
            right = landmarks_to_array(hand_landmarks)

    # Thân: chỉ lấy người đầu tiên
    body = np.zeros(POSE_DIM, dtype=np.float32)
    if pose is not None:
        pose_result = pose.detect_for_video(mp_image, timestamp_ms)
        if pose_result.pose_landmarks:
            body = landmarks_to_array(pose_result.pose_landmarks[0])

    # Ghép lại: [trái | phải | thân]
    return np.concatenate([left, right, body])


def video_to_raw(video_path, use_pose=True):
    """Đọc 1 file video -> mảng (T, 225) float32, T = số frame. Chưa chuẩn hoá.

    Tạo landmarker MỚI cho mỗi video: ở chế độ VIDEO, MediaPipe nhớ vị trí tay của frame trước
    (tracking), dùng chung sẽ mang tay của video trước sang video sau.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0   # file lỗi metadata thì coi như 30 FPS

    hand, pose = create_landmarkers(use_pose)
    vectors = []
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Timestamp theo FPS của video (không dùng đồng hồ thật), luôn tăng dần
            timestamp_ms = int(i * 1000 / fps)
            vectors.append(frame_to_vector(frame, timestamp_ms, hand, pose))
            i += 1
    finally:
        cap.release()
        hand.close()
        if pose is not None:
            pose.close()

    if not vectors:
        return np.zeros((0, VECTOR_DIM), dtype=np.float32)
    return np.stack(vectors)


if __name__ == "__main__":
    # Tự kiểm tra: đọc 30 frame webcam, in shape và phần nào khác 0
    hand, pose = create_landmarkers(use_pose=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam")

    timestamp_ms = 0
    for i in range(30):
        ok, frame = cap.read()
        if not ok:
            break
        # Không lật ảnh, đưa ảnh gốc vào (xem frame_to_vector)
        timestamp_ms += 33           # ~30 FPS, chỉ cần tăng dần
        vec = frame_to_vector(frame, timestamp_ms, hand, pose)

        has_left = vec[:HAND_DIM].any()
        has_right = vec[HAND_DIM:2 * HAND_DIM].any()
        has_body = vec[2 * HAND_DIM:].any()
        print(f"frame {i:2d}: shape={vec.shape} dtype={vec.dtype} "
              f"trái={'có' if has_left else '-'} phải={'có' if has_right else '-'} thân={'có' if has_body else '-'}")

    cap.release()
    hand.close()
    if pose is not None:
    	pose.close()
