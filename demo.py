import time

import cv2
import mediapipe as mp

# File mô hình nhận diện bàn tay (trong thư mục models/)
model_path = "models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Các cặp điểm cần nối để vẽ "khung xương" bàn tay (21 điểm, đánh số 0..20)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),          # ngón trỏ
    (5, 9), (9, 10), (10, 11), (11, 12),     # ngón giữa
    (9, 13), (13, 14), (14, 15), (15, 16),   # ngón áp út
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # ngón út + lòng bàn tay
]

# Ở chế độ LIVE_STREAM, MediaPipe xử lý bất đồng bộ và trả kết quả qua hàm callback.
# Ta lưu kết quả mới nhất vào biến này để vòng lặp chính vẽ lên khung hình.
latest_result = None


def save_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


def draw_landmarks(frame, result):
    h, w = frame.shape[:2]
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        # Toạ độ trả về là tỉ lệ 0..1, nhân với kích thước ảnh để ra pixel
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
        for p in points:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)
        # Ghi "Left"/"Right" cạnh cổ tay
        label = handedness[0].category_name
        cv2.putText(frame, label, (points[0][0], points[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=save_result,
)

with HandLandmarker.create_from_options(options) as landmarker:
    # Mở webcam (0 = camera mặc định)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam")

    start = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # lật ngang cho giống soi gương

        # OpenCV dùng thứ tự màu BGR, MediaPipe cần RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Mốc thời gian (ms) phải luôn tăng dần
        frame_timestamp_ms = int((time.monotonic() - start) * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        if latest_result is not None:
            draw_landmarks(frame, latest_result)

        cv2.imshow("Hand Landmarker - nhan q de thoat", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
