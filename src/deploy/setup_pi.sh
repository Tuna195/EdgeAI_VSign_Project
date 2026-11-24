#!/usr/bin/env bash
# Cài môi trường chạy MediaPipe trên Raspberry Pi 4. Chạy TRÊN Pi, từ thư mục gốc repo:
#   bash src/deploy/setup_pi.sh
# Chạy lại nhiều lần cũng được: bước nào đã xong thì bỏ qua.
#
# ======================= KẾT LUẬN ĐÃ CÓ (không cần thử/đo lại) =======================
# Máy: Raspberry Pi 4 (Cortex-A72), Pi OS kernel rpi-v8, Python hệ thống 3.13.5.
# Camera: OV5647 (Pi Camera Module v1, cáp CSI), lấy nét cố định, rõ nhất ở khoảng >= 1 m.
#
# Phiên bản:
#   1. MediaPipe 1.x KHÔNG chạy trên Pi 4: "compiled with aes enabled ... Illegal instruction"
#      (CPU Pi 4 không có lệnh AES của ARMv8; Pi 5 thì có). -> dùng mediapipe==0.10.18.
#   2. mediapipe 0.10.18 chỉ có bản cho Python 3.9-3.12 -> KHÔNG dùng Python 3.13 của hệ thống,
#      tạo venv Python 3.12 bằng uv (uv tự tải Python dựng sẵn, không phải biên dịch).
#   3. Không cài thêm opencv-python(-headless): mediapipe đã kéo theo opencv-contrib-python.
#   4. picamera2 (cài bằng apt) chỉ chạy với Python 3.13 hệ thống -> trong venv 3.12, đọc camera
#      bằng rpicam-vid qua ống (src/deploy/pi_live.py). Tuỳ chọn --picam của bench_mediapipe KHÔNG dùng được.
#   5. Lệnh cài uv (curl ... | sh) phải chạy RIÊNG: dán chung khối lệnh thì bộ cài nuốt mất dòng kế tiếp.
#
# Kết quả đo trên Pi 4 (11-2026), Hand+Pose:
#   bench_mediapipe, video camera 640x480: Hand 122 ms, Pose 124 ms, Hand+Pose 248 ms (~4 FPS).
#   320x240 gần như bằng (237 ms) -> giảm độ phân giải KHÔNG làm nhanh hơn.
#   pi_live (chạy trực tiếp): MediaPipe 226 ms -> 4.3 FPS; camera đạt đủ 30 FPS -> nút thắt là MediaPipe.
#   Laptop: 35-130 ms tuỳ chế độ nguồn (cắm sạc/tiết kiệm pin). Hand và Pose đang chạy lần lượt -> cộng dồn.
#   Mỗi model chỉ dùng ~1 nhân CPU (laptop 22 nhân vẫn đo được ~1.0) -> 3/4 nhân của Pi 4 đang bỏ không.
#   Model dùng 30 frame/từ, từ ~2.5 s -> cần ~12 FPS; Pi 4 đang thiếu ~2.8 lần.
# ======================================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."                     # về thư mục gốc repo (nơi có src/)
export PATH="$HOME/.local/bin:$PATH"           # nơi bộ cài đặt uv

echo "== 1. uv =="
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

echo "== 2. venv Python 3.12 + mediapipe 0.10.18 =="
if [ ! -x venv/bin/python ]; then
    uv venv --python 3.12 venv
fi
uv pip install --python venv/bin/python mediapipe==0.10.18

echo "== 3. Model MediaPipe =="
mkdir -p models videos
base=https://storage.googleapis.com/mediapipe-models
[ -s models/hand_landmarker.task ] || \
    wget -q -O models/hand_landmarker.task "$base/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
[ -s models/pose_landmarker_lite.task ] || \
    wget -q -O models/pose_landmarker_lite.task "$base/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
ls -lh models

echo "== 4. Kiểm tra =="
venv/bin/python -c "import sys, cv2, mediapipe; print('python', sys.version.split()[0], '| mediapipe', mediapipe.__version__, '| opencv', cv2.__version__)"
rpicam-hello --list-cameras 2>/dev/null | head -3 || echo "(không thấy rpicam-hello / camera)"
vcgencmd get_throttled 2>/dev/null || true

echo
echo "Xong. Dùng:  source venv/bin/activate"
echo "  Xem trực tiếp: python -m src.deploy.pi_live 2> mediapipe.log   -> laptop mở http://raspberrypi.local:8000"
echo "  Đo tốc độ:     python -m src.deploy.bench_mediapipe --video videos/cam.mp4 2>/dev/null"
