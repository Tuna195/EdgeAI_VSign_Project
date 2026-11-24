"""Đo tốc độ MediaPipe (Phần 8b): ms/frame và FPS cho Hand riêng, Pose riêng, cả hai, ở 2 độ phân giải.

Chạy trên Pi (hoặc PC để so sánh), từ thư mục gốc repo:
  python -m src.deploy.bench_mediapipe                   dùng webcam USB (camera số 0)
  python -m src.deploy.bench_mediapipe --picam           dùng Pi Camera (cáp dẹt CSI), cần picamera2
  python -m src.deploy.bench_mediapipe --video a.mp4     dùng file video (khi không có camera)

Đo bằng camera: ngồi trước camera, giơ tay ra khi script đang chụp (~4 giây đầu),
vì MediaPipe chạy nhanh/chậm khác nhau tùy có thấy tay hay không.
"""
import argparse
import platform
import time

import cv2

from src.core.landmarks import HAND_DIM, create_hand, create_pose, frame_to_vector

SIZES = [(640, 480), (320, 240)]
WARMUP, FRAMES = 10, 100      # bỏ qua 10 frame đầu (máy "làm nóng"), đo 100 frame


def grab_frames(source, n):
    """Đọc n frame từ camera hoặc file (file ngắn thì đọc vòng lại). Đo luôn tốc độ đọc camera."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được {source!r}. Không có webcam USB thì dùng --video <file>.")
    frames, start = [], time.perf_counter()
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            if not frames:
                raise SystemExit("Không đọc được frame nào")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)          # file hết thì đọc lại từ đầu
            continue
        frames.append(frame)
    read_fps = n / (time.perf_counter() - start)
    cap.release()
    return frames, read_fps


def grab_picam(n, size=(640, 480)):
    """Đọc n frame từ Pi Camera (CSI). OpenCV không đọc trực tiếp được loại này, phải qua picamera2 (libcamera)."""
    from picamera2 import Picamera2                     # import ở đây: máy không có Pi Camera vẫn chạy được script
    cam = Picamera2()
    # "RGB888" của picamera2 thực ra xếp byte theo thứ tự B,G,R -> đúng định dạng BGR mà OpenCV dùng
    cam.configure(cam.create_video_configuration(main={"size": size, "format": "RGB888"}))
    cam.start()
    frames, start = [], time.perf_counter()
    while len(frames) < n:
        frames.append(cam.capture_array())
    read_fps = n / (time.perf_counter() - start)
    cam.stop()
    cam.close()
    return frames, read_fps


def measure(frames, use_hand, use_pose):
    """Trên các frame đã chụp sẵn (không tính thời gian đọc camera), trả về:
    ms/frame trung bình, số tay thấy được trung bình mỗi frame (0..2), % frame thấy người.
    Cần 2 số sau vì tốc độ MediaPipe phụ thuộc vào việc nó có thấy tay/người hay không."""
    hand = create_hand() if use_hand else None
    pose = create_pose() if use_pose else None
    times, cpu, n_hands, n_body = [], 0.0, 0, 0
    for i, frame in enumerate(frames):
        start, start_cpu = time.perf_counter(), time.process_time()
        vec = frame_to_vector(frame, i * 33, hand, pose)  # timestamp giả, tăng đều ~30 FPS
        if i >= WARMUP:
            times.append(time.perf_counter() - start)
            cpu += time.process_time() - start_cpu        # thời gian CPU của MỌI luồng trong tiến trình
            n_hands += int(vec[:HAND_DIM].any()) + int(vec[HAND_DIM:2 * HAND_DIM].any())
            n_body += int(vec[2 * HAND_DIM:].any())
    for m in (hand, pose):
        if m is not None:
            m.close()
    n = len(times)
    # số nhân dùng trung bình = thời gian CPU / thời gian thực (1.0 = đúng 1 nhân, 4.0 = cả 4 nhân của Pi 4)
    return 1000 * sum(times) / n, n_hands / n, 100 * n_body / n, cpu / sum(times)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video", help="đường dẫn file video thay cho camera")
    src.add_argument("--picam", action="store_true", help="dùng Pi Camera (CSI) qua picamera2")
    args = p.parse_args()
    if args.picam:
        frames, read_fps = grab_picam(WARMUP + FRAMES)
    else:
        frames, read_fps = grab_frames(args.video or 0, WARMUP + FRAMES)
    h, w = frames[0].shape[:2]
    name = args.video or ("Pi Camera" if args.picam else "webcam USB")
    print(f"Máy: {platform.machine()} | {platform.platform()}")
    print(f"Nguồn: {name}, {w}x{h}, đọc được {read_fps:.1f} frame/giây"
          + (" (tốc độ đọc FILE, không phải camera)" if args.video else ""))
    print(f"\n{'kích thước':10s} {'mô hình':10s} {'ms/frame':>9s} {'FPS':>6s} {'số tay TB':>10s} {'thấy người':>11s}"
          f" {'số nhân':>8s}")
    for size in SIZES:
        small = [cv2.resize(f, size) for f in frames]     # thu nhỏ trước khi đưa vào MediaPipe
        for label, use_hand, use_pose in (("Hand", True, False), ("Pose", False, True), ("Hand+Pose", True, True)):
            ms, hands, body, cores = measure(small, use_hand, use_pose)
            hands_txt = f"{hands:.2f}" if use_hand else "-"
            body_txt = f"{body:.0f}%" if use_pose else "-"
            print(f"{size[0]}x{size[1]:<6d} {label:10s} {ms:9.1f} {1000 / ms:6.1f} {hands_txt:>10s} {body_txt:>11s}"
                  f" {cores:8.2f}", flush=True)            # flush: in ngay, không bị log của MediaPipe chen lẫn


if __name__ == "__main__":
    main()