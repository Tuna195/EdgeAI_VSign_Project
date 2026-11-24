"""Nhận diện trực tiếp (bản đầu): camera -> MediaPipe -> tiền xử lý GIỐNG HỆT lúc train -> model -> top-3.

Chạy (từ thư mục gốc repo):
  python -m src.deploy.live_recognize --run gru_shoulder_wrist_z_f30_aug_wall/seed0
      webcam: SPACE = bắt đầu / kết thúc ghi một từ, Q = thoát
  python -m src.deploy.live_recognize --run gru_shoulder_wrist_z_f30_aug_wall/seed0 --video a.mp4 b.mp4
      không cần webcam: mỗi file video = một từ, in top-3 ra màn hình
  python -m src.deploy.live_recognize --run gru_shoulder_wrist_z_f30_aug_wall/seed0 --val 20
      kiểm tra: chạy cả đường ống trên 20 video VAL của dataset, so với đáp án
Tuỳ chọn: --camera 1 (chọn camera khác), --square (ép ảnh về 224x224 như video Kaggle, xem Phần 8c),
  --flip (lật ngang ảnh: video quay bằng camera trước thường bị lưu kiểu soi gương).
--video nhận file, thư mục, hoặc mẫu có * (vd. data/my_videos hoặc "data/my_videos/*/*.mp4").

Cần: torch, mediapipe, opencv-python, pillow; 2 file models/*.task;
thư mục runs/<cấu hình>/seed<k>/ có config.json, vocab.csv, best.pt (tạo khi train).
"""
import argparse
import csv
import glob
import json
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from src.core.features import make_features, prepare_sequence, sample_positions
from src.core.landmarks import create_hand, create_pose, draw_landmarks, frame_to_vector
from src.core.paths import INDEX_FILE, RUNS_DIR, VIDEO_DIR
from src.training.models import build_model

SQUARE = 224                  # kích thước video Kaggle (dùng khi --square)
WINDOW = "VSign - SPACE: bat dau/ket thuc, Q: thoat"


# ---------- model: đọc lại đúng cấu hình đã train ----------

def load_run(run):
    """-> (model ở chế độ eval, cfg, danh sách từ). run = 'tên cấu hình/seedk' trong runs/, hoặc đường dẫn thư mục."""
    folder = Path(run) if Path(run).is_dir() else RUNS_DIR / run
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))   # norm, use_z, n_frames, model, dim...
    with open(folder / "vocab.csv", encoding="utf-8-sig") as f:
        words = [word for _, word in csv.reader(f)]                         # nhãn -> từ
    model = build_model(cfg["model"], cfg["n_frames"], cfg["dim"], cfg["n_classes"], cfg["dropout"])
    model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu"))
    model.eval()                                                            # tắt dropout
    return model, cfg, words


def to_features(raw, width, height, cfg):
    """raw (T, 225) từ MediaPipe -> (đặc trưng (n_frames, D), None) hoặc (None, lý do). ĐÚNG các bước của val/test."""
    seq, reason = prepare_sequence(raw, width, height)       # sửa tỉ lệ khung hình, lấp mất tay, cắt frame trống
    if seq is None:
        return None, reason
    idx = sample_positions(len(seq), cfg["n_frames"])        # n_frames vị trí cách đều (không ngẫu nhiên)
    return make_features(seq, idx, cfg), None                # chuẩn hoá theo cfg["norm"], cfg["use_z"]


def predict(raw, width, height, model, cfg, words, k=3):
    """-> ([(từ, xác suất)] k từ cao nhất, None) hoặc (None, lý do không đoán được)."""
    feat, reason = to_features(raw, width, height, cfg)
    if feat is None:
        return None, reason
    with torch.no_grad():
        prob = torch.softmax(model(torch.from_numpy(feat)[None]), dim=1)[0]   # [None]: thêm trục batch -> (1, N, D)
    top = prob.topk(min(k, len(words)))
    return [(words[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())], None


# ---------- vẽ lên khung hình ----------

def load_font(size=26):
    """Font có dấu tiếng Việt: arial (Windows) hoặc DejaVuSans (Linux, Pi)."""
    for name in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def put_lines(frame, lines, font, line_height=34):
    """Viết các dòng chữ (có dấu) lên góc trên trái. lines = [(chữ, màu RGB)]. OpenCV không viết được chữ có dấu nên dùng PIL."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for i, (text, color) in enumerate(lines):
        draw.text((12, 10 + i * line_height), text, font=font, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def result_lines(top, reason, n_frames):
    if top is None:
        return [(f"Không đoán được ({n_frames} frame): {reason}", (255, 200, 0))]
    return [(f"{i + 1}. {word}  {p:.0%}", (60, 255, 60) if i == 0 else (255, 255, 255)) for i, (word, p) in enumerate(top)]


# ---------- hai chế độ chạy ----------

def run_camera(args, model, cfg, words):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được camera {args.camera}. Thử --camera 1, hoặc dùng --video <file>.")
    hand, pose, font = create_hand(), create_pose(), load_font()
    start = prev = time.monotonic()
    fps, last_ts = 0.0, -1
    recording, buffer = False, []
    result = [("Bấm SPACE để bắt đầu ký một từ", (255, 255, 255))]
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.flip:
                frame = cv2.flip(frame, 1)                                          # camera lật gương -> lật lại
            h, w = frame.shape[:2]
            image = cv2.resize(frame, (SQUARE, SQUARE)) if args.square else frame   # ảnh đưa vào MediaPipe (KHÔNG lật)
            last_ts = max(last_ts + 1, int((time.monotonic() - start) * 1000))      # mốc thời gian phải tăng dần
            vec = frame_to_vector(image, last_ts, hand, pose)                       # (225,)
            if recording:
                buffer.append(vec)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)                           # FPS làm mượt
            prev = now
            draw_landmarks(frame, vec)
            status = [(f"FPS {fps:.1f}", (200, 200, 200))]
            if recording:
                status.append((f"ĐANG GHI: {len(buffer)} frame (SPACE để dừng)", (255, 70, 70)))
            cv2.imshow(WINDOW, put_lines(frame, status + result, font))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):                                               # Q hoặc ESC
                break
            if key == ord(" "):
                if not recording:                                                   # bắt đầu ghi một từ
                    recording, buffer, result = True, [], []
                else:                                                               # kết thúc -> đoán
                    recording = False
                    raw = np.stack(buffer) if buffer else np.zeros((0, 225), np.float32)
                    size = (SQUARE, SQUARE) if args.square else (w, h)
                    top, reason = predict(raw, *size, model, cfg, words)
                    result = result_lines(top, reason, len(buffer))
                    print(f"{len(buffer)} frame ->", ", ".join(f"{t} {p:.0%}" for t, p in top) if top else reason)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand.close()
        pose.close()


def video_to_raw(path, square, flip=False):
    """File video -> (raw (T, 225), rộng, cao của ảnh đưa vào MediaPipe). Model MediaPipe mới cho mỗi file.
    flip: lật ngang mỗi frame (video quay bằng camera trước bị lưu kiểu soi gương)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Không mở được {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hand, pose = create_hand(), create_pose()
    vectors, size = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        image = cv2.resize(frame, (SQUARE, SQUARE)) if square else frame
        size = (image.shape[1], image.shape[0])
        vectors.append(frame_to_vector(image, int(len(vectors) * 1000 / fps), hand, pose))
    cap.release()
    hand.close()
    pose.close()
    raw = np.stack(vectors) if vectors else np.zeros((0, 225), np.float32)
    return raw, *(size or (1, 1))


def expand_videos(items):
    """--video nhận: file, thư mục (lấy mọi .mp4 bên trong), hoặc mẫu có * (vd. data/my_videos/*/*.mp4).
    Tự mở rộng * vì PowerShell trên Windows không làm việc này (bash trên Pi thì có)."""
    paths = []
    for item in items:
        if Path(item).is_dir():
            paths += sorted(Path(item).rglob("*.mp4"))
        elif any(c in item for c in "*?["):
            found = sorted(glob.glob(item, recursive=True))
            if not found:
                raise SystemExit(f"Không có file nào khớp {item}")
            paths += [Path(f) for f in found]
        else:
            paths.append(Path(item))
    return paths


def run_videos(videos, args, model, cfg, words):
    """Mỗi file = một từ. Nếu file nằm trong data/videos/<split>/<từ>/, in thêm đáp án đúng (tên thư mục)."""
    correct = 0
    for path in videos:
        raw, w, h = video_to_raw(path, args.square, args.flip)
        top, reason = predict(raw, w, h, model, cfg, words)
        answer = unicodedata.normalize("NFC", Path(path).parent.name)
        mark = ""
        if answer in words and top:
            ok = top[0][0] == answer
            correct += ok
            mark = f"   (đáp án: {answer}, {'ĐÚNG' if ok else 'SAI'})"
        shown = ", ".join(f"{t} {p:.0%}" for t, p in top) if top else f"không đoán được: {reason}"
        print(f"{path}: {len(raw)} frame, {w}x{h} -> {shown}{mark}")
    if len(videos) > 1:
        print(f"Đúng {correct}/{len(videos)} file (chỉ tính file có đáp án theo tên thư mục)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="vd. gru_shoulder_wrist_z_f30_aug_wall/seed0 (thư mục trong runs/)")
    p.add_argument("--camera", type=int, default=0, help="số thứ tự camera (mặc định 0)")
    p.add_argument("--video", nargs="+", help="file video / thư mục / mẫu có *, mỗi file = một từ (thay cho webcam)")
    p.add_argument("--val", type=int, help="chạy trên N video val ngẫu nhiên của dataset (kiểm tra đường ống)")
    p.add_argument("--square", action="store_true", help="ép ảnh về 224x224 trước MediaPipe (nếu 8c cho thấy dataset bị ép)")
    p.add_argument("--flip", action="store_true", help="lật ngang ảnh trước MediaPipe (video/camera bị lật kiểu soi gương)")
    args = p.parse_args()
    model, cfg, words = load_run(args.run)
    print(f"Model {cfg['model']}, norm {cfg['norm']}, z {cfg['use_z']}, {cfg['n_frames']} frame, {len(words)} từ")
    if args.val:
        index = pd.read_csv(INDEX_FILE, encoding="utf-8-sig")
        keys = index[index.split == "val"].key.sample(args.val, random_state=0)
        run_videos([VIDEO_DIR / f"{k}.mp4" for k in keys], args, model, cfg, words)
    elif args.video:
        run_videos(expand_videos(args.video), args, model, cfg, words)
    else:
        run_camera(args, model, cfg, words)


if __name__ == "__main__":
    main()
