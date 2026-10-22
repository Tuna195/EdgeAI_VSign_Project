"""Xem video dataset kèm landmark để duyệt bằng mắt.

Chạy: python src/view_video.py "Bố"                     (mọi video của 1 từ)
      python src/view_video.py "test/Bố/012339"         (1 video)
      python src/view_video.py --list data/reports/review_list.txt   (danh sách có ghi chú)
      python src/view_video.py "Bố" --snapshot out.jpg  (không mở cửa sổ, lưu ảnh 8 frame/video)

Phím: Space dừng/chạy   A/D lùi/tiến 1 frame (khi dừng)   N/P video sau/trước
      +/- tốc độ        Q hoặc Esc thoát

Hình vẽ lại đúng dữ liệu đã lưu trong data/raw (thứ sẽ đưa vào model), không chạy lại MediaPipe.
  tay trái = xanh dương, tay phải = cam, thân = xanh lá, tay xám = điểm nội suy (fill_gaps)
Thanh thời gian bên dưới, mỗi cột là 1 frame:
  Trái / Phải  có màu = thấy tay, xám = nội suy, tối = mất tay, đỏ = nghi gán nhầm trái/phải
  Giữ lại      xanh = đoạn trim_idle giữ lại, vạch trắng = 30 frame được chọn cho model
"""
import argparse
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from landmarks import HAND_DIM, video_to_raw
from live_demo import HAND_CONNECTIONS, LEFT_COLOR, POSE_COLOR, POSE_CONNECTIONS, RIGHT_COLOR
from preprocess import active_range, fill_gaps, normalize, sample_indices

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "videos"
RAW_DIR = ROOT / "data" / "raw"
FONT = ImageFont.truetype("arial.ttf", 16)   # PIL vẽ được chữ có dấu, cv2.putText thì không

SCALE = 3                 # video 224x224 -> phóng to 672x672 cho dễ nhìn
HEADER_H, STRIP_H = 70, 14
FILLED_COLOR = (170, 170, 170)
SWAP_COLOR = (0, 0, 255)
L_WRIST, R_WRIST = 15, 16


def find_videos(target):
    """'Bố' -> mọi video của từ; 'test/Bố/012339' -> 1 video. Trả về danh sách (split, word, id)."""
    parts = target.strip().split("/")
    if len(parts) == 3:
        return [tuple(parts)]
    return [(split, target, p.stem) for split in ("train", "test")
            for p in sorted((VIDEO_DIR / split / target).glob("*.mp4"))]


def read_list(path):
    """Mỗi dòng: split/từ/id | ghi chú. Dòng bắt đầu bằng # là tiêu đề nhóm (hiện kèm ghi chú)."""
    items, group = [], ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            group = line.lstrip("# ")
            continue
        target, _, note = line.partition("|")
        for v in find_videos(target):
            items.append((v, f"[{group}] {note.strip()}" if group else note.strip()))
    return items


def load(split, word, vid):
    """Đọc frame của video + landmark thô, rồi tính các thông tin để vẽ."""
    path = VIDEO_DIR / split / word / f"{vid}.mp4"
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    npy = RAW_DIR / split / word / f"{vid}.npy"
    raw = np.load(npy) if npy.exists() else video_to_raw(path)
    filled = fill_gaps(raw)
    T = len(raw)

    seen = {"L": raw[:, :HAND_DIM].any(axis=1), "R": raw[:, HAND_DIM:2 * HAND_DIM].any(axis=1)}
    interp = {"L": filled[:, :HAND_DIM].any(axis=1) & ~seen["L"],
              "R": filled[:, HAND_DIM:2 * HAND_DIM].any(axis=1) & ~seen["R"]}

    # Nghi gán nhầm: cổ tay của bàn tay nhãn "Left" lại gần cổ tay PHẢI của Pose hơn (và ngược lại)
    pose = raw[:, 2 * HAND_DIM:].reshape(T, 33, 3)[:, :, :2]
    has_pose = raw[:, 2 * HAND_DIM:].any(axis=1)
    swap = {}
    for side, start, own, other in (("L", 0, L_WRIST, R_WRIST), ("R", HAND_DIM, R_WRIST, L_WRIST)):
        wrist = raw[:, start:start + 2]
        d_own = np.linalg.norm(wrist - pose[:, own], axis=1)
        d_other = np.linalg.norm(wrist - pose[:, other], axis=1)
        swap[side] = seen[side] & has_pose & (d_other < d_own)

    seq = normalize(filled)
    rng = active_range(seq) if seq is not None else None
    picked = (rng[0] + sample_indices(rng[1] - rng[0])) if rng else np.array([], int)
    return dict(frames=frames, fps=fps, raw=filled, seen=seen, interp=interp, swap=swap,
                rng=rng, picked=set(picked.tolist()), T=T)


def draw_skeleton(img, part, connections, color):
    if not part.any():
        return
    h, w = img.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in part.reshape(-1, 3)]
    for a, b in connections:
        cv2.line(img, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(img, p, 3, color, -1)


def put_text(img, xy, text, size=16, color=(0, 0, 0)):
    """Viết chữ có dấu lên ảnh OpenCV (qua PIL). color theo thứ tự RGB."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, font=ImageFont.truetype("arial.ttf", size), fill=color)
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def render_body(info, i, scale=SCALE):
    """Video có landmark (phóng to scale lần) + thanh thời gian bên dưới."""
    i = min(i, info["T"] - 1)
    vec = info["raw"][i]
    img = cv2.resize(info["frames"][i], None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    draw_skeleton(img, vec[2 * HAND_DIM:], POSE_CONNECTIONS, POSE_COLOR)
    for side, start, color in (("L", 0, LEFT_COLOR), ("R", HAND_DIM, RIGHT_COLOR)):
        c = FILLED_COLOR if info["interp"][side][i] else (SWAP_COLOR if info["swap"][side][i] else color)
        draw_skeleton(img, vec[start:start + HAND_DIM], HAND_CONNECTIONS, c)
    W = img.shape[1]

    # Thanh thời gian: 3 hàng (Trái, Phải, Giữ lại)
    T, label_w = info["T"], 70
    strip = np.full((STRIP_H * 3 + 8, W, 3), 30, np.uint8)
    col_w = (W - label_w) / T
    for t in range(T):
        x0, x1 = int(label_w + t * col_w), int(label_w + (t + 1) * col_w)
        for row, (side, color) in enumerate((("L", LEFT_COLOR), ("R", RIGHT_COLOR))):
            c = (SWAP_COLOR if info["swap"][side][t] else color) if info["seen"][side][t] else                 (FILLED_COLOR if info["interp"][side][t] else (60, 60, 60))
            cv2.rectangle(strip, (x0, row * STRIP_H + 2), (x1 - 1, (row + 1) * STRIP_H), c, -1)
        if info["rng"] and info["rng"][0] <= t < info["rng"][1]:
            cv2.rectangle(strip, (x0, 2 * STRIP_H + 2), (x1 - 1, 3 * STRIP_H), (60, 160, 60), -1)
        if t in info["picked"]:
            cv2.line(strip, ((x0 + x1) // 2, 2 * STRIP_H + 2), ((x0 + x1) // 2, 3 * STRIP_H), (255, 255, 255), 1)
    cx = int(label_w + (i + 0.5) * col_w)
    cv2.line(strip, (cx, 0), (cx, strip.shape[0] - 1), (0, 255, 255), 2)
    for row, name in enumerate(("Trái", "Phải", "Giữ lại")):
        put_text(strip, (4, row * STRIP_H), name, size=12, color=(230, 230, 230))
    return np.vstack([img, strip])


def render(info, i, title, note):
    """Ghép: tiêu đề + ghi chú, video có landmark, thanh thời gian."""
    body = render_body(info, i)
    header = np.full((HEADER_H, body.shape[1], 3), 255, np.uint8)
    put_text(header, (8, 4), f"{title}   frame {i + 1}/{info['T']}")
    for k, line in enumerate(textwrap.wrap(note, 80)[:2]):   # ghi chú tối đa 2 dòng
        put_text(header, (8, 26 + k * 20), line, color=(180, 0, 0))
    return np.vstack([header, body])


def snapshot(items, out_path, n=8):
    """Lưu ảnh: mỗi video 1 hàng gồm n frame đã vẽ landmark (để xem nhanh hoặc dán vào báo cáo)."""
    rows = []
    for (split, word, vid), note in items:
        info = load(split, word, vid)
        idx = np.linspace(0, info["T"] - 1, n).round().astype(int)
        tiles = [cv2.resize(render(info, i, f"{word} · {split}/{vid}", note), None, fx=0.4, fy=0.4) for i in idx]
        rows.append(np.hstack(tiles))
    ok, buf = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    buf.tofile(str(out_path))   # tofile: cv2.imwrite trên Windows lỗi với đường dẫn có dấu
    print("Đã lưu", out_path)


def play(items):
    k, speed = 0, 1.0
    while 0 <= k < len(items):
        (split, word, vid), note = items[k]
        info = load(split, word, vid)
        title = f"[{k + 1}/{len(items)}] {word} · {split}/{vid} · {info['fps']:.0f} fps"
        i, paused, action = 0, False, None
        while action is None:
            cv2.imshow("VSign viewer", render(info, i, title, note))
            key = cv2.waitKeyEx(0 if paused else max(1, int(1000 / info["fps"] / speed)))
            ch = chr(key).lower() if 0 <= key < 256 else ""
            closed = cv2.getWindowProperty("VSign viewer", cv2.WND_PROP_VISIBLE) < 1   # bấm nút X
            if key == 27 or ch == "q" or closed:
                action = "quit"
            elif ch == "n":
                action = "next"
            elif ch == "p":
                action = "prev"
            elif ch == " ":
                paused = not paused
            elif ch in ("+", "="):   # không viết ch in "+=": chuỗi rỗng "" (không bấm phím) cũng "nằm trong" mọi chuỗi
                speed = min(speed * 1.5, 8)
            elif ch == "-":
                speed = max(speed / 1.5, 0.125)
            elif ch == "a" or key == 2424832:     # 2424832 = phím mũi tên trái trên Windows
                paused, i = True, max(i - 1, 0)
            elif ch == "d" or key == 2555904:     # mũi tên phải
                paused, i = True, min(i + 1, info["T"] - 1)
            elif not paused:
                i = (i + 1) % info["T"]           # hết video thì phát lại từ đầu
        if action == "quit":
            break
        k += 1 if action == "next" else -1
        k = max(k, 0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*", help='"Từ" hoặc "split/Từ/id"')
    parser.add_argument("--list", help="file danh sách: mỗi dòng split/từ/id | ghi chú")
    parser.add_argument("--snapshot", help="lưu ảnh thay vì mở cửa sổ")
    args = parser.parse_args()

    items = read_list(args.list) if args.list else []
    for t in args.targets:
        items += [(v, "") for v in find_videos(t)]
    if not items:
        sys.exit("Không có video nào. Kiểm tra lại tên từ (phải đúng tên thư mục, có dấu).")
    snapshot(items, args.snapshot) if args.snapshot else play(items)


if __name__ == "__main__":
    main()
