"""Gắn nhãn người ký cho từng video, làm theo TỪNG TỪ (mỗi từ có thể do 1 nhóm người khác nhau ký).
Dùng để chia lại train/test theo người ký: cùng 1 người không được vừa ở train vừa ở test của cùng 1 từ.

Chạy: python src/label_signers.py              mở từ đầu tiên chưa xác nhận
      python src/label_signers.py "Bố"         mở 1 từ
      python src/label_signers.py --summary    tóm tắt: số người ký mỗi từ, gợi ý tự động đúng bao nhiêu

Mỗi từ hiện 1 lưới ảnh, mỗi ô 1 video (frame đầu tiên thấy người: lúc chưa ký nên áo không bị tay che).
Tool tự gợi ý nhóm theo màu áo (S1, S2, ...). Bạn sửa lại cho đúng NGƯỜI:
  chuột phải vào ô   lấy nhãn của ô đó làm "nhãn đang dùng"
  chuột trái vào ô   gán "nhãn đang dùng" cho ô đó
  M                  nhãn mới (chưa ô nào dùng) làm nhãn đang dùng
  V                  phát video ở ô đang trỏ chuột (phím bất kỳ để đóng)
  S                  xếp lại các ô theo nhãn
  Enter              xác nhận từ này, sang từ sau      N/P  từ sau/trước (chưa xác nhận)      Q  thoát
Cùng 1 người mặc áo khác -> gán cùng nhãn. Nhãn chỉ có nghĩa trong 1 từ (S1 của "Bố" khác S1 của "Chạy").

Kết quả lưu ngay sau mỗi lần sửa: data/reports/signers.csv (split, word, id, signer, suggested, done).
Cột suggested giữ gợi ý ban đầu để đo độ chính xác của gợi ý (--summary).
"""
import csv
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from landmarks import HAND_DIM
from preprocess import usable_videos
from view_video import put_text

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "videos"
RAW_DIR = ROOT / "data" / "raw"
SIGNERS_FILE = ROOT / "data" / "reports" / "signers.csv"
FIELDS = ["split", "word", "id", "signer", "suggested", "done"]

TILE, COLS, LABEL_H, GAP, HEADER_H = 128, 10, 18, 6, 70
GROUP_THRESHOLD = 0.35   # 2 nhóm có khoảng cách màu áo trung bình < 0.35 thì gộp (Bhattacharyya, 0 = giống hệt)
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40),
           (128, 0, 0), (170, 255, 195), (128, 128, 0), (0, 0, 128)]   # RGB, mỗi nhãn 1 màu
WIN = "VSign signers"


def color(label):
    r, g, b = PALETTE[(int(label[1:]) - 1) % len(PALETTE)]
    return (b, g, r)


def thumbnail_and_hist(key):
    """Ảnh đại diện + histogram màu áo (HSV) của 1 video, lấy ở 3 frame đầu tiên thấy người."""
    split, word, vid = key.split("/")
    raw = np.load(RAW_DIR / split / word / f"{vid}.npy")
    pose = raw[:, 2 * HAND_DIM:].reshape(len(raw), 33, 3)
    picked = list(np.flatnonzero(raw[:, 2 * HAND_DIM:].any(axis=1))[:3]) or [0]

    cap = cv2.VideoCapture(str(VIDEO_DIR / split / word / f"{vid}.mp4"))
    frames = []
    while len(frames) <= picked[-1]:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    picked = [k for k in picked if k < len(frames)] or [len(frames) - 1]

    hist = np.zeros((8, 4, 4), np.float32)
    for k in picked:
        img = frames[k]
        h, w = img.shape[:2]
        p = pose[k]
        if p.any():
            # Thân áo: giữa 2 vai theo chiều ngang (bớt 15% mỗi bên cho khỏi lấy nền), từ vai xuống hông
            x0, x1 = sorted((p[L_SH, 0], p[R_SH, 0]))
            pad = (x1 - x0) * 0.15
            y0, y1 = (p[L_SH, 1] + p[R_SH, 1]) / 2, (p[L_HIP, 1] + p[R_HIP, 1]) / 2
            box = [int(np.clip(v, 0, 1) * s) for v, s in ((x0 + pad, w), (y0, h), (x1 - pad, w), (y1, h))]
        else:
            box = [w // 3, h // 3, 2 * w // 3, 2 * h // 3]
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            box = [w // 3, h // 3, 2 * w // 3, 2 * h // 3]
        crop = cv2.cvtColor(img[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2HSV)
        hist += cv2.calcHist([crop], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    hist /= max(hist.sum(), 1)
    return cv2.resize(frames[picked[0]], (TILE, TILE)), hist


def suggest_groups(hists):
    """Gom nhóm kiểu average linkage: lặp lại việc gộp 2 nhóm gần nhau nhất cho tới khi khoảng cách > ngưỡng.
    Trả về nhãn 'S1', 'S2', ... cho từng video (S1 = nhóm đông nhất)."""
    n = len(hists)
    dist = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        dist[i, j] = dist[j, i] = cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_BHATTACHARYYA)
    groups = [[i] for i in range(n)]
    while len(groups) > 1:
        d, a, b = min((dist[np.ix_(ga, gb)].mean(), a, b)
                      for (a, ga), (b, gb) in combinations(enumerate(groups), 2))
        if d > GROUP_THRESHOLD:
            break
        groups[a] += groups.pop(b)
    labels = [""] * n
    for g, members in enumerate(sorted(groups, key=len, reverse=True), 1):
        for i in members:
            labels[i] = f"S{g}"
    return labels


def load_rows():
    if not SIGNERS_FILE.exists():
        return {}
    with open(SIGNERS_FILE, encoding="utf-8-sig") as f:
        return {f'{r["split"]}/{r["word"]}/{r["id"]}': r for r in csv.DictReader(f)}


def save_rows(rows):
    SIGNERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SIGNERS_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))


def open_word(word, keys, rows):
    """Đọc ảnh + gợi ý cho 1 từ. Video chưa có trong signers.csv thì lấy nhãn = gợi ý."""
    thumbs, hists = zip(*(thumbnail_and_hist(k) for k in keys))
    suggested = suggest_groups(hists)
    for key, s in zip(keys, suggested):
        if key not in rows:
            split, _, vid = key.split("/")
            rows[key] = {"split": split, "word": word, "id": vid, "signer": s, "suggested": s, "done": ""}
    return list(thumbs)


def sort_order(keys, rows):
    return sorted(range(len(keys)), key=lambda i: (int(rows[keys[i]]["signer"][1:]), keys[i]))


def tile_at(x, y, n):
    """Toạ độ chuột -> vị trí ô trong lưới (hoặc None)."""
    c, r = x // (TILE + GAP), (y - HEADER_H) // (TILE + LABEL_H + GAP)
    if y < HEADER_H or c >= COLS or x % (TILE + GAP) >= TILE or (y - HEADER_H) % (TILE + LABEL_H + GAP) >= TILE + LABEL_H:
        return None
    pos = r * COLS + c
    return pos if pos < n else None


def draw(word, k, n_words, keys, thumbs, order, rows, current):
    n_rows = (len(keys) + COLS - 1) // COLS
    canvas = np.full((HEADER_H + n_rows * (TILE + LABEL_H + GAP), COLS * (TILE + GAP), 3), 255, np.uint8)
    labels = [rows[key]["signer"] for key in keys]
    done = rows[keys[0]]["done"] == "1"
    put_text(canvas, (8, 4), f"[{k + 1}/{n_words}] {word} · {len(keys)} video · {len(set(labels))} người"
             + ("   ·   ĐÃ XÁC NHẬN" if done else "   ·   chưa xác nhận (Enter)"),
             color=(0, 120, 0) if done else (90, 90, 90))
    put_text(canvas, (8, 26), "Phải: lấy nhãn · Trái: gán nhãn · M: nhãn mới · V: xem video · S: xếp lại · N/P · Q",
             size=14, color=(90, 90, 90))
    put_text(canvas, (8, 46), "Nhãn đang dùng:", size=15)
    cv2.rectangle(canvas, (125, 46), (175, 64), color(current), -1)
    cv2.putText(canvas, current, (132, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    for pos, i in enumerate(order):
        x = (pos % COLS) * (TILE + GAP)
        y = HEADER_H + (pos // COLS) * (TILE + LABEL_H + GAP)
        canvas[y:y + TILE, x:x + TILE] = thumbs[i]
        lab = labels[i]
        cv2.rectangle(canvas, (x, y), (x + TILE - 1, y + TILE - 1), color(lab), 4)
        cv2.rectangle(canvas, (x, y), (x + 38, y + 20), color(lab), -1)
        cv2.putText(canvas, lab, (x + 3, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        split, _, vid = keys[i].split("/")
        cv2.putText(canvas, f"{split[:2]} {vid}", (x + 2, y + TILE + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)
    return canvas


def play_clip(key):
    split, word, vid = key.split("/")
    cap = cv2.VideoCapture(str(VIDEO_DIR / split / word / f"{vid}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, None, fx=2, fy=2))
    cap.release()
    i = 0
    while frames:
        cv2.imshow("video", frames[i % len(frames)])
        if cv2.waitKey(max(1, int(1000 / fps))) != -1:
            break
        i += 1
    cv2.destroyWindow("video")


def label(start_word=None):
    videos, _ = usable_videos()
    words = sorted(videos)
    rows = load_rows()
    if start_word:
        if start_word not in videos:
            sys.exit(f"Không có từ '{start_word}' (sai tên, hoặc từ đã bị bỏ vì < 20 video).")
        k = words.index(start_word)
    else:
        k = next((n for n, w in enumerate(words) if not all(rows.get(key, {}).get("done") == "1" for key in videos[w])), 0)

    state = {"hover": None, "click": None}

    def on_mouse(event, x, y, *_):
        state["hover"] = tile_at(x, y, len(keys))
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) and state["hover"] is not None:
            state["click"] = ("left" if event == cv2.EVENT_LBUTTONDOWN else "right", state["hover"])

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)   # cửa sổ kéo giãn được, OpenCV tự đổi toạ độ chuột về ảnh gốc
    cv2.setMouseCallback(WIN, on_mouse)
    while 0 <= k < len(words):
        word, keys = words[k], videos[words[k]]
        thumbs = open_word(word, keys, rows)
        save_rows(rows)
        order = sort_order(keys, rows)
        current = rows[keys[order[0]]]["signer"]
        move = 0
        while move == 0:
            cv2.imshow(WIN, draw(word, k, len(words), keys, thumbs, order, rows, current))
            key = cv2.waitKeyEx(30)
            ch = chr(key).lower() if 0 <= key < 256 else ""
            if state["click"]:
                button, pos = state["click"]
                state["click"] = None
                row = rows[keys[order[pos]]]
                if button == "right":
                    current = row["signer"]
                elif row["signer"] != current:
                    row["signer"] = current
                    save_rows(rows)
            elif key in (13, 10):        # Enter: xác nhận từ này
                for key_ in keys:
                    rows[key_]["done"] = "1"
                save_rows(rows)
                move = 1
            elif ch == "m":
                used = {int(rows[key_]["signer"][1:]) for key_ in keys}
                current = f"S{next(n for n in range(1, len(keys) + 2) if n not in used)}"
            elif ch == "v" and state["hover"] is not None:
                play_clip(keys[order[state["hover"]]])
            elif ch == "s":
                order = sort_order(keys, rows)
            elif ch == "n":
                move = 1
            elif ch == "p":
                move = -1
            elif ch == "q" or key == 27 or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                move = 99
        if move == 99:
            break
        k = max(k + move, 0)
    cv2.destroyAllWindows()
    summary()


def summary():
    """Số người ký mỗi từ, và gợi ý tự động đúng bao nhiêu (chỉ tính các từ đã xác nhận)."""
    videos, dropped = usable_videos()
    rows = load_rows()
    same_ok = same_all = merged_ok = merged_all = 0
    print("Từ             video  người  video mỗi người")
    for word in sorted(videos):
        rs = [rows[k] for k in videos[word] if k in rows]
        if not rs or not all(r["done"] == "1" for r in rs):
            print(f"  {word:14s} {len(videos[word]):4d}   chưa xác nhận")
            continue
        sizes = sorted((sum(r["signer"] == s for r in rs) for s in {r["signer"] for r in rs}), reverse=True)
        print(f"  {word:14s} {len(rs):4d}   {len(sizes):4d}   {sizes}")
        # Xét từng cặp video: thật sự cùng người? gợi ý có để chung nhóm không?
        for a, b in combinations(rs, 2):
            same, merged = a["signer"] == b["signer"], a["suggested"] == b["suggested"]
            same_all += same
            same_ok += same and merged
            merged_all += merged
            merged_ok += merged and same
    if same_all:
        print(f"Gợi ý tự động: {same_ok / same_all:.0%} cặp cùng người được gợi ý chung nhóm, "
              f"{merged_ok / max(merged_all, 1):.0%} cặp gợi ý chung nhóm đúng là cùng người")
    if dropped:
        print("Từ đã bỏ (< 20 video):", ", ".join(dropped))


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary()
    else:
        label(sys.argv[1] if len(sys.argv) > 1 else None)
