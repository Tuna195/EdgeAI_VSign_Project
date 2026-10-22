"""Duyệt video bằng mắt, bấm nút để quyết định giữ / loại. Quyết định lưu ngay, tắt giữa chừng vẫn giữ.

Chạy (sau check_dataset.py):
  python src/review_tool.py copy     cặp nghi bản sao: 2 video phát cạnh nhau
  python src/review_tool.py blank    video có frame trống (vd. train/Chào/140605)
  python src/review_tool.py swap     video bị gán nhầm trái/phải nhiều
  python src/review_tool.py gap      video mất tay liên tiếp lâu trong đoạn ký hiệu
  python src/review_tool.py nohand   video gần như không thấy tay (đã TỰ ĐỘNG loại, xem lại nếu muốn)
  python src/review_tool.py          in tóm tắt: mỗi nhóm đã duyệt bao nhiêu, mỗi từ còn bao nhiêu video

Mở lại sẽ nhảy tới mục đầu tiên chưa duyệt. Bấm nút bằng chuột hoặc phím 1/2/3.
Phím khác: Space dừng/chạy   A/D lùi/tiến 1 frame   N/P mục sau/trước   Q thoát

Kết quả: data/reports/review_decisions.csv (mọi quyết định) và data/exclude.txt (tạo lại sau mỗi lần bấm,
preprocess.py đọc file này). Đừng sửa tay exclude.txt: lần bấm sau sẽ ghi đè.
"""
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from view_video import load, put_text, render_body

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "data" / "reports"
DECISIONS_FILE = REPORT_DIR / "review_decisions.csv"
EXCLUDE_FILE = ROOT / "data" / "exclude.txt"

NO_HAND = 0.3       # active_hand_rate < 0.3 -> tự động loại (giống MIN_ACTIVE_HAND ở check_dataset.py)
SWAP_RATE = 0.2     # >= 20% frame có tay bị gán nhầm trái/phải -> đưa vào nhóm swap
LONG_GAP = 8        # mất tay >= 8 frame liền trong đoạn ký hiệu -> đưa vào nhóm gap

QUEUES = {
    "copy": "Nghi bản sao: cùng 1 video (copy) hay quay lần 2?",
    "blank": "Có frame trống ở đầu/cuối",
    "swap": "Gán nhầm trái/phải nhiều (khung tay màu đỏ)",
    "gap": "Mất tay nhiều frame liền khi đang ký (vùng tối trong đoạn Giữ lại)",
    "nohand": "Gần như không thấy tay khi ký (đã tự động loại)",
}
PAIR_BUTTONS = [("1  Loại video TRÁI", "loai_a"), ("2  Loại video PHẢI", "loai_b"),
                ("3  Quay lần 2: giữ cả hai", "giu")]
SINGLE_BUTTONS = [("1  Giữ", "giu"), ("2  Loại", "loai"), ("3  Loại nếu đủ mẫu", "neu_du")]
LABELS = {"loai_a": "Loại video trái", "loai_b": "Loại video phải", "giu": "Giữ",
          "loai": "Loại", "neu_du": "Loại nếu đủ mẫu", "auto_loai": "Tự động loại"}
WIN = "VSign review"


def read_csv(name):
    with open(REPORT_DIR / name, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def as_video(word, split_id):
    """'train/012345' (cách duplicates.csv ghi) -> ('train', từ, '012345')."""
    split, vid = split_id.split("/")
    return split, word, vid


def build_items(queue):
    """Danh sách mục cần duyệt. Mỗi mục: key (duy nhất), videos [(split, từ, id)], note."""
    if queue == "copy":
        return [{"key": f'copy|{r["word"]}|{r["video_a"]}|{r["video_b"]}',
                 "videos": [as_video(r["word"], r["video_a"]), as_video(r["word"], r["video_b"])],
                 "note": f'chuyển động lệch {float(r["pose_distance"]):.3f} · ảnh giống {float(r["image_corr"]):.2f}'
                         + (" · 1 bên train, 1 bên test" if r["train_test_leak"] == "True" else "")}
                for r in read_csv("duplicates.csv")]

    rows = read_csv("videos.csv")
    pick = {
        "blank": lambda r: int(r["blank_frames"]) > 0,
        "swap": lambda r: float(r["label_swap_rate"]) >= SWAP_RATE and float(r["active_hand_rate"]) >= NO_HAND,
        "gap": lambda r: int(r["max_gap"]) >= LONG_GAP and float(r["active_hand_rate"]) >= NO_HAND,
        "nohand": lambda r: float(r["active_hand_rate"]) < NO_HAND,
    }[queue]
    return [{"key": f'{queue}|{r["split"]}/{r["word"]}/{r["id"]}', "videos": [(r["split"], r["word"], r["id"])],
             "note": f'{r["blank_frames"]} frame trống · thấy tay {float(r["active_hand_rate"]):.0%} khi ký · '
                     f'mất tay lâu nhất {r["max_gap"]} frame · gán nhầm {float(r["label_swap_rate"]):.0%}'}
            for r in rows if pick(r)]


def load_decisions():
    if not DECISIONS_FILE.exists():
        return {}
    with open(DECISIONS_FILE, encoding="utf-8-sig") as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def excluded_videos(item, decision):
    """Quyết định -> các video bị loại, dạng 'split/từ/id'."""
    v = ["/".join(x) for x in item["videos"]]
    return {"loai_a": [v[0]], "loai_b": v[1:2], "loai": v, "auto_loai": v}.get(decision, [])


def save(decisions):
    """Ghi review_decisions.csv và tạo lại data/exclude.txt từ đầu."""
    with open(DECISIONS_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["key", "queue", "decision", "excluded", "time"])
        w.writeheader()
        w.writerows(decisions.values())

    by_queue = {}
    for d in decisions.values():
        for v in filter(None, d["excluded"].split(";")):
            by_queue.setdefault(d["queue"], set()).add(v)
    with open(EXCLUDE_FILE, "w", encoding="utf-8") as f:
        f.write("# Tạo tự động bởi src/review_tool.py từ data/reports/review_decisions.csv. Đừng sửa tay.\n")
        for q, videos in by_queue.items():
            f.write(f"# --- {QUEUES[q]} ---\n")
            f.writelines(v + "\n" for v in sorted(videos))


def seed_auto(decisions):
    """Lần đầu: video gần như không thấy tay -> ghi sẵn quyết định 'tự động loại' (xem lại ở nhóm nohand)."""
    changed = False
    for item in build_items("nohand"):
        if item["key"] not in decisions:
            decisions[item["key"]] = {"key": item["key"], "queue": "nohand", "decision": "auto_loai",
                                      "excluded": ";".join(excluded_videos(item, "auto_loai")), "time": ""}
            changed = True
    if changed:
        save(decisions)


def remaining_counts(decisions):
    """Mỗi từ còn bao nhiêu video (train + test) sau khi trừ các video đã loại."""
    excluded = {v for d in decisions.values() for v in filter(None, d["excluded"].split(";"))}
    total = Counter(r["word"] for r in read_csv("videos.csv"))
    gone = Counter(v.split("/")[1] for v in excluded)
    return {w: (n - gone[w], n) for w, n in total.items()}


def draw_buttons(canvas, buttons, current, y):
    """Vẽ nút, trả về [(x0, y0, x1, y1, decision)] để bắt sự kiện chuột."""
    rects, w = [], canvas.shape[1] // len(buttons)
    for k, (label, decision) in enumerate(buttons):
        x0, x1 = k * w + 8, (k + 1) * w - 8
        chosen = decision == current
        cv2.rectangle(canvas, (x0, y), (x1, y + 44), (60, 160, 60) if chosen else (90, 90, 90), -1)
        put_text(canvas, (x0 + 12, y + 12), label, size=17, color=(255, 255, 255))
        rects.append((x0, y, x1, y + 44, decision))
    return rects


def review(queue):
    items = build_items(queue)
    decisions = load_decisions()
    if not items:
        sys.exit("Nhóm này không có video nào.")
    buttons = PAIR_BUTTONS if queue == "copy" else SINGLE_BUTTONS
    scale = 2 if queue == "copy" else 3
    clicked = {"decision": None, "rects": []}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            for x0, y0, x1, y1, decision in clicked["rects"]:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    clicked["decision"] = decision

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    # Mở lại thì bắt đầu ở mục đầu tiên chưa duyệt (nhóm nohand: luôn từ đầu)
    k = next((n for n, it in enumerate(items) if it["key"] not in decisions), 0)
    while 0 <= k < len(items):
        item = items[k]
        infos = [load(*v) for v in item["videos"]]
        fps = min(info["fps"] for info in infos)
        counts = remaining_counts(decisions)
        # Video đã bị loại ở nhóm KHÁC (1 video có thể nằm ở nhiều nhóm, vd. vừa gán nhầm vừa mất tay)
        gone_elsewhere = {v: d["queue"] for d in decisions.values() if d["key"] != item["key"]
                          for v in filter(None, d["excluded"].split(";"))}
        i, paused, move = 0, False, 0
        while move == 0:
            bodies = []
            t_max = max(info["T"] for info in infos)
            for (split, word, vid), info in zip(item["videos"], infos):
                # 2 video dài khác nhau: kéo giãn theo video dài hơn để cùng bắt đầu, cùng kết thúc
                body = render_body(info, (i % t_max) * info["T"] // t_max, scale)
                label = np.full((30, body.shape[1], 3), 235, np.uint8)
                put_text(label, (6, 6), f"{word} · {split}/{vid} · {info['T']} frame")
                other = gone_elsewhere.get(f"{split}/{word}/{vid}")
                if other:
                    put_text(label, (body.shape[1] - 190, 6), f"ĐÃ LOẠI ở nhóm {other}", color=(200, 0, 0))
                bodies.append(np.vstack([label, body]))
            gap = np.full((bodies[0].shape[0], 12, 3), 255, np.uint8)
            middle = np.hstack([x for b in bodies for x in (b, gap)][:-1])

            current = decisions.get(item["key"], {}).get("decision")
            word = item["videos"][0][1]
            header = np.full((76, middle.shape[1], 3), 255, np.uint8)
            put_text(header, (8, 4), f"[{k + 1}/{len(items)}] {QUEUES[queue]}")
            put_text(header, (8, 28), item["note"], color=(180, 0, 0))
            put_text(header, (8, 52), f"Từ '{word}' còn {counts[word][0]}/{counts[word][1]} video sau khi loại"
                     + (f"   ·   Đã chọn: {LABELS[current]}" if current else "   ·   Chưa duyệt"),
                     color=(0, 100, 0) if current else (90, 90, 90))
            footer = np.full((60, middle.shape[1], 3), 255, np.uint8)
            clicked["rects"] = [(x0, y0 + header.shape[0] + middle.shape[0], x1, y1 + header.shape[0] + middle.shape[0], d)
                                for x0, y0, x1, y1, d in draw_buttons(footer, buttons, current, 8)]
            cv2.imshow(WIN, np.vstack([header, middle, footer]))

            key = cv2.waitKeyEx(0 if paused else max(1, int(1000 / fps)))
            ch = chr(key).lower() if 0 <= key < 256 else ""
            if ch in ("1", "2", "3"):
                clicked["decision"] = buttons[int(ch) - 1][1]
            if clicked["decision"]:
                d = clicked["decision"]
                clicked["decision"] = None
                decisions[item["key"]] = {"key": item["key"], "queue": queue, "decision": d,
                                          "excluded": ";".join(excluded_videos(item, d)),
                                          "time": time.strftime("%Y-%m-%d %H:%M")}
                save(decisions)
                move = 1   # quyết định xong -> sang mục tiếp theo
            elif key == 27 or ch == "q" or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                move = 99
            elif ch == "n":
                move = 1
            elif ch == "p":
                move = -1
            elif ch == " ":
                paused = not paused
            elif ch == "a" or key == 2424832:     # mũi tên trái
                paused, i = True, max(i - 1, 0)
            elif ch == "d" or key == 2555904:     # mũi tên phải
                paused, i = True, i + 1
            elif not paused:
                i += 1
        if move == 99:
            break
        k = max(k + move, 0)
    cv2.destroyAllWindows()
    summary()


def summary():
    decisions = load_decisions()
    seed_auto(decisions)
    print("Nhóm      đã duyệt / tổng")
    for q in QUEUES:
        items = build_items(q)
        done = sum(it["key"] in decisions for it in items)
        print(f"  {q:8s} {done:4d} / {len(items)}")
    excluded = {v for d in decisions.values() for v in filter(None, d["excluded"].split(";"))}
    maybe = [d["key"].split("|", 1)[1] for d in decisions.values() if d["decision"] == "neu_du"]
    print(f"Đang loại {len(excluded)} video (data/exclude.txt)")
    if maybe:
        print("Loại nếu đủ mẫu (chưa loại):", ", ".join(maybe))
    low = sorted((left, total, w) for w, (left, total) in remaining_counts(decisions).items() if left < 30)
    if low:
        print("Từ còn ít video:", ", ".join(f"{w} {left}/{total}" for left, total, w in low))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in QUEUES:
        d = load_decisions()
        seed_auto(d)
        review(sys.argv[1])
    elif len(sys.argv) > 1:
        sys.exit(f"Nhóm không hợp lệ. Chọn một trong: {', '.join(QUEUES)}")
    else:
        summary()
