"""Kiểm tra chất lượng dữ liệu đã trích (data/raw) để hỗ trợ việc lọc video bằng mắt.

Chạy: python -m src.review.check_dataset   (sau khi đã chạy extract_dataset)

Ghi ra data/reports/:
  videos.csv      mỗi video 1 dòng: số frame, FPS, tỉ lệ frame thấy tay/thân, tay nào cử động nhiều hơn,
                  chất lượng trong đoạn ký hiệu (thấy tay bao nhiêu, mất tay lâu nhất, gán nhầm trái/phải)
  words.csv       mỗi từ 1 dòng: số video, số video nghi trùng, tỉ lệ thấy tay trung bình
  duplicates.csv  các cặp nghi là cùng 1 video (bản copy / augmentation): cần xem lại bằng mắt
  suggested_exclude.txt  gợi ý loại: bản sao + video gần như không thấy tay khi ký
"""
import csv
import hashlib
from collections import defaultdict
import cv2
import numpy as np

from src.core.paths import RAW_DIR, REPORT_DIR, VIDEO_DIR
from src.core.preprocess import HAND_DIM, active_range, fill_gaps, normalize

POSE_START = 2 * HAND_DIM   # vector 225 = [tay trái 63 | tay phải 63 | thân 99]
UPPER_BODY = 25             # Pose điểm 0..24: đầu, vai, tay, hông (bỏ chân)
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16
# Cặp điểm trái <-> phải của Pose, dùng để tạo bản lật gương khi so sánh
MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
                (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)]
COMPARE_FRAMES = 30         # đưa mọi video về cùng 30 frame trước khi so sánh
# Nghi trùng khi CẢ HAI điều kiện đúng (chọn bằng cách xem tay các cặp, xem báo cáo):
POSE_THRESHOLD = 0.06       # chuyển động thân trên lệch < 0.06 độ rộng vai. Chỉ điều kiện này thì
                            # chưa đủ: 2 người khác nhau ký cùng 1 từ cũng có thể lệch ~0.06
IMAGE_THRESHOLD = 0.8       # hình ảnh tương quan > 0.8: cùng người, cùng áo, cùng khung hình
SIG_FRAMES, SIG_SIZE = 8, 24   # chữ ký hình ảnh: 8 frame, thu nhỏ 24x24 ảnh xám
MIN_ACTIVE_HAND = 0.3       # trong đoạn ký hiệu thấy tay < 30% số frame -> gợi ý loại


def pose_xy(raw):
    """(T, 225) -> (T, 25, 2): x, y của thân trên."""
    return raw[:, POSE_START:].reshape(-1, 33, 3)[:, :UPPER_BODY, :2]


def normalize_pose(xy):
    """Dời gốc về giữa hai vai, chia cho độ rộng vai -> không phụ thuộc người đứng gần/xa, lệch trái/phải."""
    center = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2
    width = np.linalg.norm(xy[:, L_SHOULDER] - xy[:, R_SHOULDER], axis=1)
    width = np.maximum(width, 1e-6)
    return (xy - center[:, None]) / width[:, None, None]


def resample(seq, n):
    """Nội suy tuyến tính theo thời gian: (T, ...) -> (n, ...)."""
    t_old = np.linspace(0, 1, len(seq))
    t_new = np.linspace(0, 1, n)
    flat = seq.reshape(len(seq), -1)
    out = np.stack([np.interp(t_new, t_old, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape((n,) + seq.shape[1:])


def mirror(xy):
    """Bản lật gương của chuỗi Pose đã chuẩn hoá: đổi dấu x và đổi chỗ điểm trái <-> phải."""
    m = xy.copy()
    m[..., 0] *= -1
    for a, b in MIRROR_PAIRS:
        m[:, [a, b]] = m[:, [b, a]]
    return m


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def image_signature(video):
    """Đọc video -> (fps, số frame trống, chữ ký (SIG_FRAMES, SIG_SIZE*SIG_SIZE)): 8 frame cách đều, thu nhỏ, ảnh xám.

    Mỗi frame được trừ trung bình, chia độ lệch chuẩn -> tích vô hướng 2 chữ ký = hệ số tương quan,
    không bị ảnh hưởng khi augmentation đổi độ sáng / độ tương phản.
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, blank = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        blank += int(frame.std() < 5)   # frame gần như 1 màu (độn thêm cho đủ độ dài)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(gray, (SIG_SIZE, SIG_SIZE), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        return fps, blank, None
    idx = np.linspace(0, len(frames) - 1, SIG_FRAMES).round().astype(int)
    sig = np.stack([frames[k] for k in idx]).astype(np.float32)
    sig = (sig - sig.mean(axis=(1, 2), keepdims=True)) / (sig.std(axis=(1, 2), keepdims=True) + 1e-6)
    return fps, blank, sig


def correlation(sig_a, sig_b):
    return float((sig_a * sig_b).mean())


def quality(raw):
    """Chất lượng trong đoạn ký hiệu (sau trim_idle): (tỉ lệ thấy tay, số frame mất tay liên tiếp
    dài nhất, tỉ lệ frame tay bị gán nhầm trái/phải).

    Gán nhầm: cổ tay của bàn tay nhãn "Left" lại gần cổ tay PHẢI của Pose hơn cổ tay trái (và ngược lại).
    """
    filled = fill_gaps(raw)
    seq = normalize(filled) if len(raw) else None
    rng = active_range(seq) if seq is not None else None
    if rng is None:
        return 0.0, len(raw), 0.0
    a, b = rng
    has_hand = filled[a:b, :POSE_START].any(axis=1)
    gap = longest = 0
    for h in has_hand:
        gap = 0 if h else gap + 1
        longest = max(longest, gap)

    pose = raw[:, POSE_START:].reshape(len(raw), 33, 3)[:, :, :2]
    has_pose = raw[:, POSE_START:].any(axis=1)
    wrong = total = 0
    for start, own, other in ((0, L_WRIST, R_WRIST), (HAND_DIM, R_WRIST, L_WRIST)):
        k = raw[:, start:start + HAND_DIM].any(axis=1) & has_pose
        wrist = raw[k, start:start + 2]
        wrong += int((np.linalg.norm(wrist - pose[k, other], axis=1)
                      < np.linalg.norm(wrist - pose[k, own], axis=1)).sum())
        total += int(k.sum())
    return float(has_hand.mean()), longest, wrong / max(total, 1)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows, seqs, sigs = [], [], {}
    for npy in sorted(RAW_DIR.glob("*/*/*.npy")):
        split, word = npy.parent.parent.name, npy.parent.name
        video = VIDEO_DIR / split / word / (npy.stem + ".mp4")
        raw = np.load(npy)
        fps, blank, sigs[len(rows)] = image_signature(video)

        left = raw[:, :HAND_DIM].any(axis=1)
        right = raw[:, HAND_DIM:POSE_START].any(axis=1)
        body = raw[:, POSE_START:].any(axis=1)

        # Tỉ lệ cử động cổ tay phải / tổng 2 cổ tay (theo Pose): > 0.65 ký chủ yếu tay phải,
        # < 0.35 chủ yếu tay trái (người thuận tay trái, hoặc video bị lật gương -> mở ra xem),
        # ở giữa = ký hiệu 2 tay (vd. Cà phê), không kết luận được gì.
        dominant, right_share = "?", ""
        if body.sum() >= 5:
            xy = normalize_pose(pose_xy(raw[body]))
            move_l = xy[:, L_WRIST].std(axis=0).sum()
            move_r = xy[:, R_WRIST].std(axis=0).sum()
            right_share = move_r / max(move_l + move_r, 1e-6)
            dominant = "right" if right_share > 0.65 else ("left" if right_share < 0.35 else "both")
            right_share = round(float(right_share), 2)
            seqs.append((len(rows), resample(xy, COMPARE_FRAMES)))

        rows.append({
            "split": split, "word": word, "id": npy.stem,
            "frames": len(raw), "blank_frames": blank, "fps": round(fps, 2),
            "seconds": round(len(raw) / fps, 2) if fps else "",
            "left_rate": round(left.mean(), 2) if len(raw) else 0,
            "right_rate": round(right.mean(), 2) if len(raw) else 0,
            "any_hand_rate": round((left | right).mean(), 2) if len(raw) else 0,
            "pose_rate": round(body.mean(), 2) if len(raw) else 0,
            "dominant_hand": dominant, "right_share": right_share,
            **dict(zip(("active_hand_rate", "max_gap", "label_swap_rate"),
                       (round(v, 2) for v in quality(raw)))),
            "md5": md5(video),
        })

    # So từng cặp video cùng từ (cả train lẫn test): chuyển động (Pose) và hình ảnh.
    # Có so cả với bản lật gương, phòng trường hợp augmentation bằng cách lật ảnh.
    by_word = defaultdict(list)
    for idx, seq in seqs:
        by_word[rows[idx]["word"]].append((idx, seq))
    dups = []
    for word, items in by_word.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (a, sa), (b, sb) = items[i], items[j]
                d = np.linalg.norm(sa - sb, axis=-1).mean()
                d_mirror = np.linalg.norm(sa - mirror(sb), axis=-1).mean()
                if min(d, d_mirror) >= POSE_THRESHOLD and rows[a]["md5"] != rows[b]["md5"]:
                    continue   # chuyển động khác hẳn -> khỏi so hình (tiết kiệm thời gian)
                img = correlation(sigs[a], sigs[b])
                img_mirror = correlation(sigs[a], sigs[b][:, :, ::-1])
                same_file = rows[a]["md5"] == rows[b]["md5"]
                normal = d < POSE_THRESHOLD and img > IMAGE_THRESHOLD
                mirrored = d_mirror < POSE_THRESHOLD and img_mirror > IMAGE_THRESHOLD
                if same_file or normal or mirrored:
                    dups.append({
                        "word": word,
                        "video_a": f'{rows[a]["split"]}/{rows[a]["id"]}',
                        "video_b": f'{rows[b]["split"]}/{rows[b]["id"]}',
                        "pose_distance": round(float(d), 4),
                        "image_corr": round(img, 3),
                        "kind": "same_file" if same_file else ("normal" if normal else "mirror"),
                        "train_test_leak": rows[a]["split"] != rows[b]["split"],
                    })

    dup_count = defaultdict(set)
    for d in dups:
        dup_count[d["word"]].update([d["video_a"], d["video_b"]])

    # Gom các cặp thành nhóm (A trùng B, B trùng C -> A, B, C cùng nhóm), mỗi nhóm giữ 1 video.
    # Ưu tiên giữ bản ở train, loại các bản còn lại -> test không còn bản sao của train.
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for d in dups:
        parent[find((d["word"], d["video_a"]))] = find((d["word"], d["video_b"]))
    groups = defaultdict(list)
    for key in parent:
        groups[find(key)].append(key)
    suggested = []
    for members in groups.values():
        members.sort(key=lambda k: (not k[1].startswith("train"), k[1]))
        for word, video in members[1:]:
            split, vid = video.split("/")
            suggested.append(f"{split}/{word}/{vid}")
    no_hand = [f'{r["split"]}/{r["word"]}/{r["id"]}' for r in rows if r["active_hand_rate"] < MIN_ACTIVE_HAND]
    with open(REPORT_DIR / "suggested_exclude.txt", "w", encoding="utf-8") as f:
        f.write("# Gợi ý loại. Xem lại bằng mắt (src/review/view_video.py) rồi chép dòng đồng ý sang data/exclude.txt\n")
        f.write("# --- Nghi là bản copy / augmentation của video khác ---\n")
        f.writelines(line + "\n" for line in sorted(suggested))
        f.write(f"# --- Trong đoạn ký hiệu thấy tay < {MIN_ACTIVE_HAND:.0%} số frame ---\n")
        f.writelines(line + "\n" for line in sorted(set(no_hand) - set(suggested)))

    words = []
    for word in sorted({r["word"] for r in rows}):
        rs = [r for r in rows if r["word"] == word]
        words.append({
            "word": word,
            "train": sum(r["split"] == "train" for r in rs),
            "test": sum(r["split"] == "test" for r in rs),
            "videos_in_dup_pairs": len(dup_count[word]),
            "leak_pairs": sum(d["train_test_leak"] for d in dups if d["word"] == word),
            "mean_hand_rate": round(float(np.mean([r["any_hand_rate"] for r in rs])), 2),
            "low_hand_videos": sum(r["any_hand_rate"] < 0.5 for r in rs),
            "videos_with_blank": sum(r["blank_frames"] > 0 for r in rs),
            "no_hand_videos": sum(r["active_hand_rate"] < MIN_ACTIVE_HAND for r in rs),
            "right_hand": sum(r["dominant_hand"] == "right" for r in rs),
            "both_hands": sum(r["dominant_hand"] == "both" for r in rs),
            "left_hand": sum(r["dominant_hand"] == "left" for r in rs),
            "frames_min": min(r["frames"] for r in rs),
            "frames_max": max(r["frames"] for r in rs),
        })

    for name, data in (("videos.csv", rows), ("words.csv", words), ("duplicates.csv", dups)):
        if not data:
            continue
        with open(REPORT_DIR / name, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: Excel đọc đúng dấu
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    print(f"{len(rows)} video, {len(words)} từ, {len(dups)} cặp nghi trùng "
          f"({sum(d['train_test_leak'] for d in dups)} cặp lẫn giữa train và test), "
          f"{len(groups)} nhóm -> gợi ý loại {len(suggested)} video; "
          f"{len(no_hand)} video gần như không thấy tay khi ký")
    print("Báo cáo ở", REPORT_DIR)


if __name__ == "__main__":
    main()
