"""Vẽ landmark (từ file .npz của extract_landmark.py) lên video để xem tận mắt frame bị sót tay.

Chạy:
    python -m src.pi.draw --video data/videos/test_30s.mp4 \
        --npz runs/landmarks/test_30s__hI_pI_s1_d0.5.npz
    thêm --missed-only để chỉ ghi các frame bị sót tay (video ngắn, dễ xem)

Hình vẽ:
    xanh lá = tay nhãn MediaPipe 'Left' | xanh dương = 'Right' | xám = pose
    vòng vàng = cổ tay theo pose (chỗ đáng lẽ có bàn tay)
    viền đỏ + "SOT TAY" = pose thấy cổ tay trong khung, cao hơn hông, mà hand không tìm thấy tay ở gần
Đầu ra: runs/draw/<tên npz>[_missed].mp4 + bảng tỉ lệ sót tay theo cỡ bàn tay.
"""
import argparse
import json
import os

import cv2
import numpy as np

from src.pi.extract_landmark import IN_H, IN_W, POSE_INDEX, POSE_WRIST, enhance, fit_frame

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
              (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22), (0, 11), (0, 12)]
HIP = (23, 24)
NEAR_PX = 60          # cổ tay hand cách cổ tay pose < 60 px -> coi là cùng một bàn tay
VIS_MIN = 0.5         # độ chắc chắn tối thiểu của điểm cổ tay pose
SIZE_BINS = [0, 10, 15, 20, 25, 30, 1000]   # cỡ tay theo pose (px, cổ tay -> ngón trỏ)


def load_phases(video):
    path = os.path.splitext(video)[0] + ".json"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("phases", [])


def analyse(d):
    """Với mỗi frame và mỗi cổ tay pose: có 'cần thấy' không, và hand có thấy không."""
    lm = d["landmarks"]
    T = len(lm)
    pose = lm[:, 126:].reshape(T, 33, 3)
    vis = d["pose_vis"]
    hands = lm[:, :126].reshape(T, 2, 21, 3)
    has_hand = np.abs(hands).sum((2, 3)) > 0                 # (T, 2) ô trái/phải có dữ liệu
    hand_wrist = hands[:, :, 0, :2] * (IN_W, IN_H)           # điểm 0 = cổ tay
    expected = np.zeros((T, 2), bool)
    found = np.zeros((T, 2), bool)
    hip_y = pose[:, HIP, 1].mean(1)
    for k in range(2):
        w = pose[:, POSE_WRIST[k], :2]
        inside = (w[:, 0] > 0) & (w[:, 0] < 1) & (w[:, 1] > 0) & (w[:, 1] < 1)
        expected[:, k] = inside & (vis[:, POSE_WRIST[k]] > VIS_MIN) & (w[:, 1] < hip_y)
        wp = w * (IN_W, IN_H)
        dist = np.linalg.norm(hand_wrist - wp[:, None, :], axis=2)   # (T, 2 ô tay)
        found[:, k] = ((dist < NEAR_PX) & has_hand).any(1)
    return expected, found


def print_size_table(d, expected, found):
    size = d["pose_hand_px"]   # (T, 2) theo cổ tay pose
    print("\nTỉ lệ THẤY tay theo cỡ bàn tay (ước lượng từ pose: cổ tay -> ngón trỏ, px)")
    print(f"{'cỡ (px)':<12}{'số lần cần thấy':>16}{'tỉ lệ thấy':>12}")
    e, f, s = expected.ravel(), found.ravel(), size.ravel()
    for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
        m = e & (s >= lo) & (s < hi)
        if m.sum():
            print(f"{f'{lo}-{hi}' if hi < 1000 else f'>={lo}':<12}{int(m.sum()):>16}{f[m].mean():>12.2f}")
    if e.sum():
        print(f"{'tổng':<12}{int(e.sum()):>16}{f[e].mean():>12.2f}")
    hp = d["hand_px"][d["hand_px"] > 0]
    sp = size[expected]
    if hp.size and sp.size:
        print(f"(tham khảo: cỡ tay theo hand p50 {np.median(hp):.0f} px, theo pose p50 {np.median(sp):.0f} px)")


def draw_frame(img, row, d, i, exp_i, fnd_i, phase):
    lm = d["landmarks"][i]
    pose = lm[126:].reshape(33, 3)
    if np.abs(pose).sum() > 0:
        pts = (pose[:, :2] * (IN_W, IN_H)).astype(int)
        for a, b in POSE_EDGES:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), (160, 160, 160), 2)
        for k in range(2):
            cv2.circle(img, tuple(pts[POSE_WRIST[k]]), 10, (0, 220, 255), 2)
    for slot, color, name in ((0, (0, 200, 0), "L"), (1, (255, 120, 0), "R")):
        h = lm[slot * 63:(slot + 1) * 63].reshape(21, 3)
        if np.abs(h).sum() == 0:
            continue
        pts = (h[:, :2] * (IN_W, IN_H)).astype(int)
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2)
        for p in pts:
            cv2.circle(img, tuple(p), 3, color, -1)
        cv2.putText(img, f"{name} {d['hand_score'][i, slot]:.2f}", tuple(pts[0] + (5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    missed = bool((exp_i & ~fnd_i).any())
    if missed:
        cv2.rectangle(img, (0, 0), (IN_W - 1, IN_H - 1), (0, 0, 255), 8)
        cv2.putText(img, "SOT TAY", (IN_W - 150, IN_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    info = (f"frame {row}  {phase}  tay={d['n_hands'][i]}  "
            f"hand_px={d['hand_px'][i].astype(int).tolist()}  pose_px={d['pose_hand_px'][i].astype(int).tolist()}")
    cv2.rectangle(img, (0, 0), (IN_W, 26), (0, 0, 0), -1)
    cv2.putText(img, info, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return missed


def ascii_phase(label):
    # cv2.putText không vẽ được dấu tiếng Việt -> bỏ dấu
    import unicodedata
    s = unicodedata.normalize("NFD", label).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def main():
    ap = argparse.ArgumentParser(description="Vẽ landmark lên video")
    ap.add_argument("--video", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--missed-only", action="store_true", help="chỉ ghi các frame bị sót tay")
    a = ap.parse_args()

    d = dict(np.load(a.npz))
    meta = json.loads(str(d["meta"]))
    fit = meta["config"]["fit"]
    stride = meta["config"]["stride"]
    expected, found = analyse(d)
    print_size_table(d, expected, found)

    stem = os.path.splitext(os.path.basename(a.npz))[0]
    out = a.out or os.path.join("runs/draw", stem + ("_missed" if a.missed_only else "") + ".mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    phases = load_phases(a.video)
    fps_out = (meta["src_fps"] or 25) / stride
    if a.missed_only:
        fps_out = min(fps_out, 5)  # chậm lại cho dễ nhìn

    row_of = {int(f): i for i, f in enumerate(d["frame_idx"])}
    cap = cv2.VideoCapture(a.video)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (IN_W, IN_H))
    idx, n_written, n_missed = -1, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        i = row_of.get(idx)
        if i is None:
            continue
        # vẽ trên đúng ảnh MediaPipe đã thấy (đã làm sáng nếu có)
        img = enhance(fit_frame(frame, fit), meta["config"].get("enhance", "none"))
        phase = next((ascii_phase(p["label"]) for p in phases
                      if p["start_frame"] <= idx < p["end_frame"]), "")
        missed = draw_frame(img, idx, d, i, expected[i], found[i], phase)
        n_missed += missed
        if a.missed_only and not missed:
            continue
        writer.write(img)
        n_written += 1
    cap.release()
    writer.release()
    print(f"\nFrame bị sót tay: {n_missed}/{len(d['frame_idx'])}")
    print(f"Đã ghi {n_written} frame -> {out}")


if __name__ == "__main__":
    main()
