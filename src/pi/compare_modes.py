"""So sánh running_mode IMAGE và VIDEO, riêng cho hand và cho pose, ở 2 mức lấy mẫu frame.

Chạy:
    python -m src.pi.compare_modes --video data/videos/v3_gan_den.mp4

Vì hand và pose là 2 model độc lập (chế độ của pose không ảnh hưởng landmark của hand),
chỉ cần 4 lần trích thay vì 8:
    stride 1: (hand IMAGE + pose VIDEO), (hand VIDEO + pose IMAGE)
    stride 3: như trên
-> đủ 4 chế độ (hand/pose x IMAGE/VIDEO) ở cả stride 1 và stride 3.
Kết quả trích lưu ở runs/compare/<video>/ và được DÙNG LẠI nếu chạy lần sau.

Bảng kết quả (đơn vị px trên ảnh 640x480):
  (a) thấy:  tỉ lệ frame thấy tay (>=1 / đủ 2) trong đoạn chắc chắn có tay
             (đoạn kịch bản "Giữ..." và "Ký..."; video không có kịch bản thì dùng cổ tay pose)
  (b) rung:  khi tay GIỮ YÊN, landmark xê dịch bao nhiêu giữa 2 frame liên tiếp (p50).
             Càng nhỏ càng ổn định. Chỉ tính được nếu kịch bản có đoạn "Giữ...".
  (c) lệch s1-s3: trên các frame chung (0, 3, 6...), landmark khi xử lý mọi frame khác
             landmark khi xử lý 1/3 frame bao nhiêu (p50). IMAGE phải ~0;
             VIDEO > 0 nghĩa là kết quả phụ thuộc tốc độ lấy mẫu.
  (d) ms:    thời gian p50 / p95 (stride 1)
Quy tắc chọn (phiên điều phối): (i) loại chế độ thấy tay kém rõ rệt; (ii) ưu tiên chế độ
ít phụ thuộc stride; (iii) rồi mới chọn chế độ nhanh hơn.
"""
import argparse
import json
import os

import numpy as np

from src.pi.draw import analyse
from src.pi.extract_landmark import IN_H, IN_W, extract_video

SCALE = np.array([IN_W, IN_H], np.float32)
POSE_UPPER = list(range(25))   # pose: điểm 0-24 (thân trên; chân thường ngoài khung)


def load_phases(video):
    path = os.path.splitext(video)[0] + ".json"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("phases", [])


def frame_mask(frame_idx, phases, words):
    m = np.zeros(len(frame_idx), bool)
    for p in phases:
        if any(p["label"].startswith(w) for w in words):
            m |= (frame_idx >= p["start_frame"]) & (frame_idx < p["end_frame"])
    return m


def hand_pts(d):
    """(T, 2 tay, 21, 2) px và mặt nạ có tay (T, 2)."""
    h = d["landmarks"][:, :126].reshape(-1, 2, 21, 3)
    return h[..., :2] * SCALE, np.abs(h).sum((2, 3)) > 0


def pose_pts(d):
    p = d["landmarks"][:, 126:].reshape(-1, 33, 3)[:, POSE_UPPER, :2] * SCALE
    return p[:, None], (np.abs(p).sum((1, 2)) > 0)[:, None]      # thêm trục "1 người"


def jitter(pts, has, sel):
    """Dịch chuyển TB của các điểm giữa 2 frame liên tiếp, chỉ trong vùng sel, p50."""
    vals = []
    for t in range(1, len(pts)):
        if not (sel[t] and sel[t - 1]):
            continue
        for k in range(pts.shape[1]):
            if has[t, k] and has[t - 1, k]:
                vals.append(np.linalg.norm(pts[t, k] - pts[t - 1, k], axis=-1).mean())
    return (round(float(np.median(vals)), 2), len(vals)) if vals else (None, 0)


def stride_diff(d1, d3, get):
    """Lệch landmark giữa bản stride 1 và stride 3 trên frame chung, p50."""
    p1, h1 = get(d1)
    p3, h3 = get(d3)
    row1 = {int(f): i for i, f in enumerate(d1["frame_idx"])}
    vals, agree, n = [], 0, 0
    for j, f in enumerate(d3["frame_idx"]):
        i = row1.get(int(f))
        if i is None:
            continue
        n += 1
        agree += int((h1[i] == h3[j]).all())
        for k in range(p1.shape[1]):
            if h1[i, k] and h3[j, k]:
                vals.append(np.linalg.norm(p1[i, k] - p3[j, k], axis=-1).mean())
    med = round(float(np.median(vals)), 2) if vals else None
    return med, round(agree / n, 2) if n else None


def run_or_load(video, hand_mode, pose_mode, stride, a, out_dir):
    stem = os.path.splitext(os.path.basename(video))[0]
    path = os.path.join(out_dir, f"{stem}__h{hand_mode[0]}_p{pose_mode[0]}_s{stride}_e{a.enhance}.npz")
    if os.path.exists(path):
        print(f"  dùng lại {path}")
        return dict(np.load(path))
    cfg = {
        "hand_mode": hand_mode, "pose_mode": pose_mode, "stride": stride,
        "num_hands": 2, "fit": "letterbox", "enhance": a.enhance,
        "input_size": [IN_W, IN_H], "mirrored": False,
        "hand_model": a.hand_model, "pose_model": a.pose_model,
        "hand_thresholds": {"detection": a.hand_det, "presence": 0.5, "tracking": 0.5},
        "pose_thresholds": {"detection": 0.5, "presence": 0.5, "tracking": 0.5},
    }
    print(f"  trích hand={hand_mode} pose={pose_mode} stride={stride} ...")
    res = extract_video(video, cfg, a.max_frames, log_every=0)
    np.savez_compressed(path, **res)
    return {k: (v if k != "meta" else np.array(v)) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser(description="So sánh IMAGE / VIDEO cho hand và pose")
    ap.add_argument("--video", required=True)
    ap.add_argument("--stride2", type=int, default=3, help="mức lấy mẫu thứ hai (mặc định 3)")
    ap.add_argument("--enhance", default="none", choices=["none", "gamma", "clahe", "gamma_clahe"])
    ap.add_argument("--hand-det", type=float, default=0.5)
    ap.add_argument("--hand-model", default="data/models/hand_landmarker.task")
    ap.add_argument("--pose-model", default="data/models/pose_landmarker_lite.task")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    stem = os.path.splitext(os.path.basename(a.video))[0]
    out_dir = os.path.join("runs/compare", stem)
    os.makedirs(out_dir, exist_ok=True)
    s2 = a.stride2
    print(f"So sánh chế độ trên {a.video} (stride 1 và {s2})")
    R = {}
    for s in (1, s2):
        R[("hI_pV", s)] = run_or_load(a.video, "IMAGE", "VIDEO", s, a, out_dir)
        R[("hV_pI", s)] = run_or_load(a.video, "VIDEO", "IMAGE", s, a, out_dir)
    # Mỗi chế độ lấy từ lần trích tương ứng
    src = {("hand", "IMAGE"): "hI_pV", ("hand", "VIDEO"): "hV_pI",
           ("pose", "IMAGE"): "hV_pI", ("pose", "VIDEO"): "hI_pV"}

    phases = load_phases(a.video)
    rows = []
    for (part, mode), key in src.items():
        d1, d3 = R[(key, 1)], R[(key, s2)]
        f1 = d1["frame_idx"]
        static = frame_mask(f1, phases, ["Giữ"])
        if phases:
            known = frame_mask(f1, phases, ["Giữ", "Ký"])
        else:  # không có kịch bản: dùng cổ tay pose
            known = analyse(d1)[0].any(1)
        get = hand_pts if part == "hand" else pose_pts
        pts, has = get(d1)
        if part == "hand":
            see1 = float(np.mean(has[known].any(1))) if known.any() else None
            see2 = float(np.mean(has[known].all(1))) if known.any() else None
            ms = d1["hand_ms"]
        else:
            see1 = see2 = float(np.mean(has[:, 0]))
            ms = d1["pose_ms"]
        jit, n_jit = jitter(pts, has, static)
        diff, agree = stride_diff(d1, d3, get)
        rows.append({
            "phan": part, "che_do": mode,
            "thay_>=1": round(see1, 2) if see1 is not None else None,
            "thay_2": round(see2, 2) if see2 is not None else None,
            "rung_px_p50": jit, "so_cap_rung": n_jit,
            f"lech_s1_s{s2}_px_p50": diff, f"khop_phat_hien_s1_s{s2}": agree,
            "ms_p50": round(float(np.percentile(ms, 50)), 1),
            "ms_p95": round(float(np.percentile(ms, 95)), 1),
        })

    print(f"\n{'phần':<6}{'chế độ':<8}{'thấy≥1':>8}{'thấy 2':>8}{'rung px':>9}"
          f"{f'lệch s1-s{s2}':>12}{'khớp':>7}{'ms p50':>8}{'ms p95':>8}")
    for r in rows:
        f = lambda v: "-" if v is None else v
        print(f"{r['phan']:<6}{r['che_do']:<8}{f(r['thay_>=1']):>8}{f(r['thay_2']):>8}"
              f"{f(r['rung_px_p50']):>9}{f(r[f'lech_s1_s{s2}_px_p50']):>12}"
              f"{f(r[f'khop_phat_hien_s1_s{s2}']):>7}{r['ms_p50']:>8}{r['ms_p95']:>8}")
    print("(pose: 'thấy' = tỉ lệ frame thấy người; 'khớp' = tỉ lệ frame chung mà s1 và s3 "
          "cùng thấy/cùng không thấy)")
    if not frame_mask(R[("hI_pV", 1)]["frame_idx"], phases, ["Giữ"]).any():
        print("Không có đoạn 'Giữ...' trong kịch bản -> không đo được độ rung.")
    with open(os.path.join(out_dir, f"summary_e{a.enhance}.json"), "w", encoding="utf-8") as fh:
        json.dump({"video": a.video, "stride2": s2, "enhance": a.enhance, "rows": rows},
                  fh, ensure_ascii=False, indent=2)
    print(f"Đã lưu {out_dir}/summary_e{a.enhance}.json")


if __name__ == "__main__":
    main()
