"""Trích landmark MediaPipe (Hand + Pose) từ 1 video -> file .npz.

Đây là bộ trích DÙNG CHUNG cho Pi và Kaggle: cả hai chạy đúng file này để landmark giống nhau.

Chạy (dùng cấu hình đã chốt, KHUYÊN DÙNG):
    python -m src.pi.extract_landmark --video a.mp4 --config configs/extractor_config.json
Dùng trong code (vd notebook Kaggle của nhánh M):
    from src.pi.extract_landmark import load_config, extract_video
    cfg = load_config("configs/extractor_config.json", model_dir="/kaggle/working/models")
    res = extract_video("video.mp4", cfg)      # dict mảng numpy, landmarks (T, 225)
Chạy thử nghiệm (tự chọn tham số, không dùng file cấu hình):
    python -m src.pi.extract_landmark --video data/videos/test_30s.mp4
    python -m src.pi.extract_landmark --video a.mp4 --hand-mode VIDEO --pose-mode VIDEO --stride 3

Các bước cho mỗi frame được xử lý:
    1. Đọc frame (bỏ qua stride-1 frame giữa hai lần xử lý)
    2. Đưa về 640x480, GIỮ tỉ lệ khung hình (--fit letterbox: thêm viền đen; crop: cắt giữa)
    2b. (tuỳ chọn --enhance) làm sáng / tăng tương phản ảnh tối
    3. BGR -> RGB, KHÔNG lật gương
    4. Hand và Pose (mỗi cái có running_mode riêng: IMAGE hoặc VIDEO)
    5. Xếp vào vector 225 số: tay trái 21x3 | tay phải 21x3 | pose 33x3, không thấy thì 0

Nội dung file .npz:
    landmarks    (T, 225) float32 : dữ liệu chính cho nhánh M
    frame_idx    (T,)     int32   : số thứ tự frame gốc của mỗi dòng
    n_hands      (T,)     int8    : số tay MediaPipe thấy
    hand_score   (T, 2)   float32 : độ tin cậy nhãn tay [trái, phải], 0 nếu không thấy
    same_label   (T,)     bool    : 2 tay bị gán cùng nhãn Left/Right
    hand_px      (T, 2)   float32 : cỡ bàn tay (pixel, cạnh dài khung bao 21 điểm) [trái, phải]
    pose_vis     (T, 33)  float32 : độ chắc chắn từng điểm pose
    pose_hand_px (T, 2)   float32 : cỡ bàn tay ước lượng từ pose (cổ tay -> ngón trỏ), có cả khi sót tay
    hand_ms, pose_ms (T,) float32 : thời gian xử lý
    enh_ms       (T,)     float32 : thời gian làm sáng ảnh (0 nếu không dùng)
    brightness   (T,)     float32 : độ sáng TB ảnh TRƯỚC khi làm sáng (0-255)
    meta         str (JSON)       : cấu hình, phiên bản, sha256 model, thông tin video
"""
import argparse
import hashlib
import json
import os
import platform
import time

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

IN_W, IN_H = 640, 480
N_HAND, N_POSE = 21, 33
DIM = 2 * N_HAND * 3 + N_POSE * 3  # 225
# Chỉ số pose: cổ tay và ngón trỏ, theo nhãn của pose (left_*, right_*)
POSE_WRIST = (15, 16)
POSE_INDEX = (19, 20)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_frame(frame, mode="letterbox", w=IN_W, h=IN_H):
    """Đưa frame về đúng w x h mà KHÔNG kéo giãn hình."""
    fh, fw = frame.shape[:2]
    if (fw, fh) == (w, h):
        return frame
    if mode == "letterbox":                      # thu nhỏ vừa khung, thêm viền đen
        s = min(w / fw, h / fh)
        nw, nh = round(fw * s), round(fh * s)
        small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        out = np.zeros((h, w, 3), np.uint8)
        x0, y0 = (w - nw) // 2, (h - nh) // 2
        out[y0:y0 + nh, x0:x0 + nw] = small
        return out
    if mode == "crop":                           # phóng cho đầy khung, cắt phần thừa ở giữa
        s = max(w / fw, h / fh)
        nw, nh = round(fw * s), round(fh * s)
        big = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        x0, y0 = (nw - w) // 2, (nh - h) // 2
        return big[y0:y0 + h, x0:x0 + w]
    raise ValueError(f"fit không hợp lệ: {mode}")


GAMMA_TARGET = 110.0   # độ sáng TB mong muốn sau khi làm sáng
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance(frame, mode="none"):
    """Làm sáng ảnh tối trước khi đưa vào MediaPipe.

    gamma : gamma TỰ ĐỘNG theo độ sáng TB của frame, chỉ làm SÁNG (ảnh đã đủ sáng thì
            gần như không đổi -> video VSL400 quay studio ít bị ảnh hưởng).
            out = 255 * (in/255)^g, chọn g sao cho độ sáng TB ~ GAMMA_TARGET, g trong [0.4, 1].
    clahe : cân bằng tương phản theo từng ô 8x8 trên kênh độ sáng L (không đổi màu).
    gamma_clahe : gamma rồi CLAHE.
    """
    if mode == "none":
        return frame
    out = frame
    if mode in ("gamma", "gamma_clahe"):
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        px = gray[gray > 2]                      # bỏ viền đen letterbox khi tính độ sáng
        m = max(float(px.mean()) if px.size else 0.0, 1.0)
        g = 1.0 if m >= GAMMA_TARGET else np.log(GAMMA_TARGET / 255.0) / np.log(m / 255.0)
        g = float(np.clip(g, 0.4, 1.0))
        if g < 0.999:
            lut = (255.0 * (np.arange(256) / 255.0) ** g).clip(0, 255).astype(np.uint8)
            out = cv2.LUT(out, lut)
    if mode in ("clahe", "gamma_clahe"):
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out


class Extractor:
    """Giữ 2 model và xử lý từng frame. Dùng lại được ở chạy trực tiếp trên Pi."""

    def __init__(self, cfg):
        self.cfg = cfg
        hm = vision.RunningMode[cfg["hand_mode"]]
        pm = vision.RunningMode[cfg["pose_mode"]]
        th, tp = cfg["hand_thresholds"], cfg["pose_thresholds"]
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=cfg["hand_model"]),
            running_mode=hm, num_hands=cfg["num_hands"],
            min_hand_detection_confidence=th["detection"],
            min_hand_presence_confidence=th["presence"],
            min_tracking_confidence=th["tracking"]))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=cfg["pose_model"]),
            running_mode=pm, num_poses=1,
            min_pose_detection_confidence=tp["detection"],
            min_pose_presence_confidence=tp["presence"],
            min_tracking_confidence=tp["tracking"],
            output_segmentation_masks=False))
        self.hand_video = cfg["hand_mode"] == "VIDEO"
        self.pose_video = cfg["pose_mode"] == "VIDEO"

    def process(self, frame_bgr, ts_ms):
        """frame_bgr đã là 640x480. Trả về dict kết quả của 1 frame."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t0 = time.perf_counter()
        hr = self.hand.detect_for_video(img, ts_ms) if self.hand_video else self.hand.detect(img)
        t1 = time.perf_counter()
        pr = self.pose.detect_for_video(img, ts_ms) if self.pose_video else self.pose.detect(img)
        t2 = time.perf_counter()

        vec = np.zeros(DIM, np.float32)
        score = np.zeros(2, np.float32)
        hand_px = np.zeros(2, np.float32)
        # Xếp tay theo nhãn MediaPipe: 'Left' -> ô 0, 'Right' -> ô 1 (giữ nguyên nhãn, không đảo)
        hands = []
        for lms, cats in zip(hr.hand_landmarks, hr.handedness):
            slot = 0 if cats[0].category_name == "Left" else 1
            hands.append((cats[0].score, slot, lms))
        same_label = len(hands) == 2 and hands[0][1] == hands[1][1]
        if same_label:
            # TẠM THỜI (chưa chốt): tay tin cậy cao hơn giữ nhãn, tay còn lại sang ô kia
            hands.sort(key=lambda x: -x[0])
            hands[1] = (hands[1][0], 1 - hands[0][1], hands[1][2])
        for sc, slot, lms in hands:
            pts = np.array([[p.x, p.y, p.z] for p in lms], np.float32)
            vec[slot * 63:(slot + 1) * 63] = pts.reshape(-1)
            score[slot] = sc
            wh = (pts[:, :2].max(0) - pts[:, :2].min(0)) * (IN_W, IN_H)
            hand_px[slot] = wh.max()

        pose_vis = np.zeros(N_POSE, np.float32)
        pose_hand_px = np.zeros(2, np.float32)
        if pr.pose_landmarks:
            lms = pr.pose_landmarks[0]
            pts = np.array([[p.x, p.y, p.z] for p in lms], np.float32)
            vec[126:] = pts.reshape(-1)
            pose_vis = np.array([p.visibility for p in lms], np.float32)
            for k in range(2):
                d = (pts[POSE_WRIST[k], :2] - pts[POSE_INDEX[k], :2]) * (IN_W, IN_H)
                pose_hand_px[k] = float(np.hypot(*d))

        return {"vec": vec, "n_hands": len(hands), "score": score, "same_label": same_label,
                "hand_px": hand_px, "pose_vis": pose_vis, "pose_hand_px": pose_hand_px,
                "hand_ms": (t1 - t0) * 1e3, "pose_ms": (t2 - t1) * 1e3}

    def close(self):
        self.hand.close()
        self.pose.close()


def extract_video(video, cfg, max_frames=0, log_every=100):
    """Trích cả video. Trả về dict mảng numpy (chưa lưu)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ex = Extractor(cfg)
    stride = cfg["stride"]
    keys = ["vec", "n_hands", "score", "same_label", "hand_px", "pose_vis",
            "pose_hand_px", "hand_ms", "pose_ms", "enh_ms", "brightness"]
    out = {k: [] for k in keys}
    frame_idx = []
    idx = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride:                      # chỉ xử lý frame 0, k, 2k, ...
                continue
            # timestamp theo thời gian trong video -> giống nhau ở Pi và Kaggle
            ts_ms = int(round(idx * 1000.0 / fps))
            img = fit_frame(frame, cfg["fit"])
            bright = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
            te = time.perf_counter()
            img = enhance(img, cfg.get("enhance", "none"))
            enh_ms = (time.perf_counter() - te) * 1e3
            r = ex.process(img, ts_ms)
            r["enh_ms"] = enh_ms
            r["brightness"] = bright
            for k in keys:
                out[k].append(r[k])
            frame_idx.append(idx)
            if log_every and len(frame_idx) % log_every == 0:
                print(f"  {len(frame_idx)} frame...")
            if max_frames and len(frame_idx) >= max_frames:
                break
    finally:
        cap.release()
        ex.close()

    meta = {
        "video": os.path.basename(video), "src_size": [src_w, src_h], "src_fps": fps,
        "src_frames": n_src, "config": cfg,
        "versions": {"mediapipe": mp.__version__, "python": platform.python_version(),
                     "opencv": cv2.__version__, "numpy": np.__version__,
                     "machine": platform.machine()},
        "model_sha256": {"hand": sha256(cfg["hand_model"]), "pose": sha256(cfg["pose_model"])},
    }
    return {
        "landmarks": np.array(out["vec"], np.float32).reshape(-1, DIM),
        "frame_idx": np.array(frame_idx, np.int32),
        "n_hands": np.array(out["n_hands"], np.int8),
        "hand_score": np.array(out["score"], np.float32).reshape(-1, 2),
        "same_label": np.array(out["same_label"], bool),
        "hand_px": np.array(out["hand_px"], np.float32).reshape(-1, 2),
        "pose_vis": np.array(out["pose_vis"], np.float32).reshape(-1, N_POSE),
        "pose_hand_px": np.array(out["pose_hand_px"], np.float32).reshape(-1, 2),
        "hand_ms": np.array(out["hand_ms"], np.float32),
        "pose_ms": np.array(out["pose_ms"], np.float32),
        "enh_ms": np.array(out["enh_ms"], np.float32),
        "brightness": np.array(out["brightness"], np.float32),
        "meta": json.dumps(meta, ensure_ascii=False),
    }


def load_config(path="configs/extractor_config.json", model_dir="data/models", check_sha=True):
    """Đọc configs/extractor_config.json -> cfg cho Extractor / extract_video."""
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    hand_model = os.path.join(model_dir, c["hand_model"]["file"])
    pose_model = os.path.join(model_dir, c["pose_model"]["file"])
    if check_sha:  # bảo đảm đúng file model như Pi
        for p, want in ((hand_model, c["hand_model"]["sha256"]), (pose_model, c["pose_model"]["sha256"])):
            got = sha256(p)
            if got != want:
                raise RuntimeError(f"sha256 của {p} không khớp cấu hình: {got} != {want}")
    if mp.__version__ != c["moi_truong"]["mediapipe"]:
        print(f"CẢNH BÁO: mediapipe {mp.__version__} khác cấu hình {c['moi_truong']['mediapipe']}")
    th, tp = c["nguong"]["hand"], c["nguong"]["pose"]
    return {
        "hand_mode": c["hand_running_mode"], "pose_mode": c["pose_running_mode"],
        "stride": c["lay_mau_frame"]["kaggle_stride"], "num_hands": c["num_hands"],
        "fit": c["dau_vao"]["fit"], "enhance": c["dau_vao"]["enhance"],
        "input_size": c["dau_vao"]["do_phan_giai"], "mirrored": c["dau_vao"]["lat_guong"],
        "hand_model": hand_model, "pose_model": pose_model,
        "hand_thresholds": {"detection": th["min_hand_detection_confidence"],
                            "presence": th["min_hand_presence_confidence"],
                            "tracking": th["min_tracking_confidence"]},
        "pose_thresholds": {"detection": tp["min_pose_detection_confidence"],
                            "presence": tp["min_pose_presence_confidence"],
                            "tracking": tp["min_tracking_confidence"]},
        "config_file": os.path.basename(path), "config_version": c.get("phien_ban"),
    }


def make_cfg(a):
    return {
        "hand_mode": a.hand_mode, "pose_mode": a.pose_mode, "stride": a.stride,
        "num_hands": a.num_hands, "fit": a.fit, "enhance": a.enhance, "input_size": [IN_W, IN_H], "mirrored": False,
        "hand_model": a.hand_model, "pose_model": a.pose_model,
        "hand_thresholds": {"detection": a.hand_det, "presence": a.hand_pres, "tracking": a.hand_track},
        "pose_thresholds": {"detection": a.pose_det, "presence": a.pose_pres, "tracking": a.pose_track},
    }


def main():
    ap = argparse.ArgumentParser(description="Trích landmark MediaPipe từ video -> .npz")
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="", help="file cấu hình (bỏ qua các tham số chế độ/ngưỡng bên dưới)")
    ap.add_argument("--model-dir", default="data/models")
    ap.add_argument("--out", default="", help="mặc định runs/landmarks/<video>__h.._p.._s...npz")
    ap.add_argument("--hand-mode", choices=["IMAGE", "VIDEO"], default="IMAGE")
    ap.add_argument("--pose-mode", choices=["IMAGE", "VIDEO"], default="IMAGE")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fit", choices=["letterbox", "crop"], default="letterbox")
    ap.add_argument("--enhance", choices=["none", "gamma", "clahe", "gamma_clahe"], default="none")
    ap.add_argument("--num-hands", type=int, default=2)
    ap.add_argument("--hand-model", default="data/models/hand_landmarker.task")
    ap.add_argument("--pose-model", default="data/models/pose_landmarker_lite.task")
    ap.add_argument("--hand-det", type=float, default=0.5)
    ap.add_argument("--hand-pres", type=float, default=0.5)
    ap.add_argument("--hand-track", type=float, default=0.5)
    ap.add_argument("--pose-det", type=float, default=0.5)
    ap.add_argument("--pose-pres", type=float, default=0.5)
    ap.add_argument("--pose-track", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config(a.config, a.model_dir) if a.config else make_cfg(a)
    if a.config:  # tên file / thông báo lấy theo cấu hình
        a.hand_mode, a.pose_mode, a.stride = cfg["hand_mode"], cfg["pose_mode"], cfg["stride"]
        a.fit, a.enhance = cfg["fit"], cfg["enhance"]
        a.hand_det = cfg["hand_thresholds"]["detection"]
    stem = os.path.splitext(os.path.basename(a.video))[0]
    out = a.out or os.path.join(
        "runs/landmarks", f"{stem}__h{a.hand_mode[0]}_p{a.pose_mode[0]}_s{a.stride}"
                          f"_d{a.hand_det:g}" + ("" if a.enhance == "none" else f"_e{a.enhance}") + ".npz")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"Trích {a.video}: hand={a.hand_mode} pose={a.pose_mode} stride={a.stride} fit={a.fit} enhance={a.enhance} hand_det={a.hand_det}")
    t0 = time.perf_counter()
    res = extract_video(a.video, cfg, a.max_frames)
    np.savez_compressed(out, **res)
    T = len(res["frame_idx"])
    if T == 0:
        print("Không có frame nào.")
        return
    nh = res["n_hands"]
    print(f"Xong {T} frame trong {time.perf_counter() - t0:.1f} s -> {out}")
    print(f"  thấy >=1 tay: {np.mean(nh >= 1):.2f} | thấy 2 tay: {np.mean(nh == 2):.2f} | "
          f"thấy pose: {np.mean(res['pose_vis'].max(1) > 0):.2f} | 2 tay cùng nhãn: {res['same_label'].sum()} frame")
    print(f"  độ sáng TB trước làm sáng {res['brightness'].mean():.0f} | làm sáng p50 "
          f"{np.percentile(res['enh_ms'], 50):.1f} ms")
    print(f"  hand p50 {np.percentile(res['hand_ms'], 50):.0f} ms | "
          f"pose p50 {np.percentile(res['pose_ms'], 50):.0f} ms")
    hp = res["hand_px"][res["hand_px"] > 0]
    if hp.size:
        print(f"  cỡ bàn tay (khi thấy) p50 {np.median(hp):.0f} px")


if __name__ == "__main__":
    main()
