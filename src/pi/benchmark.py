"""Đo thời gian từng khâu của MediaPipe (Hand + Pose) trên Pi.

Ví dụ (chạy trong venv, ở thư mục gốc dự án):
    python -m src.pi.benchmark --source data/videos/test_30s.mp4 --mode IMAGE
    python -m src.pi.benchmark --source data/videos/test_30s.mp4 --mode VIDEO
    python -m src.pi.benchmark --source data/videos/test_30s.mp4 --mode VIDEO --stride 7
    python -m src.pi.benchmark --source cam --frames 300
    python -m src.pi.benchmark --source data/videos/test_30s.mp4 --only hand

Đo gì (mỗi frame, bằng time.perf_counter):
    capture : lấy 1 frame (camera: chờ rpicam-vid; file: giải nén mp4)
    convert : BGR -> RGB + tạo mp.Image
    hand    : HandLandmarker.detect / detect_for_video
    pose    : PoseLandmarker.detect / detect_for_video
    total   : cả vòng lặp
Quy tắc:
    - Khởi động trước (--warmup frame) KHÔNG tính vào thống kê: lần chạy đầu phải nạp model,
      cấp phát bộ nhớ, CPU còn ở xung thấp (governor ondemand).
      Với file video: khởi động bằng frame đầu lặp lại, để cả video vẫn được đo đủ.
    - Báo p50 (thời gian điển hình) và p95 (lúc chậm) thay vì chỉ trung bình.
    - Luồng phụ ghi mỗi giây: nhiệt độ, xung CPU thực, get_throttled, %CPU từng nhân.
--stride k: chỉ xử lý 1 trong k frame của video. Chạy trực tiếp ở ~4 fps thì giữa 2 lần xử lý
    tay đã di chuyển 7-8 frame; stride 7 mô phỏng đúng điều đó cho chế độ VIDEO (bám theo tay).
Kết quả: runs/pi_bench/<tên>/frames.csv, system.csv, summary.json
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import psutil

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from src.pi.camera import VideoFileSource, open_source

STAGES = ["capture", "convert", "hand", "pose", "total"]


# ---------------------------------------------------------------- hệ thống
def run_cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def read_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except OSError:
        return float("nan")


def read_arm_mhz():
    # vcgencmd cho xung THỰC (đã tính hạ xung); sysfs chỉ là xung được yêu cầu
    out = run_cmd(["vcgencmd", "measure_clock", "arm"])  # frequency(48)=1800404864
    if "=" in out:
        return int(out.split("=")[1]) / 1e6
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            return int(f.read()) / 1000.0
    except OSError:
        return float("nan")


def read_throttled():
    out = run_cmd(["vcgencmd", "get_throttled"])  # throttled=0x0
    return int(out.split("=")[1], 16) if "=" in out else -1


THROTTLE_BITS = {
    0: "thiếu điện (hiện tại)", 1: "giới hạn xung (hiện tại)",
    2: "đang hạ xung", 3: "chạm ngưỡng nhiệt mềm (hiện tại)",
    16: "từng thiếu điện", 17: "từng giới hạn xung",
    18: "từng hạ xung", 19: "từng chạm ngưỡng nhiệt mềm",
}


def decode_throttled(v):
    if v < 0:
        return ["không đọc được (không có vcgencmd)"]
    return [name for bit, name in THROTTLE_BITS.items() if v & (1 << bit)] or ["bình thường"]


class SystemMonitor(threading.Thread):
    """Mỗi `interval` giây ghi 1 dòng trạng thái máy."""

    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.rows = []
        self._halt = threading.Event()
        self.proc = psutil.Process()

    def run(self):
        psutil.cpu_percent(percpu=True)   # lần đầu luôn trả 0 -> gọi mồi
        self.proc.cpu_percent()
        t0 = time.perf_counter()
        while not self._halt.wait(self.interval):
            per_core = psutil.cpu_percent(percpu=True)
            thr = read_throttled()
            self.rows.append({
                "t_s": round(time.perf_counter() - t0, 1),
                "temp_c": read_temp_c(),
                "arm_mhz": round(read_arm_mhz(), 0),
                "throttled": hex(thr) if thr >= 0 else "na",
                "proc_cpu_pct": self.proc.cpu_percent(),   # 100 = 1 nhân, 400 = cả 4 nhân
                "proc_threads": self.proc.num_threads(),
                **{f"core{i}_pct": v for i, v in enumerate(per_core)},
            })

    def stop(self):
        self._halt.set()
        self.join()


# ---------------------------------------------------------------- MediaPipe
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_landmarkers(a):
    mode = vision.RunningMode.VIDEO if a.mode == "VIDEO" else vision.RunningMode.IMAGE
    hand = pose = None
    if a.only in ("both", "hand"):
        hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=a.hand_model),
            running_mode=mode, num_hands=a.num_hands,
            min_hand_detection_confidence=a.hand_det, min_hand_presence_confidence=a.hand_pres,
            min_tracking_confidence=a.hand_track))
    if a.only in ("both", "pose"):
        pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=a.pose_model),
            running_mode=mode, num_poses=1,
            min_pose_detection_confidence=a.pose_det, min_pose_presence_confidence=a.pose_pres,
            min_tracking_confidence=a.pose_track, output_segmentation_masks=False))
    return hand, pose


def process(frame, ts_ms, hand, pose, video_mode):
    """Chạy 1 frame, trả về thời gian từng khâu (ms) và kết quả tóm tắt."""
    t0 = time.perf_counter()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    t1 = time.perf_counter()
    n_hands, labels = 0, ""
    if hand is not None:
        r = hand.detect_for_video(img, ts_ms) if video_mode else hand.detect(img)
        n_hands = len(r.hand_landmarks)
        labels = "+".join(h[0].category_name for h in r.handedness)
    t2 = time.perf_counter()
    pose_found = 0
    if pose is not None:
        r = pose.detect_for_video(img, ts_ms) if video_mode else pose.detect(img)
        pose_found = int(len(r.pose_landmarks) > 0)
    t3 = time.perf_counter()
    return ((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3), n_hands, labels, pose_found


# ---------------------------------------------------------------- thống kê
def stats(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    return {"n": int(len(x)), "mean": round(float(x.mean()), 1),
            "p50": round(float(np.percentile(x, 50)), 1),
            "p95": round(float(np.percentile(x, 95)), 1),
            "max": round(float(x.max()), 1)}


def load_phases(source):
    """Đọc file .json đi kèm video (do record.py tạo) nếu có."""
    path = os.path.splitext(source)[0] + ".json"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("phases", [])


def phase_of(frame_idx, phases):
    for p in phases:
        if p["start_frame"] <= frame_idx < p["end_frame"]:
            return p["label"]
    return ""


# ---------------------------------------------------------------- chính
def main():
    ap = argparse.ArgumentParser(description="Đo thời gian MediaPipe trên Pi")
    ap.add_argument("--source", required=True, help="'cam' hoặc đường dẫn video")
    ap.add_argument("--mode", choices=["IMAGE", "VIDEO"], default="IMAGE")
    ap.add_argument("--only", choices=["both", "hand", "pose"], default="both")
    ap.add_argument("--frames", type=int, default=0, help="số frame đo (0 = hết video; cam mặc định 300)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1, help="file video: xử lý 1 trong k frame")
    ap.add_argument("--repeat", type=int, default=1, help="file video: chạy lặp N lần (thử nóng máy)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hand-model", default="data/models/hand_landmarker.task")
    ap.add_argument("--pose-model", default="data/models/pose_landmarker_lite.task")
    ap.add_argument("--num-hands", type=int, default=2)
    ap.add_argument("--hand-det", type=float, default=0.5)
    ap.add_argument("--hand-pres", type=float, default=0.5)
    ap.add_argument("--hand-track", type=float, default=0.5)
    ap.add_argument("--pose-det", type=float, default=0.5)
    ap.add_argument("--pose-pres", type=float, default=0.5)
    ap.add_argument("--pose-track", type=float, default=0.5)
    ap.add_argument("--name", default="", help="tên thư mục kết quả")
    ap.add_argument("--out-dir", default="runs/pi_bench")
    a = ap.parse_args()

    is_cam = a.source == "cam"
    if is_cam and a.frames == 0:
        a.frames = 300
    stem = "cam" if is_cam else os.path.splitext(os.path.basename(a.source))[0]
    name = a.name or f"{datetime.now():%m%d_%H%M%S}_{a.mode}_{a.only}_{stem}_s{a.stride}"
    out_dir = os.path.join(a.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    video_mode = a.mode == "VIDEO"
    phases = [] if is_cam else load_phases(a.source)

    print(f"Nạp model ({a.mode}, {a.only})...")
    hand, pose = build_landmarkers(a)
    src = open_source(a.source, a.width, a.height)
    src_fps = src.fps or 30
    mon = SystemMonitor()
    mon.start()

    # --- Khởi động trước
    ts = 0  # timestamp (ms) cho chế độ VIDEO, phải luôn tăng
    if is_cam:
        for _ in range(a.warmup):
            ok, frame = src.read()
            ts = int(time.monotonic() * 1000)
            process(frame, ts, hand, pose, video_mode)
    else:
        ok, first = src.read()
        for _ in range(a.warmup):
            ts += 33
            process(first, ts, hand, pose, video_mode)
        src.close()
    print(f"Đã khởi động {a.warmup} frame. Bắt đầu đo...")

    # --- Đo
    rows = []
    wall0 = time.perf_counter()
    t_mon0 = len(mon.rows)
    try:
        for loop in range(a.repeat if not is_cam else 1):
            if not is_cam:
                src = VideoFileSource(a.source)
            idx = -1
            while True:
                if a.frames and len(rows) >= a.frames:
                    break
                if not is_cam:
                    # bỏ qua (stride-1) frame: chỉ grab, không giải nén, không tính giờ
                    for _ in range(a.stride - 1):
                        if not src.cap.grab():
                            break
                        idx += 1
                t0 = time.perf_counter()
                ok, frame = src.read()
                if not ok:
                    break
                idx += 1
                t_cap = (time.perf_counter() - t0) * 1e3
                if is_cam:
                    ts = int(time.monotonic() * 1000)
                else:
                    ts += int(round(1000 * a.stride / src_fps))  # thời gian theo video
                (t_conv, t_hand, t_pose), n_hands, labels, pose_found = \
                    process(frame, ts, hand, pose, video_mode)
                t_total = (time.perf_counter() - t0) * 1e3
                rows.append({"i": len(rows), "loop": loop, "frame": idx,
                             "phase": phase_of(idx, phases),
                             "capture_ms": round(t_cap, 2), "convert_ms": round(t_conv, 2),
                             "hand_ms": round(t_hand, 2), "pose_ms": round(t_pose, 2),
                             "total_ms": round(t_total, 2),
                             "n_hands": n_hands, "handedness": labels, "pose_found": pose_found})
                if len(rows) % 50 == 0:
                    print(f"  {len(rows)} frame, total gần nhất {t_total:.0f} ms, "
                          f"{read_temp_c():.1f} °C")
            if not is_cam:
                src.close()
            if a.frames and len(rows) >= a.frames:
                break
    except KeyboardInterrupt:
        print("Dừng sớm, vẫn lưu kết quả.")
    wall = time.perf_counter() - wall0
    if is_cam:
        src.close()
    mon.stop()
    sysrows = mon.rows[t_mon0:] or mon.rows

    # --- Lưu CSV
    if rows:
        with open(os.path.join(out_dir, "frames.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if sysrows:
        with open(os.path.join(out_dir, "system.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sysrows[0].keys()))
            w.writeheader()
            w.writerows(sysrows)

    # --- Thống kê
    col = lambda k, sel=rows: [r[k] for r in sel]
    st = {s: stats(col(f"{s}_ms")) for s in STAGES}
    with_hand = [r for r in rows if r["n_hands"] > 0]
    no_hand = [r for r in rows if r["n_hands"] == 0]
    by_phase = {}
    for p in phases:
        sel = [r for r in rows if r["phase"] == p["label"] and p["start_frame"] <= r["frame"] < p["end_frame"]]
        if sel:
            key = f'{p["label"]} [{p["start_frame"]}-{p["end_frame"]})'
            by_phase[key] = {"hand": stats(col("hand_ms", sel)), "pose": stats(col("pose_ms", sel)),
                             "ty_le_thay_tay": round(sum(r["n_hands"] > 0 for r in sel) / len(sel), 2)}
    thr_all = 0
    for r in sysrows:
        if r["throttled"] != "na":
            thr_all |= int(r["throttled"], 16)
    cores = [k for k in (sysrows[0].keys() if sysrows else []) if k.startswith("core")]
    sysinfo = {
        "temp_c": stats([r["temp_c"] for r in sysrows]),
        "arm_mhz": stats([r["arm_mhz"] for r in sysrows]),
        "proc_cpu_pct": stats([r["proc_cpu_pct"] for r in sysrows]),
        "proc_threads_max": max((r["proc_threads"] for r in sysrows), default=0),
        "core_pct_mean": {c: round(float(np.mean([r[c] for r in sysrows])), 1) for c in cores},
        "throttled_or": hex(thr_all), "throttled_meaning": decode_throttled(thr_all),
    }
    p50 = st["total"]["p50"] if st["total"] else float("nan")
    summary = {
        "name": name, "time": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "mediapipe": mp.__version__, "python": platform.python_version(),
            "opencv": cv2.__version__, "numpy": np.__version__,
            "machine": platform.machine(), "platform": platform.platform(),
            "source": a.source, "input_size": [src.width, src.height],
            "running_mode": a.mode, "only": a.only, "num_hands": a.num_hands,
            "hand_model": a.hand_model if hand else None,
            "hand_model_sha256": sha256(a.hand_model) if hand else None,
            "pose_model": a.pose_model if pose else None,
            "pose_model_sha256": sha256(a.pose_model) if pose else None,
            "hand_thresholds": {"detection": a.hand_det, "presence": a.hand_pres, "tracking": a.hand_track},
            "pose_thresholds": {"detection": a.pose_det, "presence": a.pose_pres, "tracking": a.pose_track},
            "warmup": a.warmup, "stride": a.stride, "repeat": a.repeat, "mirrored": False,
        },
        "frames_measured": len(rows), "wall_s": round(wall, 1),
        "fps_from_p50": round(1000 / p50, 2) if p50 == p50 else None,
        "fps_real": round(len(rows) / wall, 2) if wall > 0 else None,
        "stages_ms": st,
        "hand_ms_khi_co_tay": stats(col("hand_ms", with_hand)),
        "hand_ms_khi_khong_tay": stats(col("hand_ms", no_hand)),
        "ty_le_frame_thay_tay": round(len(with_hand) / len(rows), 2) if rows else None,
        "ty_le_frame_thay_pose": round(sum(col("pose_found")) / len(rows), 2) if rows else None,
        "by_phase": by_phase, "system": sysinfo,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # --- In bảng
    print("\n" + "=" * 62)
    print(f"{name}")
    print(f"mediapipe {mp.__version__} | Python {platform.python_version()} | {a.mode} | "
          f"num_hands={a.num_hands} | stride={a.stride} | {len(rows)} frame")
    print(f"{'khâu':<9}{'mean':>8}{'p50':>8}{'p95':>8}{'max':>8}   (ms)")
    for s in STAGES:
        if st[s]:
            print(f"{s:<9}{st[s]['mean']:>8}{st[s]['p50']:>8}{st[s]['p95']:>8}{st[s]['max']:>8}")
    print(f"fps (1000/p50 total) = {summary['fps_from_p50']} | fps thực = {summary['fps_real']}")
    for k, lab in (("hand_ms_khi_co_tay", "hand khi CÓ tay"), ("hand_ms_khi_khong_tay", "hand khi KHÔNG tay")):
        if summary[k]:
            print(f"{lab:<20} p50 {summary[k]['p50']} ms, p95 {summary[k]['p95']} ms (n={summary[k]['n']})")
    print(f"tỉ lệ frame thấy tay {summary['ty_le_frame_thay_tay']}, thấy pose {summary['ty_le_frame_thay_pose']}")
    for k, v in by_phase.items():
        h = v["hand"]
        print(f"  đoạn {k}: hand p50 {h['p50'] if h else '-'} ms, thấy tay {v['ty_le_thay_tay']}")
    si = sysinfo
    if si["temp_c"]:
        arm_min = min(r["arm_mhz"] for r in sysrows)
        print(f"nhiệt độ p50 {si['temp_c']['p50']} °C (max {si['temp_c']['max']}) | "
              f"xung ARM min {arm_min:.0f} / p50 {si['arm_mhz']['p50']:.0f} MHz")
        print(f"%CPU tiến trình p50 {si['proc_cpu_pct']['p50']} (100 = 1 nhân) | "
              f"số luồng {si['proc_threads_max']} | từng nhân {list(si['core_pct_mean'].values())}")
    print(f"get_throttled (gộp) {si['throttled_or']}: {', '.join(si['throttled_meaning'])}")
    print(f"Đã lưu: {out_dir}/ (frames.csv, system.csv, summary.json)")


if __name__ == "__main__":
    main()
