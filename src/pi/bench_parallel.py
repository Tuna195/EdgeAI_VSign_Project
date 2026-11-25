"""Đo tốc độ khi chạy hand và pose SONG SONG trên nhiều nhân (nhiều tiến trình).

Chạy:
    python -m src.pi.bench_parallel --video data/videos/v3_gan_den.mp4
    python -m src.pi.bench_parallel --video ... --hand-workers 1     # chỉ tách hand / pose
    python -m src.pi.bench_parallel --video ... --pose-every 2       # pose cách 1 frame chạy 1 lần

Cách chia việc:
    tiến trình chính : đọc frame, chia việc, gom kết quả
    N tiến trình hand (IMAGE): frame nào tới, tiến trình nào rảnh thì nhận
                       (IMAGE xử lý mỗi frame độc lập nên chia được)
    1 tiến trình pose (VIDEO): nhận frame theo đúng thứ tự (VIDEO cần frame liên tiếp);
                       --pose-every k: chỉ chạy pose ở frame 0, k, 2k... (frame giữa dùng lại/nội suy)
    Tối đa --max-inflight frame đang xử lý cùng lúc (giống camera: không dồn frame cũ).
Đo:
    fps thực    = số frame xong (có đủ hand + pose) / thời gian
    độ trễ      = từ lúc gửi frame đi tới lúc có đủ hand + pose
    hand/pose ms= thời gian chạy model trong từng tiến trình (song song có thể chậm hơn chạy riêng)
    + nhiệt độ, xung CPU, get_throttled, %CPU từng nhân (mỗi giây)
"""
import argparse
import json
import multiprocessing as mproc
import os
import queue
import time
from datetime import datetime

import cv2
import numpy as np

from src.pi.benchmark import SystemMonitor, decode_throttled, stats


def hand_worker(task_q, res_q, model, num_hands):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.IMAGE, num_hands=num_hands))
    res_q.put(("ready", os.getpid()))
    while True:
        item = task_q.get()
        if item is None:
            break
        i, frame = item
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        res_q.put(("hand", i, (time.perf_counter() - t0) * 1e3, len(r.hand_landmarks), os.getpid()))
    lm.close()


def pose_worker(task_q, res_q, model):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO, num_poses=1, output_segmentation_masks=False))
    res_q.put(("ready", os.getpid()))
    while True:
        item = task_q.get()
        if item is None:
            break
        i, frame, ts = item
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        res_q.put(("pose", i, (time.perf_counter() - t0) * 1e3, len(r.pose_landmarks), os.getpid()))
    lm.close()


def main():
    ap = argparse.ArgumentParser(description="Đo hand + pose chạy song song nhiều nhân")
    ap.add_argument("--video", required=True)
    ap.add_argument("--hand-workers", type=int, default=3)
    ap.add_argument("--max-inflight", type=int, default=0, help="mặc định = số tiến trình hand + 1")
    ap.add_argument("--frames", type=int, default=0, help="0 = hết video")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--pose-every", type=int, default=1, help="chạy pose mỗi k frame")
    ap.add_argument("--hand-model", default="data/models/hand_landmarker.task")
    ap.add_argument("--pose-model", default="data/models/pose_landmarker_lite.task")
    ap.add_argument("--note", default="truoc khi lap quat", help="ghi chú (vd: truoc/sau khi lap quat)")
    a = ap.parse_args()
    max_inflight = a.max_inflight or a.hand_workers + 1

    ctx = mproc.get_context("spawn")
    res_q = ctx.Queue()
    hand_q = ctx.Queue()
    pose_q = ctx.Queue()
    procs = [ctx.Process(target=hand_worker, args=(hand_q, res_q, a.hand_model, 2), daemon=True)
             for _ in range(a.hand_workers)]
    procs.append(ctx.Process(target=pose_worker, args=(pose_q, res_q, a.pose_model), daemon=True))
    for p in procs:
        p.start()
    print(f"Khởi động {a.hand_workers} tiến trình hand + 1 pose...")
    for _ in procs:
        res_q.get()  # chờ từng tiến trình nạp xong model

    cap = cv2.VideoCapture(a.video)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mon = SystemMonitor()
    mon.start()

    sent_t, done_parts, results, need = {}, {}, [], {}
    hand_ms, pose_ms, hand_pid = [], [], {}
    i_next, n_done, eof = 0, 0, False
    t_start = None
    try:
        while True:
            # Gửi frame mới khi còn chỗ
            while not eof and len(sent_t) < max_inflight:
                ok, frame = cap.read()
                if not ok or (a.frames and i_next >= a.frames + a.warmup):
                    eof = True
                    break
                if i_next == a.warmup:
                    t_start = time.perf_counter()
                sent_t[i_next] = time.perf_counter()
                hand_q.put((i_next, frame))
                need[i_next] = 1
                if i_next % a.pose_every == 0:
                    pose_q.put((i_next, frame, int(round(i_next * 1000 / fps_src))))
                    need[i_next] = 2
                i_next += 1
            if eof and not sent_t:
                break
            # Nhận kết quả
            try:
                kind, i, ms, n, pid = res_q.get(timeout=10)
            except queue.Empty:
                print("Quá 10 s không có kết quả, dừng.")
                break
            done_parts.setdefault(i, {})[kind] = n
            if i >= a.warmup:
                (hand_ms if kind == "hand" else pose_ms).append(ms)
                if kind == "hand":
                    hand_pid[pid] = hand_pid.get(pid, 0) + 1
            if len(done_parts[i]) == need[i]:    # đủ hand (+ pose nếu frame này có chạy pose)
                lat = (time.perf_counter() - sent_t.pop(i)) * 1e3
                parts = done_parts.pop(i)
                need.pop(i)
                if i >= a.warmup:
                    results.append({"frame": i, "latency_ms": lat, "n_hands": parts["hand"]})
                n_done += 1
                if n_done % 100 == 0:
                    rows = mon.rows
                    t = rows[-1]["temp_c"] if rows else float("nan")
                    mhz = rows[-1]["arm_mhz"] if rows else float("nan")
                    print(f"  {n_done} frame | {t:.1f} °C | {mhz:.0f} MHz")
    except KeyboardInterrupt:
        print("Dừng sớm.")
    wall = time.perf_counter() - (t_start or time.perf_counter())
    mon.stop()
    for _ in range(a.hand_workers):
        hand_q.put(None)
    pose_q.put(None)
    for p in procs:
        p.join(timeout=5)
    cap.release()

    sysrows = mon.rows
    thr = 0
    for r in sysrows:
        if r["throttled"] != "na":
            thr |= int(r["throttled"], 16)
    lat = [r["latency_ms"] for r in results]
    fps_real = len(results) / wall if wall > 0 else 0
    cores = [k for k in (sysrows[0] if sysrows else {}) if k.startswith("core")]
    summary = {
        "time": datetime.now().isoformat(timespec="seconds"), "note": a.note,
        "video": a.video, "hand_workers": a.hand_workers, "max_inflight": max_inflight,
        "hand_mode": "IMAGE", "pose_mode": "VIDEO", "pose_every": a.pose_every,
        "frames": len(results), "wall_s": round(wall, 1), "fps_real": round(fps_real, 2),
        "latency_ms": stats(lat), "hand_ms": stats(hand_ms), "pose_ms": stats(pose_ms),
        "hand_frames_per_worker": list(hand_pid.values()),
        "temp_c": stats([r["temp_c"] for r in sysrows]),
        "arm_mhz": stats([r["arm_mhz"] for r in sysrows]),
        "arm_mhz_min": min((r["arm_mhz"] for r in sysrows), default=None),
        "core_pct_mean": {c: round(float(np.mean([r[c] for r in sysrows])), 1) for c in cores},
        "throttled_or": hex(thr), "throttled_meaning": decode_throttled(thr),
    }
    out_dir = "runs/pi_parallel"
    os.makedirs(out_dir, exist_ok=True)
    name = f"{datetime.now():%m%d_%H%M%S}_h{a.hand_workers}_pe{a.pose_every}"
    with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Song song: {a.hand_workers} tiến trình hand (IMAGE) + 1 pose (VIDEO, mỗi "
          f"{a.pose_every} frame) | {a.note}")
    print(f"{len(results)} frame trong {wall:.1f} s -> fps thực = {fps_real:.2f}")
    s = summary
    for k, lab in (("latency_ms", "độ trễ"), ("hand_ms", "hand"), ("pose_ms", "pose")):
        if s[k]:
            print(f"{lab:<8} TB {s[k]['mean']:>6} | p50 {s[k]['p50']:>6} | p95 {s[k]['p95']:>6} ms")
    print(f"số frame mỗi tiến trình hand xử lý: {s['hand_frames_per_worker']}")
    if s["temp_c"]:
        print(f"nhiệt độ p50 {s['temp_c']['p50']} °C, max {s['temp_c']['max']} °C | "
              f"xung p50 {s['arm_mhz']['p50']:.0f} MHz, min {s['arm_mhz_min']:.0f} MHz")
    print(f"%CPU từng nhân: {list(s['core_pct_mean'].values())}")
    print(f"get_throttled (gộp) {thr:#x}: {', '.join(s['throttled_meaning'])}")
    print(f"Đã lưu {out_dir}/{name}.json")


if __name__ == "__main__":
    main()
