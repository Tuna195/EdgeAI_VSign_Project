"""Trích landmark BÀN TAY (hand IMAGE) cho toàn bộ VSL400 front_view, chạy song song trên Kaggle.

Dùng NGUYÊN bộ trích của nhánh P (src/pi/extract_landmark.py + configs/extractor_config.json):
    load_config, fit_frame, enhance, Extractor.process (xếp tay Left/Right, luật 2 tay cùng nhãn).
Chỉ khác: KHÔNG chạy pose (thay model pose bằng _NoPose trả kết quả rỗng) -> phần tay giống hệt
extract_video. Pose (VIDEO) sẽ trích ở lượt sau, ghép lại theo cùng video_id và cùng chỉ số frame.

Chạy:
    python -m src.data.extract_hands_kaggle --root <VSL400> --model-dir /kaggle/working/models \
        --out-dir /kaggle/working/hands [--limit 20] [--resume-from <thư mục output cũ> ...]

Đầu ra trong --out-dir:
    shard_<run>_<k>.npz  mỗi shard ~--shard-size video, các mảng nối theo frame:
        video_id  (V,)            str     : mã video
        offset    (V,)            int64   : frame đầu của video trong các mảng theo frame
        length    (V,)            int32   : T = số frame đọc được (frame 0..T-1, stride 1)
        fps, src_w, src_h (V,)            : thông tin video gốc
        lb        (V, 4)          int32   : letterbox [x0, y0, nw, nh] trên ảnh 640x480
        hands     (sumT, 2, 21, 3) float32 : x,y,z chuẩn hoá theo ảnh 640x480; ô 0 = nhãn 'Left', ô 1 = 'Right'; 0 nếu không thấy
        present   (sumT, 2)       bool    : ô tay có dữ liệu
        score     (sumT, 2)       float32 : điểm handedness
        same_label(sumT,)         bool    : 2 tay bị MediaPipe gán cùng nhãn
        hand_px   (sumT, 2)       float32 : cỡ bàn tay (px, cạnh dài khung bao) trên ảnh 640x480
    manifest.csv  1 dòng / video đã xử lý (kể cả lỗi)
    meta_<run>.json phiên bản python/mediapipe/opencv/numpy, sha256 model, cấu hình
Lấy 1 video: i = list(z["video_id"]).index(v); s = slice(z["offset"][i], z["offset"][i] + z["length"][i])
"""
import argparse
import csv
import json
import multiprocessing as mp_
import os
import platform
import time
from pathlib import Path

# Kernel Kaggle đặt MPLBACKEND=module://matplotlib_inline..., venv không có backend này;
# mediapipe import matplotlib -> lỗi. Ép backend không giao diện TRƯỚC khi import mediapipe.
os.environ["MPLBACKEND"] = "Agg"

import cv2
import numpy as np

from src.pi import extract_landmark as P

MANIFEST_COLS = ["video_id", "part", "path", "in_json", "status", "error", "shard",
                 "n_frames", "header_frames", "fps", "src_w", "src_h",
                 "ratio_ge1_hand", "ratio_2_hands", "n_same_label", "hand_px_p50", "seconds"]


# ---------- Worker: mỗi tiến trình một HandLandmarker ----------

class _NoPoseResult:
    pose_landmarks = []


class _NoPose:
    """Thay model pose: trả kết quả rỗng, để dùng lại nguyên Extractor.process cho phần tay."""
    def detect(self, img):
        return _NoPoseResult()

    def detect_for_video(self, img, ts_ms):
        return _NoPoseResult()

    def close(self):
        pass


class HandOnlyExtractor(P.Extractor):
    def __init__(self, cfg):
        # Tạo HandLandmarker đúng như P.Extractor.__init__, bỏ phần pose
        self.cfg = cfg
        th = cfg["hand_thresholds"]
        self.hand = P.vision.HandLandmarker.create_from_options(P.vision.HandLandmarkerOptions(
            base_options=P.BaseOptions(model_asset_path=cfg["hand_model"]),
            running_mode=P.vision.RunningMode[cfg["hand_mode"]], num_hands=cfg["num_hands"],
            min_hand_detection_confidence=th["detection"],
            min_hand_presence_confidence=th["presence"],
            min_tracking_confidence=th["tracking"]))
        self.pose = _NoPose()
        self.hand_video = cfg["hand_mode"] == "VIDEO"
        self.pose_video = False


_EX = None
_CFG = None


def _init_worker(cfg):
    global _EX, _CFG
    cv2.setNumThreads(1)  # song song theo video, tránh tranh nhân CPU
    _CFG = cfg
    _EX = HandOnlyExtractor(cfg)


def letterbox_params(fw, fh, w=P.IN_W, h=P.IN_H):
    # Cùng công thức với P.fit_frame(mode="letterbox")
    s = min(w / fw, h / fh)
    nw, nh = round(fw * s), round(fh * s)
    return [(w - nw) // 2, (h - nh) // 2, nw, nh]


def process_video(task):
    """task = dict(video_id, part, path(tuyệt đối), rel, in_json) -> dict kết quả (mảng numpy)."""
    t0 = time.perf_counter()
    info = dict(video_id=task["video_id"], part=task["part"], path=task["rel"], in_json=task["in_json"])
    cap = cv2.VideoCapture(task["path"])
    try:
        if not cap.isOpened():
            raise RuntimeError("không mở được video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_head = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = _CFG["stride"]
        hands, score, same, hpx = [], [], [], []
        idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            img = P.fit_frame(frame, _CFG["fit"])
            img = P.enhance(img, _CFG.get("enhance", "none"))
            r = _EX.process(img, int(round(idx * 1000.0 / fps)))
            hands.append(r["vec"][:126].reshape(2, P.N_HAND, 3))
            score.append(r["score"])
            same.append(r["same_label"])
            hpx.append(r["hand_px"])
        if not hands:
            raise RuntimeError("không đọc được frame nào")
    except Exception as e:  # ghi lỗi vào manifest, không dừng cả lượt chạy
        cap.release()
        return dict(info, status="error", error=str(e)[:200], seconds=round(time.perf_counter() - t0, 2))
    cap.release()

    score = np.array(score, np.float32)
    present = score > 0
    hpx = np.array(hpx, np.float32)
    n = present.sum(1)
    return dict(
        info, status="ok", error="",
        n_frames=len(hands), header_frames=n_head, fps=fps, src_w=fw, src_h=fh,
        lb=letterbox_params(fw, fh),
        hands=np.array(hands, np.float32), present=present, score=score,
        same_label=np.array(same, bool), hand_px=hpx,
        ratio_ge1_hand=round(float((n >= 1).mean()), 4), ratio_2_hands=round(float((n == 2).mean()), 4),
        n_same_label=int(np.sum(same)),
        hand_px_p50=round(float(np.median(hpx[present])), 1) if present.any() else 0.0,
        seconds=round(time.perf_counter() - t0, 2))


# ---------- Ghi shard + manifest ----------

def write_shard(out_dir, name, results):
    lengths = np.array([r["n_frames"] for r in results], np.int32)
    offset = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    path = out_dir / f"{name}.npz"
    tmp = out_dir / f"{name}.tmp.npz"
    np.savez_compressed(
        tmp,
        video_id=np.array([r["video_id"] for r in results]),
        offset=offset, length=lengths,
        fps=np.array([r["fps"] for r in results], np.float32),
        src_w=np.array([r["src_w"] for r in results], np.int32),
        src_h=np.array([r["src_h"] for r in results], np.int32),
        lb=np.array([r["lb"] for r in results], np.int32),
        hands=np.concatenate([r["hands"] for r in results]),
        present=np.concatenate([r["present"] for r in results]),
        score=np.concatenate([r["score"] for r in results]),
        same_label=np.concatenate([r["same_label"] for r in results]),
        hand_px=np.concatenate([r["hand_px"] for r in results]),
    )
    os.replace(tmp, path)  # ghi xong mới đổi tên -> không có shard dở dang


def append_manifest(out_dir, rows):
    f = out_dir / "manifest.csv"
    new = not f.exists()
    with open(f, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def done_ids(dirs):
    """video_id đã trích thành công ở các lượt trước (đọc manifest.csv)."""
    ids = set()
    for d in dirs:
        for m in Path(d).rglob("manifest.csv"):
            with open(m, encoding="utf-8") as fh:
                ids |= {r["video_id"] for r in csv.DictReader(fh) if r["status"] == "ok"}
    return ids


# ---------- Danh sách video ----------

def build_tasks(root, videos_csv):
    """Ưu tiên 1: video trong front_view.json (videos.csv). Ưu tiên 2: mp4 còn lại trong front_view/."""
    tasks, seen = [], set()
    with open(videos_csv, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            tasks.append(dict(video_id=r["video_id"], part=int(r["part"]), rel=r["path"],
                              path=str(root / r["path"]), in_json=True))
            seen.add(r["path"])
    extra = []
    for p in sorted(root.glob("Part_*/split_*/front_view/*.mp4")):
        rel = p.relative_to(root).as_posix()
        if rel not in seen:
            part = int(p.parts[-4].split("_")[1])
            extra.append(dict(video_id=f"x{part}_{p.stem}", part=part, rel=rel, path=str(p), in_json=False))
    return tasks, extra


def write_meta(out_dir, cfg, args, run_id):
    import mediapipe
    meta = dict(
        versions=dict(python=platform.python_version(), mediapipe=mediapipe.__version__,
                      opencv=cv2.__version__, numpy=np.__version__, machine=platform.machine()),
        model_sha256=dict(hand=P.sha256(cfg["hand_model"])),
        config=cfg, cpu_count=os.cpu_count(), workers=args.workers,
        args={k: str(v) for k, v in vars(args).items()},
        layout="hands (sumT,2,21,3): ô 0 = nhãn MediaPipe 'Left', ô 1 = 'Right', ảnh không lật gương",
    )
    with open(out_dir / f"meta_{run_id}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("VSL_ROOT", ""), help="thư mục VSL400")
    ap.add_argument("--videos", default="configs/data/videos.csv")
    ap.add_argument("--config", default="configs/extractor_config.json")
    ap.add_argument("--model-dir", default="data/models")
    ap.add_argument("--out-dir", default="/kaggle/working/hands")
    ap.add_argument("--resume-from", nargs="*", default=[], help="thư mục output cũ: bỏ qua video đã xong")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--time-budget-hours", type=float, default=11.0)
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N video đầu (chạy thử)")
    ap.add_argument("--no-extra", action="store_true", help="không trích mp4 ngoài front_view.json")
    ap.add_argument("--log-every-sec", type=float, default=120)
    args = ap.parse_args()
    if not args.root:
        raise SystemExit("Cần --root hoặc VSL_ROOT")
    t_start = time.time()
    budget = args.time_budget_hours * 3600

    cfg = P.load_config(args.config, args.model_dir)  # kiểm sha256 2 file .task
    root, out_dir = Path(args.root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    meta = write_meta(out_dir, cfg, args, run_id)
    print("Phiên bản:", meta["versions"], "| workers:", args.workers, flush=True)

    tasks, extra = build_tasks(root, args.videos)
    n_json = len(tasks)
    if not args.no_extra:
        tasks += extra
    done = done_ids([out_dir, *args.resume_from])
    todo = [t for t in tasks if t["video_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Tổng {len(tasks)} video (trong json {n_json}, ngoài json {len(extra)}"
          f"{', bỏ qua' if args.no_extra else ''}), đã xong {len(done)}, lượt này {len(todo)}", flush=True)
    if not todo:
        print("Không còn gì để trích.")
        return

    buf, n_ok, n_err, n_frames, k_shard = [], 0, 0, 0, 0
    t_last = time.time()
    stopped = False

    def flush():
        nonlocal buf, k_shard
        if not buf:
            return
        ok = [r for r in buf if r["status"] == "ok"]
        name = f"shard_{run_id}_{k_shard:03d}"
        if ok:
            write_shard(out_dir, name, ok)
            k_shard += 1
        for r in buf:
            r["shard"] = name if r["status"] == "ok" else ""
        append_manifest(out_dir, buf)
        buf = []

    pool = mp_.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(cfg,))
    try:
        for r in pool.imap_unordered(process_video, todo, chunksize=1):
            buf.append(r)
            if r["status"] == "ok":
                n_ok += 1
                n_frames += r["n_frames"]
            else:
                n_err += 1
                print(f"  LỖI {r['path']}: {r['error']}", flush=True)
            if len(buf) >= args.shard_size:
                flush()
            el = time.time() - t_start
            n = n_ok + n_err
            if time.time() - t_last >= args.log_every_sec or n == len(todo):
                t_last = time.time()
                rate = n / el * 60
                eta = (len(todo) - n) / max(rate, 1e-9)
                print(f"[{el / 60:6.1f} phút] {n}/{len(todo)} video (lỗi {n_err}) | {rate:.1f} video/phút | "
                      f"{n_frames / el:.0f} frame/s | còn ~{eta / 60:.1f} giờ", flush=True)
            if el > budget:
                print(f"Hết ngân sách {args.time_budget_hours} giờ -> dừng, lưu phần đã xong.", flush=True)
                stopped = True
                break
    finally:
        flush()
        if stopped:
            pool.terminate()
        else:
            pool.close()
        pool.join()

    el = time.time() - t_start
    print(f"\nXONG lượt: {n_ok} ok, {n_err} lỗi, {n_frames} frame trong {el / 60:.1f} phút "
          f"({n_frames / max(el, 1e-9):.0f} frame/s). Còn lại: {len(todo) - n_ok - n_err} video.")
    print(f"Output: {out_dir} (shard_*.npz, manifest.csv, meta_*.json)")


if __name__ == "__main__":
    main()
