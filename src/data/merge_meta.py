"""Gộp metadata VSL400 (chỉ front_view) thành một bảng videos.csv.

Input : <root>/Part_*/split_*/front_view.json  (+ video <root>/Part_*/split_*/front_view/<video_id>.mp4)
Output: videos.csv, mỗi dòng là một video front_view có trong JSON.

Chạy (trên Kaggle):
    python -m src.data.merge_meta --root /kaggle/input/.../raw/raw/VSL400 --out configs/data/videos.csv
"""
import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = os.environ.get("VSL_ROOT", "")


def natural_key(p):
    # "Part_10" xếp sau "Part_2"
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", str(p))]


def clean_text(s):
    # Chuẩn hoá Unicode NFC: tránh cùng một từ tiếng Việt bị đếm thành 2 gloss
    return unicodedata.normalize("NFC", str(s).strip())


def read_view_ids(split_path, view):
    """Tập video_id trong <view>.json (dùng để biết file thừa thuộc góc nhìn nào)."""
    f = split_path / f"{view}.json"
    if not f.exists():
        return set()
    with open(f, encoding="utf-8") as fh:
        return {str(r["video_id"]) for r in json.load(fh)}


def read_header(path):
    """Đọc width, height, fps, số frame từ header video (không giải mã hình)."""
    import cv2  # import muộn để --no-header chạy được khi máy không có cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    h = dict(
        vid_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        vid_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        vid_fps=round(cap.get(cv2.CAP_PROP_FPS), 3),
        vid_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return h


def collect(root):
    """Đọc mọi front_view.json → list dict; đồng thời thống kê file video thừa."""
    rows, extra = [], []
    json_files = sorted(root.glob("Part_*/split_*/front_view.json"), key=natural_key)
    if not json_files:
        raise SystemExit(f"Không thấy front_view.json nào dưới {root}")

    for jf in json_files:
        split_path = jf.parent
        part = int(split_path.parent.name.split("_")[1])
        split_dir = split_path.name
        with open(jf, encoding="utf-8") as fh:
            records = json.load(fh)

        ids = set()
        for r in records:
            vid = str(r["video_id"])
            ids.add(vid)
            rows.append(dict(
                part=part,
                split_dir=split_dir,
                video_id=vid,
                signer_id=str(r["signer_id"]),
                gloss_raw=str(r["gloss"]),
                gloss=clean_text(r["gloss"]),
                num_frames=r.get("num_frames"),
                fps=r.get("fps"),
                length_seconds=r.get("length_seconds"),
                resolution=r.get("resolution"),
                path=f"{split_path.parent.name}/{split_dir}/front_view/{vid}.mp4",
            ))

        # File .mp4 trong front_view/ mà không có trong front_view.json
        on_disk = {p.stem for p in (split_path / "front_view").glob("*.mp4")}
        left, right = read_view_ids(split_path, "left_view"), read_view_ids(split_path, "right_view")
        not_in_json = on_disk - ids
        extra.append(dict(
            part=part, split_dir=split_dir,
            json_records=len(records), mp4_on_disk=len(on_disk),
            missing_file=len(ids - on_disk),
            extra_file=len(not_in_json),
            extra_in_left_json=len(not_in_json & left),
            extra_in_right_json=len(not_in_json & right),
            extra_in_no_json=len(not_in_json - left - right),
            left_records=len(left), right_records=len(right),
        ))
    return pd.DataFrame(rows), pd.DataFrame(extra)


def add_headers(df, root):
    heads = []
    for i, p in enumerate(df["path"]):
        f = root / p
        h = read_header(f) if f.exists() else None
        heads.append(h or {})
        if (i + 1) % 1000 == 0:
            print(f"  đã đọc header {i + 1}/{len(df)}", flush=True)
    hdf = pd.DataFrame(heads, index=df.index,
                       columns=["vid_width", "vid_height", "vid_fps", "vid_frames"])
    for c in ["vid_width", "vid_height", "vid_frames"]:
        hdf[c] = hdf[c].astype("Int64")  # số nguyên, cho phép ô trống
    df = pd.concat([df, hdf], axis=1)
    df["vid_aspect"] = (df["vid_width"] / df["vid_height"]).round(4)
    return df


def report(df, extra, with_header):
    print("\n=== Tổng quan ===")
    print(f"Số bản ghi JSON: {len(df)}  | số split: {len(extra)}")
    print(f"Số gloss (sau NFC): {df['gloss'].nunique()}  | trước NFC: {df['gloss_raw'].nunique()}")
    print(f"Số người ký: {df['signer_id'].nunique()}")

    print("\n=== File theo từng split ===")
    print(extra.to_string(index=False))

    print("\n=== Trùng khoá ===")
    print(f"video_id trùng trên toàn bộ: {df['video_id'].duplicated().sum()}")
    print(f"(part, split_dir, video_id) trùng: {df.duplicated(['part', 'split_dir', 'video_id']).sum()}")

    na = df[["signer_id", "gloss", "num_frames", "fps"]].isna().sum()
    print(f"\nÔ trống trong cột JSON:\n{na.to_string()}")

    print(f"\nVideo thiếu file: {(~df['exists']).sum()}")
    if not with_header:
        return

    ok = df["exists"]
    bad_open = ok & df["vid_width"].isna()
    print(f"Video có file nhưng cv2 không mở được: {bad_open.sum()}")

    d = df[ok & df["vid_width"].notna()]
    frame_diff = (d["num_frames"] - d["vid_frames"]).abs()
    fps_diff = (d["fps"] - d["vid_fps"]).abs()
    res_bad = (d["resolution"] != d["vid_width"]) & (d["resolution"] != d["vid_height"])
    print("\n=== Lệch JSON vs header ===")
    print(f"|num_frames - vid_frames| > 1 : {(frame_diff > 1).sum()}  (lệch lớn nhất {frame_diff.max()})")
    print(f"|fps - vid_fps| > 0.5        : {(fps_diff > 0.5).sum()}")
    print(f"resolution khác cả W lẫn H    : {res_bad.sum()}")
    print("\nĐộ phân giải thật (W x H) : số video")
    print(d.groupby(["vid_width", "vid_height"]).size().to_string())
    print("\nfps thật : số video")
    print(d["vid_fps"].value_counts().sort_index().to_string())
    if (frame_diff > 1).any():
        print("\nVí dụ lệch số frame:")
        print(d.loc[frame_diff > 1, ["path", "num_frames", "vid_frames"]].head(10).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="thư mục VSL400 (chứa Part_1..Part_7); hoặc đặt biến VSL_ROOT")
    ap.add_argument("--out", default="configs/data/videos.csv")
    ap.add_argument("--no-header", action="store_true", help="bỏ đọc header video (chạy nhanh)")
    args = ap.parse_args()
    if not args.root:
        raise SystemExit("Cần --root hoặc biến môi trường VSL_ROOT")
    root = Path(args.root)

    df, extra = collect(root)
    df["exists"] = [(root / p).exists() for p in df["path"]]

    # Nhãn 0..N-1 theo gloss sắp xếp (đã NFC) → cố định, chạy lại ra y hệt
    glosses = sorted(df["gloss"].unique())
    df["label"] = df["gloss"].map({g: i for i, g in enumerate(glosses)})

    if not args.no_header:
        print("Đọc header video...")
        df = add_headers(df, root)

    report(df, extra, with_header=not args.no_header)

    cols = ["part", "split_dir", "video_id", "signer_id", "gloss", "label",
            "num_frames", "fps", "length_seconds", "resolution", "path", "exists"]
    if not args.no_header:
        cols += ["vid_width", "vid_height", "vid_fps", "vid_frames", "vid_aspect"]
    # Sắp xếp: part, số thứ tự split, video_id (chuỗi có số 0 đầu nên so chuỗi được)
    df["_split_num"] = df["split_dir"].str.extract(r"(\d+)")[0].astype(int)
    df = df.sort_values(["part", "_split_num", "video_id"]).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(out, index=False, encoding="utf-8-sig")  # có BOM để Excel đọc đúng tiếng Việt
    print(f"\nĐã ghi {len(df)} dòng → {out}")


if __name__ == "__main__":
    main()
