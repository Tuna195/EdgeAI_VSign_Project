"""Thống kê VSL400 từ videos.csv (không cần video, chạy được ở bất kỳ đâu).

Input : configs/data/videos.csv (output của merge_meta)
Output: in ra màn hình + configs/data/signer_gloss_counts.csv (người ký × từ, ô = số lần ký)

Chạy:
    python -m src.data.stats --videos configs/data/videos.csv
"""
import argparse
from pathlib import Path

import pandas as pd

FPS_LEVELS = [(1, 25.0), (2, 12.5), (3, 8.33), (4, 6.25), (5, 5.0)]  # (stride, fps) cho E5


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def load(path):
    # Giữ số 0 đầu của video_id, signer_id
    return pd.read_csv(path, dtype={"video_id": str, "signer_id": str}, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="configs/data/videos.csv")
    ap.add_argument("--out-counts", default="configs/data/signer_gloss_counts.csv")
    args = ap.parse_args()
    df = load(args.videos)
    n_gloss = df["gloss"].nunique()

    section("1. Theo Part")
    by_part = df.groupby("part").agg(
        split_dir=("split_dir", "first"), n_video=("video_id", "size"),
        n_signer=("signer_id", "nunique"), n_gloss=("gloss", "nunique"),
        vid_min=("video_id", "min"), vid_max=("video_id", "max"))
    print(by_part.to_string())
    print("\nNgười ký × Part (số video):")
    print(pd.crosstab(df["signer_id"], df["part"], margins=True).to_string())

    section("2. Theo người ký")
    counts = pd.crosstab(df["signer_id"], df["gloss"])  # 26 × 400
    by_signer = pd.DataFrame({
        "n_video": df.groupby("signer_id").size(),
        "n_gloss": (counts > 0).sum(axis=1),
        "parts": df.groupby("signer_id")["part"].unique().map(lambda p: ",".join(map(str, sorted(p)))),
    })
    print(by_signer.to_string())
    print("\nTừ còn thiếu của từng người:")
    for s, row in counts.iterrows():
        miss = list(row[row == 0].index)
        if miss:
            print(f"  {s}: {miss}")
    print("\nSố người ký mỗi từ (từ nào thiếu người):")
    signers_per_gloss = (counts > 0).sum(axis=0)
    print(signers_per_gloss.value_counts().sort_index().rename("n_gloss").to_string())

    section("3. Số lần lặp mỗi cặp (người, từ)")
    reps = counts.stack()
    print(reps.value_counts().sort_index().rename("n_cặp").to_string())
    print(f"Trung bình {reps.mean():.2f} lần, trung vị {reps.median():.0f}")
    print("\nTrung bình số lần lặp theo người ký:")
    print(counts.replace(0, pd.NA).mean(axis=1).round(2).to_string())

    section(f"4. Người ký đủ {n_gloss} từ")
    full = by_signer.index[by_signer["n_gloss"] == n_gloss].tolist()
    print(f"{len(full)} người: {full}")
    print(f"Không đủ: {by_signer.index[by_signer['n_gloss'] < n_gloss].tolist()}")

    section("5. Độ dài video")
    print(df[["num_frames", "length_seconds"]].describe().round(2).to_string())
    print("\nSố frame theo Part:")
    print(df.groupby("part")["num_frames"].describe().round(1).to_string())
    print("\nSố frame theo người ký:")
    print(df.groupby("signer_id")["num_frames"].describe().round(1).to_string())
    print("\nSố frame còn lại khi lấy mẫu theo stride (E5):")
    rows = []
    for stride, fps in FPS_LEVELS:
        n = (df["num_frames"] + stride - 1) // stride  # frame 0, s, 2s, ...
        rows.append(dict(stride=stride, fps=fps, min=n.min(), p5=n.quantile(0.05),
                         median=n.median(), p95=n.quantile(0.95), max=n.max()))
    print(pd.DataFrame(rows).to_string(index=False))

    section("6. Độ phân giải, fps, video hỏng")
    print(df.groupby(["vid_width", "vid_height"], dropna=False).size().rename("n").to_string())
    print(df["vid_fps"].value_counts(dropna=False).rename("n").to_string())
    bad = df[~df["exists"] | df["vid_width"].isna()]
    print(f"\nVideo thiếu hoặc không mở được: {len(bad)}")
    if len(bad):
        print(bad[["video_id", "signer_id", "gloss", "path"]].to_string(index=False))

    # Kiểm tra tổng
    assert counts.values.sum() == len(df), "tổng bảng người ký × từ phải bằng số video"
    out = Path(args.out_counts)
    counts.to_csv(out, encoding="utf-8-sig")
    print(f"\nTổng video {len(df)} = tổng bảng người ký × từ {counts.values.sum()}  ✓")
    print(f"Đã ghi {counts.shape} → {out}")


if __name__ == "__main__":
    main()
