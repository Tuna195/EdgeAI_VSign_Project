"""Chạy chuỗi thí nghiệm E0-E6 (ke_hoach.md mục 5): mỗi lần đổi 1 yếu tố, cấu hình thắng truyền sang thí nghiệm sau.

Chạy:
  python -m src.training.experiments E0 E1          chạy E0 rồi E1 (lần đã chạy xong thì bỏ qua)
  python -m src.training.experiments --summary E1   chỉ in bảng kết quả, không train
  python -m src.training.experiments --list         in danh sách thí nghiệm và cấu hình gốc của từng cái

Mỗi cấu hình chạy với seed {0, 1, 2}, báo trung bình ± độ lệch chuẩn. Thắng = val top-1 trung bình cao nhất.
Nếu chênh lệch với hạng 2 nhỏ hơn độ lệch giữa các seed thì bảng ghi "hoà" (vẫn lấy cấu hình cao hơn để đi tiếp,
nhưng nên cân nhắc chọn cái đơn giản / nhanh hơn).
Test KHÔNG chạy ở đây: chỉ chạy cho cấu hình cuối cùng bằng train.py --test.
Bảng kết quả: runs/summary_<Ek>.csv
"""
import argparse
import csv
import json

import numpy as np
import torch

from src.core.paths import RUNS_DIR
from src.training.train import DEFAULTS, run, run_name

SEEDS = [0, 1, 2]
# tên: (mô tả, yếu tố thay đổi, các giá trị). E0 đặc biệt: cấu hình gốc + MLP làm mốc, chỉ seed 0.
EXPERIMENTS = {
    "E0": ("Chạy thử: GRU vs mốc MLP", "model", ["gru", "mlp"]),
    "E1": ("Chuẩn hoá", "norm", ["none", "shoulder", "shoulder_wrist"]),
    "E2": ("z", "use_z", [True, False]),
    "E3": ("Số frame", "n_frames", [16, 20, 30]),
    "E4": ("Model", "model", ["mlp", "gru", "bigru", "conv1d"]),
    "E5": ("Augmentation", "augment", [True, False]),
    "E6": ("Số từ (0 = tất cả)", "n_words", [10, 20, 0]),
}
CHAIN = ["E1", "E2", "E3", "E4", "E5"]   # thắng ở các thí nghiệm này được truyền sang thí nghiệm sau


def seeds_of(exp):
    return [0] if exp == "E0" else SEEDS


def load_metrics(cfg, seed):
    f = RUNS_DIR / run_name(cfg) / f"seed{seed}" / "metrics.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def base_config(exp):
    """Cấu hình gốc của thí nghiệm exp = DEFAULTS + giá trị thắng của các thí nghiệm trước nó trong CHAIN."""
    cfg = dict(DEFAULTS)
    for prev in CHAIN[:CHAIN.index(exp)] if exp in CHAIN else (CHAIN if exp == "E6" else []):
        rows = summarize(prev, cfg)
        if any(r["done"] < len(seeds_of(prev)) for r in rows):
            raise SystemExit(f"{exp} cần kết quả đầy đủ của {prev} (chạy: python -m src.training.experiments {prev})")
        cfg[EXPERIMENTS[prev][1]] = winner(rows)["value"]
    return cfg


def configs(exp, base):
    _, key, values = EXPERIMENTS[exp]
    return [(v, {**base, key: v}) for v in values]


def summarize(exp, base):
    """Mỗi giá trị của yếu tố đang thử: trung bình ± độ lệch các chỉ số trên những seed đã chạy xong."""
    rows = []
    for value, cfg in configs(exp, base):
        ms = [m for m in (load_metrics(cfg, s) for s in seeds_of(exp)) if m]
        row = {"value": value, "name": run_name(cfg), "done": len(ms)}
        for k in ("val_top1", "val_top3", "val_macro", "train_top1", "best_epoch", "train_sec", "infer_ms", "params"):
            vals = [m[k] for m in ms]
            row[k] = float(np.mean(vals)) if vals else float("nan")
            row[k + "_std"] = float(np.std(vals)) if vals else float("nan")
        rows.append(row)
    return rows


def winner(rows):
    return max(rows, key=lambda r: r["val_top1"])


def print_summary(exp, base):
    desc, key, _ = EXPERIMENTS[exp]
    rows = summarize(exp, base)
    best = winner(rows)
    others = sorted((r for r in rows if r is not best), key=lambda r: -r["val_top1"])
    tie = others and best["val_top1"] - others[0]["val_top1"] < max(best["val_top1_std"], others[0]["val_top1_std"])
    fixed = ", ".join(f"{k}={v}" for k, v in base.items() if k in ("model", "norm", "use_z", "n_frames", "augment", "n_words")
                      and k != key)
    print(f"\n{exp}. {desc} — thay đổi '{key}', giữ cố định: {fixed}")
    print(f"  {'giá trị':16s} {'seed':>4s}  {'val top1':>15s}  {'val top3':>15s}  {'val macro':>15s}  "
          f"{'train top1':>10s}  {'epoch':>5s}  {'tham số':>9s}  {'ms/mẫu':>6s}  {'giây':>5s}")
    for r in rows:
        mark = " *" if r is best else "  "
        print(f"{mark}{str(r['value']):16s} {r['done']:4d}  "
              f"{r['val_top1']:.3f} ± {r['val_top1_std']:.3f}  {r['val_top3']:.3f} ± {r['val_top3_std']:.3f}  "
              f"{r['val_macro']:.3f} ± {r['val_macro_std']:.3f}  {r['train_top1']:10.3f}  {r['best_epoch']:5.0f}  "
              f"{r['params']:9,.0f}  {r['infer_ms']:6.2f}  {r['train_sec']:5.0f}")
    print(f"  * = thắng (val top-1 trung bình cao nhất): {key} = {best['value']}"
          + ("   [HOÀ: chênh lệch với hạng 2 nhỏ hơn độ lệch giữa các seed]" if tie else ""))
    RUNS_DIR.mkdir(exist_ok=True)
    with open(RUNS_DIR / f"summary_{exp}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["winner", "tie"])
        w.writeheader()
        w.writerows({**{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()},
                     "winner": r is best, "tie": bool(tie) and r is best} for r in rows)


def run_experiment(exp, device="cpu"):
    base = base_config(exp)
    todo = [(cfg, s) for _, cfg in configs(exp, base) for s in seeds_of(exp)]
    print(f"\n######## {exp}. {EXPERIMENTS[exp][0]}: {len(todo)} lần chạy ########")
    for k, (cfg, seed) in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}]", end=" ")
        run(cfg, seed, device)
    print_summary(exp, base)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="*", metavar="Ek", help=", ".join(EXPERIMENTS))
    p.add_argument("--summary", action="store_true", help="chỉ in bảng kết quả")
    p.add_argument("--list", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.list:
        for exp, (desc, key, values) in EXPERIMENTS.items():
            print(f"{exp}: {desc} — {key} ∈ {values}, seed {seeds_of(exp)}")
        return
    unknown = [e for e in args.experiments if e not in EXPERIMENTS]
    if unknown:
        p.error(f"không có thí nghiệm {unknown}; chọn trong {list(EXPERIMENTS)}")
    for exp in args.experiments:
        if args.summary:
            print_summary(exp, base_config(exp))
        else:
            run_experiment(exp, args.device)


if __name__ == "__main__":
    main()
