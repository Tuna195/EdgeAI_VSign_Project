"""Vẽ bảng tóm tắt một thí nghiệm (runs/summary_<Ek>.csv): val top-1 trung bình ± độ lệch giữa 3 seed.
Chạy trong Colab: %run -m src.explore.plot_summary E1"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

from src.core.paths import RUNS_DIR

exp = sys.argv[1] if len(sys.argv) > 1 else "E1"
s = pd.read_csv(RUNS_DIR / f"summary_{exp}.csv", encoding="utf-8-sig")
colors = ["tab:orange" if w else "tab:blue" for w in s.winner]
plt.figure(figsize=(1.8 * len(s) + 2, 4))
plt.bar(s.value.astype(str), s.val_top1, yerr=s.val_top1_std, capsize=6, color=colors)
for i, r in s.iterrows():
    plt.text(i, r.val_top1 + r.val_top1_std + 0.003, f"{r.val_top1:.3f} ± {r.val_top1_std:.3f}", ha="center")
plt.ylim(max(0.0, s.val_top1.min() - 0.1), 1.0)
plt.ylabel("val top-1 (trung bình 3 seed)")
plt.title(f"{exp} (cam = thắng; vạch đen = ± độ lệch giữa các seed)")
plt.show()
