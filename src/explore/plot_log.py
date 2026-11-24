"""Vẽ đường học (log.csv) của các lần chạy. Chạy trong Colab:
  %run -m src.explore.plot_log gru_shoulder_z_f30_aug_wall/seed0 mlp_shoulder_z_f30_aug_wall/seed0"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

from src.core.paths import RUNS_DIR

runs = sys.argv[1:] or ["gru_shoulder_z_f30_aug_wall/seed0"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
for run in runs:
    log = pd.read_csv(RUNS_DIR / run / "log.csv")
    best = log.loc[log.val_top1.idxmax(), "epoch"]                 # epoch val tốt nhất (gần đúng: bằng nhau thì lấy đầu)
    line, = ax1.plot(log.epoch, log.val_loss, label=f"{run} val")
    ax1.plot(log.epoch, log.train_loss, ls="--", c=line.get_color(), label=f"{run} train")
    ax2.plot(log.epoch, log.val_top1, c=line.get_color(), label=f"{run} val")
    ax2.plot(log.epoch, log.train_top1, ls="--", c=line.get_color(), label=f"{run} train (có augmentation)")
    ax2.axvline(best, c=line.get_color(), lw=0.8, ls=":")
ax1.set_title("Loss")
ax2.set_title("Top-1 (chấm chấm: epoch val tốt nhất)")
for ax in (ax1, ax2):
    ax.set_xlabel("epoch")
    ax.legend(fontsize=7)
plt.show()
