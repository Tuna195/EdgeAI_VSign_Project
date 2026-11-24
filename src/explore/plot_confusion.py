"""Vẽ ma trận nhầm lẫn đã lưu bởi `train --report`. Chạy trong Colab:
  %run -m src.explore.plot_confusion gru_shoulder_wrist_z_f30_aug_wall/seed0 val"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

from src.core.paths import RUNS_DIR

run, split = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "val")
conf = pd.read_csv(RUNS_DIR / run / f"{split}_confusion.csv", encoding="utf-8-sig", index_col=0)
plt.figure(figsize=(10, 9))
plt.imshow(conf.values, cmap="Blues")
plt.xticks(range(len(conf)), conf.columns, rotation=90)
plt.yticks(range(len(conf)), conf.index)
for i in range(len(conf)):
    for j in range(len(conf)):
        if conf.values[i, j]:
            plt.text(j, i, conf.values[i, j], ha="center", va="center", fontsize=7,
                     color="white" if i == j else "red")
plt.xlabel("model đoán")
plt.ylabel("từ đúng")
plt.title(f"Ma trận nhầm lẫn: {run}, {split} (ô đỏ ngoài đường chéo = nhầm)")
plt.tight_layout()
plt.show()
