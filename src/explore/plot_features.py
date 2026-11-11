"""Vẽ 1 frame TRƯỚC và SAU chuẩn hoá theo vai. 
Chạy python -m src.explore.plot_features"""
import matplotlib.pyplot as plt

from src.core.features import make_features, points, sample_positions
from src.dataset.build_sequences import load_sequences

KEY, ROW = "test/Bố/003266", 15
seq = load_sequences()[KEY]
idx = sample_positions(len(seq), 30)
feat = make_features(seq, idx, {"norm": "shoulder", "use_z": False, "n_frames": 30})    # (30, 108)

raw = points(seq)[idx[ROW], :, :2]                    # 75 điểm (x, y) gốc của frame đó
row = feat[ROW]                                       # 108 số: 42 + 42 + 22 + 2
before = [raw[:21], raw[21:42], raw[42:]]
after = [row[0:42].reshape(21, 2), row[42:84].reshape(21, 2), row[84:106].reshape(11, 2)]
print("Cờ tay trái, tay phải:", row[106:108])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, parts, title in zip(axes, (before, after), ("Trước: toạ độ ảnh", "Sau: gốc giữa 2 vai, đơn vị = độ rộng vai")):
    for p, color, name in zip(parts, ("tab:blue", "tab:red", "gray"), ("tay trái", "tay phải", "thân")):
        ax.scatter(p[:, 0], p[:, 1], s=12, c=color, label=name)
    ax.axhline(0, lw=0.5, c="k")
    ax.axvline(0, lw=0.5, c="k")
    ax.invert_yaxis()                                 # y hướng xuống như trong ảnh
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(fontsize=8)
plt.suptitle(f"{KEY}, dòng {ROW} (frame {idx[ROW]})")
plt.show()