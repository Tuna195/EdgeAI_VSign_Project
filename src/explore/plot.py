from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Các cặp điểm nối thành khung xương
HAND = [(0, 1), (1, 2), (2, 3), (3, 4),            # ngón cái
        (0, 5), (5, 6), (6, 7), (7, 8),            # ngón trỏ
        (5, 9), (9, 10), (10, 11), (11, 12),       # ngón giữa
        (9, 13), (13, 14), (14, 15), (15, 16),     # ngón áp út
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]   # ngón út + lòng bàn tay
POSE = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # vai, hai cánh tay
        (0, 2), (0, 5), (9, 10)]                            # mũi - mắt, miệng

def load(key):
    """ train/Bố/123456 -> mảng (T, 75,3)"""
    return np.load(Path("data/raw")/f"{key}.npy").reshape(-1,75,3)

def draw_frame(x, f, ax=None):
    """Vẽ bộ xương của frame f. Tay trái xanh nước, phải cam, thân xanh lá
    x: dữ liệu cần vẽ (T, 75, 3)
    f: chỉ số frame cần vẽ
    ax: đối tượng axes để vẽ lên
    """
    ax = ax or plt.gca()
    p = x[f]
    for start, color in ((0, "tab:blue"), (21, "tab:orange")):
        hand = p[start:start+21]
        if hand.any():
            for a,b in HAND:
                ax.plot(hand[[a, b], 0], hand[[a, b], 1], color=color, lw=1)
    body = p[42:]
    if body.any():
        for a,b in POSE:
            ax.plot(body[[a, b], 0], body[[a, b], 1], color="green")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # lật trục y: y tăng xuống dưới như trong ảnh
    ax.set_aspect("equal")
    ax.set_title(f"frame {f}")

def plot_wrists(x, ax=None):
    """Vẽ x, y của 2 cổ tay theo thời gian. Frame không thấy tay -> để trống"""
    ax = ax or plt.gca()
    for start, name, color in ((0, "left", "blue"), (21, "right", "orange")):
        w = x[:, start, :2].copy()
        w[~x[:, start:start+21].any(axis=(1,2))] = np.nan
        ax.plot(w[:, 0], color=color, label=f"{name} wrist: x")
        ax.plot(w[:, 1], color=color, ls='--', label=f"{name} wrist: y")
    ax.set_xlabel("frame")
    ax.legend(fontsize=8)

if __name__ == "__main__":
    f = sorted(Path("data/raw/train/Bố").glob("*.npy"))[0]
    key = f.relative_to("data/raw").with_suffix("").as_posix()    # 'train/Bố/<id>'
    x = load(key)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, t in zip(axes, np.linspace(0, len(x) - 1, 4).round().astype(int)):
        draw_frame(x, t, ax)
    fig.suptitle(key)
    plt.show()

    plt.figure(figsize=(10, 3))
    plot_wrists(x)
    plt.title(key)
    plt.show()
