from pathlib import Path
from unicodedata import normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nfc = lambda s : normalize("NFC", s.strip()) # chuẩn hóa chữ có dấu

# 1. Bảng mọi video: mỗi video 1 dòng
rows = []
for f in sorted(Path("data/raw").glob("*/*/*.npy")):
	rows.append({"key": nfc(f.relative_to("data/raw").with_suffix("").as_posix()),
	             "orig_split": f.parts[-3],
				 "word": nfc(f.parent.name),
				 "frames": len(np.load(f))})
df = pd.DataFrame(rows)
df["source"] = np.where(df.frames > 30, "A", "B")
print(df.groupby("source").frames.describe())

# 2. Video bị loại
lines = [l for l in Path("data/exclude.txt").read_text(encoding="utf-8").splitlines()
         if l.strip() and not l.startswith("#")]
excluded = {nfc(l) for l in lines}
print("Số video trong exclude.txt:", len(lines), "| video khác nhau:", len(excluded))
df["excluded"] = df.key.isin(excluded)
print(f"{len(df)} video -> loại {df.excluded.sum()} -> còn {(~df.excluded).sum()}")

# 3. Các quyết định duyệt
dec = pd.read_csv("data/reports/review_decisions.csv", encoding="utf-8-sig")
print("Số quyết định:", len(dec))
print(pd.crosstab(dec.queue, dec.decision, margins=True))

# 4. Biểu đồ độ dài video theo nguồn
axes = df.frames.hist(by=df.source, bins=30, figsize=(10, 3))
for ax in axes.flat:
    ax.set_xlabel("Số frame")
    ax.set_ylabel("Số video")
plt.tight_layout()
plt.show()