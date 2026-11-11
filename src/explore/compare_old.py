"""So index.csv và sequences.npz mới với bản cũ (giải nén từ vsign_colab.zip).

Chạy: python -m src.explore.compare_old /content/old
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.paths import INDEX_FILE, SEQ_FILE

old_dir = Path(sys.argv[1]) / "data" / "processed"
new = pd.read_csv(INDEX_FILE, encoding="utf-8-sig").set_index("key")
old = pd.read_csv(old_dir / "index.csv", encoding="utf-8-sig").set_index("key")
both = new.index.intersection(old.index)
print(f"Số video: mới {len(new)}, cũ {len(old)}, có ở cả hai {len(both)}")
for col in ["word", "label", "source", "split", "frames", "width", "height"]:
    print(f"  cột {col:7s}: giống {(new.loc[both, col] == old.loc[both, col]).sum()}/{len(both)}")

if SEQ_FILE.exists() and (old_dir / "sequences.npz").exists():      # dùng ở Phần 3
    a, b = np.load(SEQ_FILE), np.load(old_dir / "sequences.npz")
    same_keys = list(a["keys"]) == list(b["keys"])
    same_offsets = np.array_equal(a["offsets"], b["offsets"])
    print(f"sequences.npz: cùng danh sách video {same_keys}, cùng độ dài từng chuỗi {same_offsets}")
    if same_keys and same_offsets:
        print(f"  sai lệch lớn nhất giữa hai bản: {np.abs(a['data'] - b['data']).max():.2e}")