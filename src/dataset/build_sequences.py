"""index.csv + data/raw -> data/processed/sequences.npz: mỗi video một chuỗi (T, 225) đã làm sạch, CHƯA chuẩn hoá.

Chạy: python -m src.dataset.build_sequences   (sau index_kaggle)
"""
import numpy as np
import pandas as pd

from src.core.features import prepare_sequence
from src.core.paths import INDEX_FILE, RAW_DIR, SEQ_FILE


def load_sequences(path=SEQ_FILE):
    """-> {key: chuỗi (T, 225) float32}"""
    with np.load(path) as z:                                 # đọc hết 3 mảng TRƯỚC khi file đóng
        data, offsets, keys = z["data"], z["offsets"], z["keys"]
    return {str(key): data[offsets[i]:offsets[i + 1]] for i, key in enumerate(keys)}

def main():
    index = pd.read_csv(INDEX_FILE, encoding="utf-8-sig")
    keys, seqs, cut = [], [], []
    for r in index.itertuples():                             # r.key, r.width, r.height... của từng dòng
        raw = np.load(RAW_DIR / f"{r.key}.npy")
        seq, reason = prepare_sequence(raw, int(r.width), int(r.height))
        if seq is None:
            print(f"BỎ {r.key}: {reason}")
            continue
        keys.append(r.key)
        seqs.append(seq)
        cut.append(len(raw) - len(seq))                      # số frame trống đã cắt ở 2 đầu

    lengths = np.array([len(s) for s in seqs])
    offsets = np.concatenate([[0], np.cumsum(lengths)])      # [0, 57, 120, ...]
    np.savez_compressed(SEQ_FILE, keys=np.array(keys), offsets=offsets, data=np.concatenate(seqs))

    cut = np.array(cut)
    print(f"Giữ {len(keys)}/{len(index)} video -> {SEQ_FILE}")
    print(f"Cắt frame trống ở 2 đầu: {(cut > 0).sum()} video, tổng {cut.sum()} frame")
    source = index.set_index("key").source[keys]
    print(pd.Series(lengths, index=keys).groupby(source).agg(["count", "min", "median", "max"]).to_string())


if __name__ == "__main__":
    main()