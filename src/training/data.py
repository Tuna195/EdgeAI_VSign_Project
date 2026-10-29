"""Đọc index.csv + sequences.npz, tạo tensor (N, D) cho từng mẫu theo cấu hình đặc trưng.

Val/test: lấy N frame cách đều, không biến đổi -> tính 1 lần.
Train: mỗi epoch tính lại, mỗi mẫu 1 augmentation + 1 kiểu đổi tốc độ ký ngẫu nhiên (theo rng có seed).
"""
import csv

import numpy as np

from src.core.features import make_features, sample_positions
from src.core.paths import INDEX_FILE, SEQ_FILE
from src.dataset.build_sequences import load_sequences
from src.training.augment import augment

WORD_SUBSET_SEED = 42   # E6 (ít từ hơn): chọn ngẫu nhiên k từ bằng seed cố định


def load_split_data(n_words=None):
    """-> (words, {"train"/"val"/"test": (danh sách chuỗi, nhãn np.array)}).

    Chỉ lấy video có trong cả index.csv và sequences.npz (build_sequences có thể đã bỏ vài video).
    n_words: chỉ dùng k từ (chọn ngẫu nhiên, seed cố định), nhãn đánh lại 0..k-1 theo thứ tự chữ cái.
    """
    with open(INDEX_FILE, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    seqs = load_sequences(SEQ_FILE)
    words = sorted({r["word"] for r in rows})
    if n_words and n_words < len(words):
        rng = np.random.default_rng(WORD_SUBSET_SEED)
        words = sorted(rng.choice(words, size=n_words, replace=False).tolist())
    label = {w: i for i, w in enumerate(words)}
    out = {}
    for split in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == split and r["word"] in label and r["key"] in seqs]
        out[split] = ([seqs[r["key"]] for r in rs], np.array([label[r["word"]] for r in rs]))
    return words, out


def build_features(seqs, cfg, rng=None, use_augment=False):
    """Danh sách chuỗi -> (số mẫu, N, D) float32. rng = None: không ngẫu nhiên (val/test)."""
    n = cfg["n_frames"]
    out = []
    for seq in seqs:
        if rng is not None and use_augment:
            seq = augment(seq, rng)
            idx = sample_positions(len(seq), n, rng)
        else:
            idx = sample_positions(len(seq), n)
        out.append(make_features(seq, idx, cfg))
    return np.stack(out)
