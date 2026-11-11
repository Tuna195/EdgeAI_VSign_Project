"""Mọi đường dẫn của dự án"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]     # src/core/paths.py -> lùi lên 3 lần = thư mục gốc repo
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                     # raw/<train|test>/<từ>/<id>.npy : landmark (T, 225)
VIDEO_DIR = DATA_DIR / "videos"                # videos/<train|test>/<từ>/<id>.mp4
EXCLUDE_FILE = DATA_DIR / "exclude.txt"        # video bị loại khi duyệt bằng mắt
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_FILE = PROCESSED_DIR / "index.csv"       
VOCAB_FILE = PROCESSED_DIR / "vocab.csv"       
SEQ_FILE = PROCESSED_DIR / "sequences.npz"     
WORDS_FILE = ROOT / "words.txt"