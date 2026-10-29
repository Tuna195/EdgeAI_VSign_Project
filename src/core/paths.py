"""Đường dẫn dùng chung. Mọi file khác lấy đường dẫn ở đây thay vì tự tính Path(__file__).parent.parent."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # src/core/paths.py -> thư mục gốc repo
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"              # data/videos/<train|test>/<từ>/<id>.mp4
RAW_DIR = DATA_DIR / "raw"                   # data/raw/<train|test>/<từ>/<id>.npy  (T, 225)
REPORT_DIR = DATA_DIR / "reports"
PROCESSED_DIR = DATA_DIR / "processed"
EXCLUDE_FILE = DATA_DIR / "exclude.txt"      # video loại sau khi duyệt bằng mắt (review_tool tạo)
WORDS_FILE = ROOT / "words.txt"
MODELS_DIR = ROOT / "models"                 # file .task của MediaPipe
RUNS_DIR = ROOT / "runs"                     # kết quả train
INDEX_FILE = PROCESSED_DIR / "index.csv"           # mỗi video 1 dòng: từ, nhãn, split... (dataset/index_kaggle.py)
VOCAB_FILE = PROCESSED_DIR / "vocab.csv"           # nhãn, từ
SEQ_FILE = PROCESSED_DIR / "sequences.npz"         # chuỗi đã tiền xử lý (dataset/build_sequences.py)
