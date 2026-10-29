"""Đóng gói những gì Colab cần để train vào 1 file zip: code (src/*.py) + index.csv + vocab.csv + sequences.npz.

Chạy: python -m src.training.pack_colab   -> vsign_colab.zip ở thư mục gốc repo (~70 MB)

Không cần video, data/raw hay mediapipe: Colab chỉ train trên sequences.npz đã tiền xử lý.
Sửa code hoặc chạy lại build_sequences thì đóng gói lại và upload đè file cũ trên Drive.
"""
import zipfile

from src.core.paths import INDEX_FILE, ROOT, SEQ_FILE, VOCAB_FILE

OUT_FILE = ROOT / "vsign_colab.zip"


def main():
    files = sorted((ROOT / "src").rglob("*.py")) + [INDEX_FILE, VOCAB_FILE, SEQ_FILE, ROOT / "requirements.txt"]
    with zipfile.ZipFile(OUT_FILE, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    print(f"{len(files)} file -> {OUT_FILE} ({OUT_FILE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
