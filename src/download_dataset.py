"""Tải video của các từ đã chọn từ bộ Kaggle VSL v2 (chỉ thư mục frame_splited, không tải cả 82 GB).

Chạy: python src/download_dataset.py                 (đọc danh sách từ trong words.txt)
      python src/download_dataset.py "Con bò" "Chợ"  (tải thêm vài từ, không cần sửa words.txt)

Kết quả: data/videos/train/<từ>/*.mp4 và data/videos/test/<từ>/*.mp4 (file đã có thì giữ nguyên).
Cần token Kaggle trong C:\\Users\\<tên>\\.kaggle\\access_token.

Vì sao không tải thẳng từng file: Kaggle giới hạn số lần gọi API (lỗi 429 Too Many Requests),
mà mỗi từ có ~60 video. Nên script đẩy 1 notebook riêng tư (private) lên tài khoản Kaggle của bạn,
notebook gom video các từ cần thành 1 file zip ngay trên server Kaggle, rồi laptop tải đúng 1 file đó.
"""
import json
import sys
import tempfile
import time
import unicodedata
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "nguyenanfms/vsl-vietnamese-sign-language-v2"
KERNEL_SLUG = "vsign-download-words"   # tên notebook trên Kaggle, chạy lại sẽ tạo phiên bản mới
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "videos"
WORDS_FILE = ROOT / "words.txt"

# Code chạy TRÊN Kaggle: dataset được gắn sẵn ở /kaggle/input, không phải tải
KERNEL_CODE = '''
import json, os, unicodedata, zipfile
nfc = lambda s: unicodedata.normalize("NFC", s)
WORDS = set(nfc(w) for w in json.loads(%r))
# Tìm thư mục processed/processed/frame_splited. Không dùng glob("**"): nó duyệt cả 191k file trên
# ổ mạng của Kaggle, rất chậm. os.walk + bỏ qua các nhánh không cần thì chỉ đi vài thư mục.
SKIP = {"raw", "processed_augmented", "keypoints_splited", "train", "test"}
root = None
for cur, dirs, _ in os.walk("/kaggle/input"):
    if os.path.basename(cur) == "frame_splited":
        root = cur
        break
    dirs[:] = [d for d in dirs if d not in SKIP]
print("root:", root)
found, n = set(), 0
with zipfile.ZipFile("/kaggle/working/videos.zip", "w", zipfile.ZIP_STORED) as z:
    for split in ("train", "test"):
        for d in os.listdir(os.path.join(root, split)):
            if nfc(d) in WORDS:
                found.add(nfc(d))
                for f in os.listdir(os.path.join(root, split, d)):
                    z.write(os.path.join(root, split, d, f), f"{split}/{nfc(d)}/{f}")
                    n += 1
json.dump(sorted(found), open("/kaggle/working/found.json", "w", encoding="utf-8"), ensure_ascii=False)
print("videos:", n, "words:", len(found))
'''


def nfc(text):
    """Chuẩn hoá Unicode: cùng chữ "ạ" có thể gõ thành 1 hoặc 2 ký tự, đưa về 1 dạng để so sánh."""
    return unicodedata.normalize("NFC", text.strip())


def read_words():
    """Từ cần tải: lấy từ tham số dòng lệnh, nếu không có thì đọc words.txt."""
    if len(sys.argv) > 1:
        return [nfc(w) for w in sys.argv[1:]]
    lines = WORDS_FILE.read_text(encoding="utf-8").splitlines()
    return [nfc(line) for line in lines if line.strip() and not line.startswith("#")]


def main():
    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username")
    kernel = f"{user}/{KERNEL_SLUG}"
    words = read_words()
    print(f"{len(words)} từ cần tải")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 1. Đẩy notebook lên Kaggle (private, không cần internet, không cần GPU)
        (tmp / "main.py").write_text(KERNEL_CODE % json.dumps(words, ensure_ascii=False), encoding="utf-8")
        (tmp / "kernel-metadata.json").write_text(json.dumps({
            "id": kernel, "title": KERNEL_SLUG, "code_file": "main.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": False, "enable_internet": False,
            "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [],
        }), encoding="utf-8")
        api.kernels_push(str(tmp))
        print("Đã đẩy notebook", kernel, "- đang chờ Kaggle chạy (thường 1-3 phút)...")

        # 2. Chờ notebook chạy xong
        time.sleep(10)
        while True:
            status = str(api.kernels_status(kernel).status).lower()
            if "complete" in status:
                break
            if "error" in status or "cancel" in status:
                raise RuntimeError(f"Notebook lỗi ({status}), xem log tại https://www.kaggle.com/code/{kernel}")
            time.sleep(15)

        # 3. Tải file zip kết quả về và giải nén vào data/videos
        api.kernels_output(kernel, str(tmp))
        found = json.loads((tmp / "found.json").read_text(encoding="utf-8"))
        added = 0
        with zipfile.ZipFile(tmp / "videos.zip") as z:
            for name in z.namelist():
                target = OUT_DIR / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(name))
                    added += 1

    missing = [w for w in words if w not in found]
    if missing:
        print("Không tìm thấy thư mục trên Kaggle (kiểm tra lại chính tả):", ", ".join(missing))
    print(f"Xong: thêm {added} video mới vào {OUT_DIR}")


if __name__ == "__main__":
    main()
