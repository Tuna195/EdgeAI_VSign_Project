# EdgeAI VSign

Real-time Vietnamese Sign Language recognition (isolated signs, VSL400 front view) on Raspberry Pi 4,
using MediaPipe landmarks and lightweight sequence models (MLP, GRU, BiGRU, Conv1D).

## Cấu trúc

| Thư mục | Nội dung | Nhánh phụ trách |
|---|---|---|
| `src/core/` | Tiền xử lý và đặc trưng dùng chung cho train và Pi (`features.py`), đường dẫn, cấu hình | M |
| `src/data/` | Gộp metadata, trích landmark, chia dữ liệu theo người ký | M |
| `src/train/` | Dataset, model, train, đánh giá, augmentation | M |
| `src/pi/` | Camera, extractor MediaPipe, tách cử chỉ, chạy trực tiếp, benchmark | P |
| `configs/` | `extractor_config.json` (dùng chung Kaggle và Pi), cấu hình thí nghiệm | P (extractor), M (thí nghiệm) |
| `data/` | Dữ liệu cục bộ, không đưa vào git | |
| `runs/` | Kết quả thí nghiệm, không đưa vào git | |
