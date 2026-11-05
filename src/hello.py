from pathlib import Path

raw = list(Path('data/raw').glob('*/*/*.npy'))
print('Xin chào từ GitHub! Đếm được', len(raw), 'file .npy')