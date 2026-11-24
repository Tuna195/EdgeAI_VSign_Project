import json
import numpy as np
from src.core.paths import RUNS_DIR
from src.training.experiments import base_config
from src.training.train import report, run_name

cfg = base_config("E6")                          # cấu hình cuối = giá trị thắng của E1-E5, đủ 28 từ
print("Cấu hình cuối:", run_name(cfg))
for seed in (0, 1, 2):
    report(cfg, seed, "test")
top1 = [json.load(open(RUNS_DIR / run_name(cfg) / f"seed{s}" / "test.json"))["top1"] for s in (0, 1, 2)]
print(f"TEST top-1: {np.mean(top1):.3f} ± {np.std(top1):.3f}")
