"""Train 1 cấu hình với 1 hoặc nhiều seed, ghi kết quả. Chỉ dùng val để chọn; test chỉ dùng khi gọi --report test.

Chạy (từ thư mục gốc repo):
  python -m src.training.train                                  cấu hình mặc định (GRU, theo vai, có z, 30 frame), seed 0
  python -m src.training.train --model mlp                      đổi 1 yếu tố
  python -m src.training.train --norm none --seeds 0 1 2        chạy 3 seed
  python -m src.training.train --report val                     ma trận nhầm lẫn trên val của model đã train (Phần 7)
  python -m src.training.train --norm shoulder_wrist --seeds 0 1 2 --report test     CHỈ cho cấu hình cuối (Phần 8a)

Mỗi lần chạy: runs/<tên cấu hình>/seed<k>/ gồm config.json, vocab.csv, best.pt, log.csv, metrics.json,
và thêm 1 dòng vào runs/results.csv. Thư mục đã chạy xong thì BỎ QUA (chạy tiếp được khi bị ngắt), trừ khi --overwrite.
Dừng sớm: val top-1 không tiến bộ 20 epoch liền (bằng nhau thì so val loss); giữ model tốt nhất theo val.
"""
import argparse
import csv
import json
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn

from src.core.features import NORMS, feature_dim
from src.core.paths import ROOT, RUNS_DIR
from src.training.data import build_features, load_split_data
from src.training.models import MODELS, build_model, count_params

# Các yếu tố được so sánh trong thí nghiệm (tạo nên tên cấu hình) + tham số train cố định cho mọi lần chạy
DEFAULTS = {"model": "gru", "norm": "shoulder", "use_z": True, "n_frames": 30, "augment": True, "n_words": 0,
            "epochs": 200, "patience": 20, "batch": 32, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.3}
RESULTS_FILE = RUNS_DIR / "results.csv"
RESULT_FIELDS = ["finished", "name", "seed", "model", "norm", "use_z", "n_frames", "augment", "n_words",
                 "val_top1", "val_top3", "val_macro", "val_loss", "train_top1", "best_epoch", "epochs_run",
                 "params", "train_sec", "infer_ms", "device", "git"]


def run_name(cfg):
    """Tên thư mục cho 1 cấu hình, vd. gru_shoulder_z_f30_aug_wall (wall = tất cả các từ)."""
    return (f'{cfg["model"]}_{cfg["norm"]}_{"z" if cfg["use_z"] else "noz"}_f{cfg["n_frames"]}_'
            f'{"aug" if cfg["augment"] else "noaug"}_w{cfg["n_words"] or "all"}')


def git_version():
    """Mã commit git hiện tại, để biết kết quả này được tạo bằng phiên bản code nào."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def set_seed(seed):
    """Cố định mọi nguồn ngẫu nhiên -> chạy lại cùng seed ra cùng kết quả (trên cùng máy, cùng phiên bản thư viện)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, X, y, n_classes, batch=256):
    """-> dict loss, top1, top3, macro (trung bình độ chính xác từng từ), pred (từ đoán hạng 1 của mỗi mẫu)."""
    model.eval()
    logits = torch.cat([model(X[i:i + batch]) for i in range(0, len(X), batch)])
    loss = nn.functional.cross_entropy(logits, y).item()
    top3 = logits.topk(min(3, n_classes), dim=1).indices          # 3 từ có logit cao nhất của mỗi mẫu
    pred = top3[:, 0]
    correct = (pred == y).float()
    per_class = [correct[y == c].mean().item() for c in range(n_classes) if (y == c).any()]
    return {"loss": loss, "top1": correct.mean().item(), "top3": (top3 == y[:, None]).any(dim=1).float().mean().item(),
            "macro": float(np.mean(per_class)), "pred": pred.cpu().numpy()}


def infer_ms(model, X, n=200):
    """Thời gian đoán 1 mẫu (batch = 1) trên CPU, mili giây, trung bình n mẫu."""
    model = model.cpu().eval()
    X = X[:n].cpu()
    with torch.no_grad():
        for i in range(min(10, len(X))):     # chạy "làm nóng" vài lần trước khi đo
            model(X[i:i + 1])
        start = time.perf_counter()
        for i in range(len(X)):
            model(X[i:i + 1])
    return (time.perf_counter() - start) * 1000 / len(X)


def to_tensors(X, y, device):
    return torch.from_numpy(X).to(device), torch.from_numpy(y).long().to(device)


def run(cfg, seed, device="cpu", overwrite=False, verbose=True):
    """Train 1 cấu hình với 1 seed. Trả về dict kết quả (đọc lại từ metrics.json nếu đã chạy xong)."""
    name = run_name(cfg)
    out = RUNS_DIR / name / f"seed{seed}"
    if (out / "metrics.json").exists() and not overwrite:
        if verbose:
            print(f"[bỏ qua] {name} seed {seed}: đã chạy xong")
        return json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)

    # ---- chuẩn bị ----
    set_seed(seed)
    words, data = load_split_data(cfg["n_words"])
    n_classes, dim = len(words), feature_dim(cfg)
    train_seqs, y_train = data["train"]
    X_val, y_val = to_tensors(build_features(data["val"][0], cfg), data["val"][1], device)   # val: tính 1 lần
    model = build_model(cfg["model"], cfg["n_frames"], dim, n_classes, cfg["dropout"]).to(device)
    params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(seed)               # augmentation + đổi tốc độ ký
    gen = torch.Generator().manual_seed(seed)       # thứ tự video trong mỗi epoch

    (out / "config.json").write_text(json.dumps({
        **cfg, "name": name, "seed": seed, "n_classes": n_classes, "dim": dim, "params": params,
        "git": git_version(), "torch": torch.__version__, "device": device,
        "n_train": len(y_train), "n_val": len(y_val)}, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "vocab.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([[i, w] for i, w in enumerate(words)])
    if verbose:
        print(f"== {name} seed {seed}: {len(y_train)} train / {len(y_val)} val, {n_classes} từ, "
              f"D = {dim}, {params:,} tham số")

    # ---- vòng lặp train ----
    log, best, best_epoch, start = [], None, 0, time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        X_tr, y_tr = to_tensors(build_features(train_seqs, cfg, rng, cfg["augment"]), y_train, device)  # mỗi epoch 1 biến đổi mới
        model.train()
        total_loss = total_correct = 0.0
        for idx in torch.randperm(len(y_tr), generator=gen).split(cfg["batch"]):
            logits = model(X_tr[idx])
            loss = nn.functional.cross_entropy(logits, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
            total_correct += (logits.argmax(dim=1) == y_tr[idx]).sum().item()
        v = evaluate(model, X_val, y_val, n_classes)
        log.append({"epoch": epoch, "train_loss": total_loss / len(y_tr), "train_top1": total_correct / len(y_tr),
                    "val_loss": v["loss"], "val_top1": v["top1"], "val_top3": v["top3"], "val_macro": v["macro"],
                    "sec": round(time.time() - start, 1)})
        if best is None or (v["top1"], -v["loss"]) > (best["top1"], -best["loss"]):    # tốt hơn mọi epoch trước
            best, best_epoch = v, epoch
            torch.save(model.state_dict(), out / "best.pt")
        if verbose and (epoch % 10 == 0 or epoch == 1):
            r = log[-1]
            print(f"  epoch {epoch:3d}  train loss {r['train_loss']:.3f} acc {r['train_top1']:.3f} | "
                  f"val loss {r['val_loss']:.3f} top1 {r['val_top1']:.3f} top3 {r['val_top3']:.3f}  "
                  f"(tốt nhất {best['top1']:.3f} @ {best_epoch})")
        if epoch - best_epoch >= cfg["patience"]:                                      # dừng sớm
            break
    train_sec = time.time() - start

    with open(out / "log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in log)

    # ---- sau vòng lặp: model tốt nhất, chấm train KHÔNG augmentation (so với val -> biết học thuộc nhiều hay ít) ----
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    X_tr_clean, y_tr_clean = to_tensors(build_features(train_seqs, cfg), y_train, device)
    tr = evaluate(model, X_tr_clean, y_tr_clean, n_classes)
    result = {"finished": time.strftime("%Y-%m-%d %H:%M"), "name": name, "seed": seed,
              **{k: cfg[k] for k in ("model", "norm", "use_z", "n_frames", "augment", "n_words")},
              "val_top1": round(best["top1"], 4), "val_top3": round(best["top3"], 4),
              "val_macro": round(best["macro"], 4), "val_loss": round(best["loss"], 4),
              "train_top1": round(tr["top1"], 4), "best_epoch": best_epoch, "epochs_run": len(log),
              "params": params, "train_sec": round(train_sec, 1), "infer_ms": round(infer_ms(model, X_val), 3),
              "device": device, "git": git_version()}
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    new_file = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(result)
    if verbose:
        print(f"  -> val top1 {result['val_top1']:.3f} top3 {result['val_top3']:.3f} macro {result['val_macro']:.3f} "
              f"| train {result['train_top1']:.3f} | epoch tốt nhất {best_epoch}/{len(log)} | {train_sec:.0f}s")
    return result


def report(cfg, seed, split, device="cpu"):
    """Chấm model đã train (best.pt) trên val hoặc test: in độ chính xác, cặp từ hay nhầm, lưu ma trận nhầm lẫn.
    split = "test" CHỈ dùng cho cấu hình cuối cùng, sau khi đã chọn xong mọi thứ bằng val."""
    name = run_name(cfg)
    out = RUNS_DIR / name / f"seed{seed}"
    if not (out / "best.pt").exists():
        raise SystemExit(f"Chưa train {name} seed {seed}")
    words, data = load_split_data(cfg["n_words"])
    X, y = to_tensors(build_features(data[split][0], cfg), data[split][1], device)
    model = build_model(cfg["model"], cfg["n_frames"], feature_dim(cfg), len(words), cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    r = evaluate(model, X, y, len(words))
    y_np = y.cpu().numpy()
    conf = np.zeros((len(words), len(words)), int)      # hàng = từ đúng, cột = từ model đoán
    np.add.at(conf, (y_np, r["pred"]), 1)               # mỗi mẫu cộng 1 vào ô (từ đúng, từ đoán)
    print(f"{split.upper()} {name} seed {seed}: {len(y_np)} mẫu, top1 {r['top1']:.3f} top3 {r['top3']:.3f} "
          f"macro {r['macro']:.3f}")
    pairs = sorted(((conf[i, j], words[i], words[j]) for i in range(len(words)) for j in range(len(words))
                    if i != j and conf[i, j]), reverse=True)[:10]
    print("Các cặp hay nhầm nhất (từ đúng -> model đoán):")
    for n, a, b in pairs:
        print(f"  {n:2d}  {a} -> {b}")
    print("Độ chính xác từng từ:")
    for i, w in enumerate(words):
        print(f"  {w:16s} {conf[i, i]:2d}/{conf[i].sum():2d}")
    with open(out / f"{split}_confusion.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([[""] + words] + [[w] + conf[i].tolist() for i, w in enumerate(words)])
    (out / f"{split}.json").write_text(json.dumps({k: r[k] for k in ("loss", "top1", "top3", "macro")}, indent=2),
                                       encoding="utf-8")


def parse_cfg(args):
    cfg = dict(DEFAULTS)
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--norm", choices=NORMS)
    p.add_argument("--z", dest="use_z", action=argparse.BooleanOptionalAction, help="--z / --no-z")
    p.add_argument("--frames", dest="n_frames", type=int)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, help="--augment / --no-augment")
    p.add_argument("--words", dest="n_words", type=int, help="chỉ dùng k từ (0 = tất cả)")
    for k in ("epochs", "patience", "batch"):
        p.add_argument(f"--{k}", type=int)
    for k in ("lr", "weight_decay", "dropout"):
        p.add_argument(f"--{k}", type=float)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default="cpu", help="cpu (mặc định, kết quả lặp lại được) hoặc cuda")
    p.add_argument("--threads", type=int, default=4, help="số luồng CPU cho torch")
    p.add_argument("--overwrite", action="store_true", help="train lại dù đã có kết quả")
    p.add_argument("--report", choices=("val", "test"), help="không train: chấm model đã train, in ma trận nhầm lẫn")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    cfg = parse_cfg(args)
    for seed in args.seeds:
        if args.report:
            report(cfg, seed, args.report, args.device)
        else:
            run(cfg, seed, args.device, args.overwrite)


if __name__ == "__main__":
    main()
