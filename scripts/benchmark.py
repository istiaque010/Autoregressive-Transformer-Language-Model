"""Phase 7: measure training speed and memory for a config, and estimate full training time.

Runs a few real training steps (forward + backward + AdamW) on real data with the config's batch size,
skips the first warm-up steps, then reports tokens/s, memory, and the estimated time for the config's
max_steps and for one pass over train.bin.

Usage: python scripts/benchmark.py --config configs/small.yaml [--steps 30]
Appends one row per run to results/metrics/benchmark.csv
"""
import argparse
import csv
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import NextTokenDataset, load_tokens  # noqa: E402
from src.model import GPT, GPTConfig, count_parameters  # noqa: E402
from src.tokenizer import StoryTokenizer  # noqa: E402
from src.train import make_optimizer  # noqa: E402
from src.utils import autocast_ctx, load_config, peak_memory_mb, resolve_device, set_seed  # noqa: E402


def hms(seconds: float) -> str:
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h {m:02d}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=30, help="measured micro-batches")
    ap.add_argument("--warmup", type=int, default=5, help="micro-batches ignored at the start")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc, dc = cfg["train"], cfg["data"]
    set_seed(cfg["seed"])
    device = resolve_device(tc["device"])
    tok = StoryTokenizer(dc["tokenizer"])
    model_cfg = GPTConfig(vocab_size=tok.vocab_size, **cfg["model"])
    model = GPT(model_cfg).to(device).train()
    optimizer = make_optimizer(model, tc)

    train_tokens = load_tokens(dc["data_dir"], "train")
    ds = NextTokenDataset(train_tokens, model_cfg.block_size)
    batches = iter(DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True))
    tokens_per_micro = tc["batch_size"] * model_cfg.block_size

    for i in range(args.warmup + args.steps):
        if i == args.warmup:
            t0 = time.perf_counter()
        x, y = next(batches)
        with autocast_ctx(device, tc["precision"]):
            _, loss = model(x.to(device), y.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    elapsed = time.perf_counter() - t0

    tps = args.steps * tokens_per_micro / elapsed
    tokens_per_step = tokens_per_micro * tc["grad_accum_steps"]
    run_s = tc["max_steps"] * tokens_per_step / tps
    epoch_s = len(train_tokens) / tps
    row = {
        "config": cfg["run_name"],
        "params": count_parameters(model)["total"],
        "device": device,
        "threads": torch.get_num_threads(),
        "precision": tc["precision"],
        "batch": tc["batch_size"],
        "grad_accum": tc["grad_accum_steps"],
        "block_size": model_cfg.block_size,
        "tokens_per_s": round(tps),
        "ms_per_micro_batch": round(1000 * elapsed / args.steps, 1),
        "mem_mb": round(peak_memory_mb(device)),
        "max_steps": tc["max_steps"],
        "tokens_per_run": tc["max_steps"] * tokens_per_step,
        "est_run_time": hms(run_s),
        "train_tokens": len(train_tokens),
        "est_epoch_time": hms(epoch_s),
    }
    for k, v in row.items():
        print(f"{k:>20}: {v:,}" if isinstance(v, int) else f"{k:>20}: {v}")

    os.makedirs("results/metrics", exist_ok=True)
    path = "results/metrics/benchmark.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
