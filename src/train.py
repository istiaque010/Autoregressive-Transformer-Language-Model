"""Training loop (Phase 8).

Usage:
    python -m src.train --config configs/smoke.yaml
    python -m src.train --config configs/baseline.yaml --resume checkpoints/baseline/model_last.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from .dataset import NextTokenDataset, load_tokens
from .model import GPT, GPTConfig, count_parameters
from .tokenizer import StoryTokenizer
from .utils import autocast_ctx, env_info, load_config, memory_mb, peak_memory_mb, resolve_device, set_seed


def lr_at(step: int, tc: dict) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    progress = (step - tc["warmup_steps"]) / max(1, tc["max_steps"] - tc["warmup_steps"])
    progress = min(progress, 1.0)
    return tc["min_lr"] + 0.5 * (1 + math.cos(math.pi * progress)) * (tc["lr"] - tc["min_lr"])


def make_optimizer(model: GPT, tc: dict) -> torch.optim.AdamW:
    # Weight decay on matrices only; not on biases, LayerNorm or (1-D) params.
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay = [p for p in params.values() if p.dim() >= 2]
    no_decay = [p for p in params.values() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": tc["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tc["lr"], betas=tuple(tc["betas"]))


@torch.no_grad()
def evaluate_loss(model: GPT, loader: DataLoader, device: str, precision: str, max_batches: int | None = None) -> float:
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx(device, precision):
            _, loss = model(x, y)
        total += loss.item() * x.size(0)
        count += x.size(0)
    model.train()
    return total / max(1, count)


def infinite(loader: DataLoader, seed: int):
    epoch = 0
    while True:
        loader.generator.manual_seed(seed + epoch)
        for batch in loader:
            yield batch
        epoch += 1


def save_checkpoint(path: str, model, optimizer, step: int, best_val: float, cfg: dict, model_cfg: GPTConfig) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val,
            "model_config": model_cfg.to_dict(),
            "config": cfg,
            "tokenizer": cfg["data"]["tokenizer"],
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--max_steps", type=int, default=None, help="override train.max_steps")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc, dc = cfg["train"], cfg["data"]
    if args.max_steps is not None:
        tc["max_steps"] = args.max_steps
    run = cfg["run_name"]
    set_seed(cfg["seed"])
    device = resolve_device(tc["device"])
    precision = tc["precision"]

    ckpt_dir = os.path.join("checkpoints", run)
    metrics_dir = os.path.join("results", "metrics", run)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    tok = StoryTokenizer(dc["tokenizer"])
    model_cfg = GPTConfig(vocab_size=tok.vocab_size, **cfg["model"])
    model = GPT(model_cfg).to(device)
    params = count_parameters(model)
    print(f"device={device} precision={precision} params={params}")

    train_ds = NextTokenDataset(load_tokens(dc["data_dir"], "train"), model_cfg.block_size)
    val_ds = NextTokenDataset(load_tokens(dc["data_dir"], "val"), model_cfg.block_size)
    g = torch.Generator()
    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                              num_workers=tc["num_workers"], generator=g)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False, num_workers=tc["num_workers"])
    print(f"train windows={len(train_ds):,} val windows={len(val_ds):,}")

    optimizer = make_optimizer(model, tc)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and device.startswith("cuda")))

    start_step, best_val = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"], ckpt["best_val_loss"]
        print(f"resumed from {args.resume} at step {start_step}")

    run_info = {"config": cfg, "model_config": model_cfg.to_dict(), "params": params, "env": env_info(device),
                "effective_batch_seqs": tc["batch_size"] * tc["grad_accum_steps"],
                "effective_batch_tokens": tc["batch_size"] * tc["grad_accum_steps"] * model_cfg.block_size}
    with open(os.path.join(metrics_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    log_path = os.path.join(metrics_dir, "log.csv")
    new_log = not (args.resume and os.path.exists(log_path))
    log_f = open(log_path, "w" if new_log else "a", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "train_loss", "val_loss", "val_ppl", "lr", "tokens_per_sec", "mem_mb", "elapsed_s"])

    batches = infinite(train_loader, cfg["seed"] + start_step)
    tokens_per_step = tc["batch_size"] * tc["grad_accum_steps"] * model_cfg.block_size
    model.train()
    t0 = time.time()
    t_last = t0
    running, since_log = 0.0, 0

    for step in range(start_step, tc["max_steps"]):
        lr = lr_at(step, tc)
        for group in optimizer.param_groups:
            group["lr"] = lr

        step_loss = 0.0
        for _ in range(tc["grad_accum_steps"]):
            x, y = next(batches)
            x, y = x.to(device), y.to(device)
            with autocast_ctx(device, precision):
                _, loss = model(x, y)
                loss = loss / tc["grad_accum_steps"]
            scaler.scale(loss).backward()
            step_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        running += step_loss
        since_log += 1

        done = step + 1
        is_eval = done % tc["eval_interval"] == 0 or done == tc["max_steps"]
        if done % tc["log_interval"] == 0 or is_eval:
            now = time.time()
            tps = tokens_per_step * since_log / max(1e-9, now - t_last)
            t_last = now
            train_loss = running / since_log
            running, since_log = 0.0, 0
            val_loss = val_ppl = ""
            if is_eval:
                val_loss = evaluate_loss(model, val_loader, device, precision, tc["eval_batches"])
                val_ppl = math.exp(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(os.path.join(ckpt_dir, "model_best.pt"), model, optimizer, done, best_val, cfg, model_cfg)
                t_last = time.time()  # don't count eval time in throughput
            mem = memory_mb(device)
            log.writerow([done, f"{train_loss:.4f}", f"{val_loss:.4f}" if val_loss != "" else "",
                          f"{val_ppl:.2f}" if val_ppl != "" else "", f"{lr:.2e}", f"{tps:.0f}", f"{mem:.0f}",
                          f"{now - t0:.1f}"])
            log_f.flush()
            msg = f"step {done:6d} | loss {train_loss:.4f} | lr {lr:.2e} | {tps:,.0f} tok/s | mem {mem:,.0f} MB"
            if is_eval:
                msg += f" | val {val_loss:.4f} ppl {val_ppl:.2f}"
            print(msg, flush=True)

        if done % tc["ckpt_interval"] == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"model_step_{done}.pt"), model, optimizer, done, best_val, cfg, model_cfg)
        if done % tc["eval_interval"] == 0 or done == tc["max_steps"]:
            save_checkpoint(os.path.join(ckpt_dir, "model_last.pt"), model, optimizer, done, best_val, cfg, model_cfg)

    log_f.close()
    elapsed = time.time() - t0
    summary = {"best_val_loss": best_val, "best_val_ppl": math.exp(best_val), "steps": tc["max_steps"],
               "wall_clock_s": round(elapsed, 1), "peak_mem_mb": round(peak_memory_mb(device), 1)}
    with open(os.path.join(metrics_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"done in {elapsed / 60:.1f} min | best val loss {best_val:.4f} (ppl {math.exp(best_val):.2f})")


if __name__ == "__main__":
    main()
