"""Phase 14: build a self-contained teacher package for Project 2.

Copies everything needed to load the teacher and reproduce its test perplexity into one folder:
weights (no optimizer state), tokenizer, configs, the model code, validation/test tokens, prompts,
results, environment and a SHA-256 manifest. teacher_files/verify_teacher.py checks it.

Usage: python scripts/package_teacher.py [--run small] [--out teacher]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src, dst_dir, name=None):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name or os.path.basename(src))
    shutil.copy2(os.path.join(ROOT, src), dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="small")
    ap.add_argument("--out", default="teacher")
    args = ap.parse_args()

    out = os.path.join(ROOT, args.out)
    if os.path.exists(out):
        shutil.rmtree(out)  # always rebuilt from the project files, never edited by hand
    os.makedirs(out)
    cfg = load_config(os.path.join(ROOT, "configs", f"{args.run}.yaml"))
    data_dir = cfg["data"]["data_dir"]

    # 1. weights only: the optimizer state is only for resuming training (it triples the file size)
    ckpt = torch.load(os.path.join(ROOT, "checkpoints", args.run, "model_best.pt"), map_location="cpu",
                      weights_only=False)
    teacher = {
        "model": ckpt["model"],
        "model_config": ckpt["model_config"],
        "config": ckpt["config"],
        "step": ckpt["step"],
        "best_val_loss": ckpt["best_val_loss"],
        "tokenizer": "tokenizer.json",
        "source_checkpoint": f"checkpoints/{args.run}/model_best.pt",
    }
    torch.save(teacher, os.path.join(out, "teacher.pt"))
    with open(os.path.join(out, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(ckpt["model_config"], f, indent=2)

    # 2. tokenizer, configs, prompts, environment
    copy(cfg["data"]["tokenizer"], out)
    copy(f"configs/{args.run}.yaml", os.path.join(out, "configs"))
    copy("configs/prompts.txt", os.path.join(out, "configs"))
    copy("requirements.lock.txt", out)
    copy("requirements.txt", out)

    # 3. model + evaluation code (the model is plain PyTorch, so its class is needed to load it)
    for name in sorted(os.listdir(os.path.join(ROOT, "src"))):
        if name.endswith(".py"):
            copy(f"src/{name}", os.path.join(out, "src"))
    for name in ["verify_teacher.py", "load_teacher.py"]:
        copy(f"scripts/teacher_files/{name}", out)
    copy("scripts/teacher_files/README.md", out)

    # 4. data needed to reproduce perplexity (train.bin is 745 MB and not needed for that)
    for name in ["val.bin", "test.bin", "stats.json"]:
        copy(f"{data_dir}/{name}", os.path.join(out, "data"))

    # 5. results
    m = f"results/metrics/{args.run}"
    for name in ["eval.json", "eval_report.md", "run_info.json", "train_summary.json", "log.csv",
                 "madeup_words.json"]:
        copy(f"{m}/{name}", os.path.join(out, "results"))
    copy("results/metrics/env_report.json", os.path.join(out, "results"))
    copy("results/metrics/benchmark.csv", os.path.join(out, "results"))
    copy(f"results/generations/{args.run}/error_analysis.csv", os.path.join(out, "results"))
    copy(f"results/generations/{args.run}/samples.jsonl", os.path.join(out, "results"))
    copy("results/figures/loss_baseline_small.png", os.path.join(out, "results"))

    # 6. expected scores + manifest
    with open(os.path.join(ROOT, m, "eval.json"), encoding="utf-8") as f:
        ev = json.load(f)
    expected = {
        "test_loss": ev["test_loss"], "test_perplexity": ev["test_perplexity"],
        "val_loss": ev["val_loss"], "val_perplexity": ev["val_perplexity"],
        "params": ev["params"]["total"], "precision": ev["precision"], "eval_batch_size": 16,
        "recommended_decoding": {"method": "top_p", "top_p": 0.95, "temperature": 1.0},
    }
    with open(os.path.join(out, "expected.json"), "w", encoding="utf-8") as f:
        json.dump(expected, f, indent=2)

    lines = []
    for dirpath, _, files in os.walk(out):
        for name in sorted(files):
            p = os.path.join(dirpath, name)
            rel = os.path.relpath(p, out).replace(os.sep, "/")
            if rel != "MANIFEST.sha256":
                lines.append(f"{sha256(p)}  {rel}")
    with open(os.path.join(out, "MANIFEST.sha256"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(lines, key=lambda s: s.split("  ")[1])) + "\n")

    total = sum(os.path.getsize(os.path.join(d, n)) for d, _, fs in os.walk(out) for n in fs)
    print(f"wrote {out}: {len(lines) + 1} files, {total / 2**20:.1f} MB")
    print(f"teacher.pt {os.path.getsize(os.path.join(out, 'teacher.pt')) / 2**20:.1f} MB "
          f"(training checkpoint was {os.path.getsize(os.path.join(ROOT, 'checkpoints', args.run, 'model_best.pt')) / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
