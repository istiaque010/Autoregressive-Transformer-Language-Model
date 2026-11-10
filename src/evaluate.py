"""Evaluation (Phases 9, 11, 12, 13): perplexity, sampling experiment, performance, error analysis.

Usage:
    python -m src.evaluate --ckpt checkpoints/small/model_best.pt
    python -m src.evaluate --ckpt checkpoints/small/model_best.pt --perf_only   # re-measure speed on an idle PC
Outputs (run name taken from the checkpoint):
    results/metrics/<run>/eval.json, eval_report.md
    results/generations/<run>/samples.jsonl, error_analysis.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import time

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from .dataset import NextTokenDataset, load_stories, load_tokens
from .generate import generate, generate_text
from .model import count_parameters, load_checkpoint
from .tokenizer import StoryTokenizer
from .train import evaluate_loss
from .utils import env_info, peak_memory_mb, resolve_device, set_seed

SAMPLING_SETTINGS = [
    {"name": "greedy", "method": "greedy"},
    {"name": "temp_0.7", "method": "temperature", "temperature": 0.7},
    {"name": "temp_1.0", "method": "temperature", "temperature": 1.0},
    {"name": "top_k_20", "method": "top_k", "top_k": 20},
    {"name": "top_k_50", "method": "top_k", "top_k": 50},
    {"name": "top_p_0.90", "method": "top_p", "top_p": 0.90},
    {"name": "top_p_0.95", "method": "top_p", "top_p": 0.95},
]
STORY_ENDINGS = (".", "!", "?", '"')
DISTINCT_MAX_WORDS = 100  # Distinct-n falls as text gets longer, so every story is compared on the same length
REP_FLAG_PERCENTILE = 95  # a generated story is "repetitive" if it repeats more than 95% of real stories do
CPU_BUSY_LIMIT = 15.0  # CPU % used by other programs above which speed measurements are marked as unreliable


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+|[.,!?;]", text.lower())


def ngrams(seq: list, n: int) -> list:
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def distinct_n(texts: list[str], n: int, max_words: int | None = None) -> float:
    """Unique n-grams / total n-grams across all texts (corpus-level Distinct-n)."""
    grams = [g for t in texts for g in ngrams(words(t)[:max_words], n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def grouped_distinct(groups: list[list[str]], n: int) -> tuple[float, float]:
    """Mean and standard deviation of Distinct-n over equal-sized groups of stories.

    Distinct-n depends on how many stories are pooled, so settings are only comparable on equal group sizes.
    """
    values = [distinct_n(g, n, DISTINCT_MAX_WORDS) for g in groups]
    return statistics.mean(values), statistics.pstdev(values)


def repeated_ngram_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams in a text that already appeared earlier in the same text."""
    grams = ngrams(words(text), n)
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def real_story_reference(stories: list[str], group_size: int, n_groups: int, seed: int) -> dict:
    """The same metrics on real validation stories: the yardstick for generated text."""
    rep = [repeated_ngram_rate(s) for s in stories]
    rng = random.Random(seed)
    groups = [rng.sample(stories, group_size) for _ in range(n_groups)]
    d1, d1_std = grouped_distinct(groups, 1)
    d2, d2_std = grouped_distinct(groups, 2)
    return {
        "n": len(stories),
        "distinct_1": d1, "distinct_1_std": d1_std,
        "distinct_2": d2, "distinct_2_std": d2_std,
        "rep4_rate": statistics.mean(rep),
        "rep4_flag_threshold": float(np.percentile(rep, REP_FLAG_PERCENTILE)),
        "proper_ending_rate": sum(s.endswith(STORY_ENDINGS) for s in stories) / len(stories),
    }


def auto_flags(sample: dict, rep_threshold: float) -> dict:
    """Automatic error flags. `hit_token_limit` is reported apart from a story that ended badly by itself."""
    hit_limit = not sample["ended_with_eos"]
    bad_ending = sample["ended_with_eos"] and not sample["text"].strip().endswith(STORY_ENDINGS)
    return {
        "repetition": repeated_ngram_rate(sample["completion"]) > rep_threshold,
        "incomplete": hit_limit or bad_ending,
        "hit_token_limit": hit_limit,
    }


class OtherCpuLoad:
    """System CPU % used by *other* programs between start() and stop().

    This process's own CPU use is subtracted. Without that, the work of loading the model looked like
    "PC busy" (18% in the first second after loading, while other programs used under 10%).
    """

    def __init__(self):
        self.proc = psutil.Process()
        self.ncpu = psutil.cpu_count() or 1

    def start(self):
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)

    def stop(self) -> float:
        system = psutil.cpu_percent(None)
        own = self.proc.cpu_percent(None) / self.ncpu
        return max(system - own, 0.0)


def measure_speed(model, tok: StoryTokenizer, device: str, n_tokens: int = 200, repeats: int = 3) -> dict:
    """Generation speed for each setting on a fixed length (no early stop), so settings are directly comparable.

    Reliable only if other programs used at most CPU_BUSY_LIMIT % both just before and during the timing.
    "During" alone is not enough: when this process fills every core, other programs get squeezed out and look idle.
    """
    load = OtherCpuLoad()
    ids = torch.tensor([tok.encode("Once upon a time", add_bos=True)], device=device)
    n_tokens = min(n_tokens, model.cfg.block_size - ids.size(1))  # same context cost for every run
    generate(model, ids, n_tokens, method="greedy")  # warm-up run, not timed
    load.start()
    time.sleep(1.0)
    other_before = load.stop()
    load.start()
    per_setting = {}
    for s in SAMPLING_SETTINGS:
        kw = {k: v for k, v in s.items() if k != "name"}
        t0 = time.perf_counter()
        for _ in range(repeats):
            generate(model, ids, n_tokens, **kw)
        per_setting[s["name"]] = repeats * n_tokens / (time.perf_counter() - t0)
    other_during = load.stop()
    other = max(other_before, other_during)
    return {
        "tokens_per_s": per_setting,
        "ms_per_token": {k: 1000 / v for k, v in per_setting.items()},
        "n_tokens": n_tokens,
        "repeats": repeats,
        "threads": torch.get_num_threads(),
        "other_cpu_before_pct": other_before,
        "other_cpu_during_pct": other_during,
        "other_cpu_pct": other,
        "reliable": other <= CPU_BUSY_LIMIT and torch.get_num_threads() == (os.cpu_count() or 1),
    }


def write_error_csv(path: str, rows: list[dict]) -> str:
    """Write the error sheet, never overwriting labels already entered by hand. Returns the path used."""
    manual = ["contradiction", "grammar", "entity_consistency", "notes"]
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            if any(row.get(k) for row in csv.DictReader(f) for k in manual):
                path = path.replace(".csv", "_new.csv")
                print(f"hand labels found in the existing sheet: writing {path} instead")
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = ["id", "prompt", "repetition", "incomplete", "hit_token_limit", *manual, "text"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({**row, **{k: "" for k in manual}})
    return path


def training_info(metrics_dir: str) -> dict | None:
    """How the model was trained, from the files train.py saved. None if they are missing."""
    try:
        with open(os.path.join(metrics_dir, "run_info.json"), encoding="utf-8") as f:
            info = json.load(f)
        with open(os.path.join(metrics_dir, "train_summary.json"), encoding="utf-8") as f:
            summary = json.load(f)
        with open(os.path.join(metrics_dir, "log.csv"), newline="", encoding="utf-8") as f:
            log = list(csv.DictReader(f))
    except FileNotFoundError:
        return None
    tc = info["config"]["train"]
    return {
        "processor": info["env"]["processor"],
        "cpu_count": info["env"]["cpu_count"],
        "ram_gb": info["env"]["ram_gb"],
        "threads": info["env"]["torch_threads"],
        "precision": tc["precision"],
        "batch_size": tc["batch_size"],
        "grad_accum_steps": tc["grad_accum_steps"],
        "effective_batch_seqs": info["effective_batch_seqs"],
        "effective_batch_tokens": info["effective_batch_tokens"],
        "steps": summary["steps"],
        "tokens_seen": summary["steps"] * info["effective_batch_tokens"],
        "wall_clock_h": summary["wall_clock_s"] / 3600,
        # train_summary's memory was "in use at the end" for older runs; the log's highest value is the true peak
        "peak_mem_mb": max(float(r["mem_mb"]) for r in log),
        "avg_tokens_per_s": statistics.mean(float(r["tokens_per_sec"]) for r in log),
        "data_dir": info["config"]["data"]["data_dir"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", default="configs/prompts.txt")
    ap.add_argument("--samples_per_prompt", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=512, help="512 fits 98%% of real validation stories")
    ap.add_argument("--val_batches", type=int, default=None, help="limit perplexity eval (default: full val and test sets)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_error", type=int, default=60, help="stories for error analysis")
    ap.add_argument("--error_setting", default="top_p_0.90")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--perf_only", action="store_true", help="only re-measure speed and update an existing report")
    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    model, ckpt = load_checkpoint(args.ckpt, device)
    cfg = ckpt["config"]
    run = cfg["run_name"]
    tok = StoryTokenizer(ckpt["tokenizer"])
    precision = cfg["train"]["precision"]
    metrics_dir = os.path.join("results", "metrics", run)
    gen_dir = os.path.join("results", "generations", run)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    eval_path = os.path.join(metrics_dir, "eval.json")
    report_path = os.path.join(metrics_dir, "eval_report.md")
    samples_path = os.path.join(gen_dir, "samples.jsonl")

    if args.perf_only:
        with open(eval_path, encoding="utf-8") as f:
            result = json.load(f)
        with open(samples_path, encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]
        result["speed"] = measure_speed(model, tok, device)
        result["env"] = env_info(device)
        result["training"] = training_info(metrics_dir)
        print_speed(result["speed"])
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        write_report(report_path, run, result, samples)
        return

    # --- Phase 9: perplexity (test = final score; val shown for comparison with training logs) ---
    losses = {}
    for split in ("val", "test"):
        ds = NextTokenDataset(load_tokens(cfg["data"]["data_dir"], split), model.cfg.block_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        t0 = time.time()
        losses[split] = evaluate_loss(model, loader, device, precision, args.val_batches)
        print(f"{split:>4} loss {losses[split]:.4f} | perplexity {math.exp(losses[split]):.2f} ({time.time() - t0:.0f}s)")

    # --- Phase 11: sampling experiment ---
    prompts = [p.strip() for p in open(args.prompts, encoding="utf-8") if p.strip()]
    reference = real_story_reference(load_stories("validation"), len(prompts), 50, args.seed)
    rep_threshold = reference["rep4_flag_threshold"]
    print(f"real stories | D1 {reference['distinct_1']:.3f} D2 {reference['distinct_2']:.3f} "
          f"rep4 {reference['rep4_rate']:.3f} | repetition flag above {rep_threshold:.3f}")

    samples, per_setting = [], {}
    for i, s in enumerate(SAMPLING_SETTINGS):
        kw = {k: v for k, v in s.items() if k != "name"}
        torch.manual_seed(args.seed + i)  # each setting is reproducible regardless of the others
        n_groups = 1 if s["method"] == "greedy" else args.samples_per_prompt  # greedy is deterministic
        groups = []  # one group = one story per prompt, so every group has the same size
        rows = []
        for _ in range(n_groups):
            group = []
            for p in prompts:
                r = generate_text(model, tok, p, args.max_new_tokens, device, **kw)
                r["setting"] = s["name"]
                rows.append(r)
                group.append(r["completion"])
            groups.append(group)
        flags = [auto_flags(r, rep_threshold) for r in rows]
        d1, d1_std = grouped_distinct(groups, 1)
        d2, d2_std = grouped_distinct(groups, 2)
        m = per_setting[s["name"]] = {
            "n": len(rows),
            "distinct_1": d1, "distinct_1_std": d1_std,
            "distinct_2": d2, "distinct_2_std": d2_std,
            "rep4_rate": statistics.mean(repeated_ngram_rate(r["completion"]) for r in rows),
            "repetition_flag_rate": sum(f["repetition"] for f in flags) / len(rows),
            "eos_rate": sum(r["ended_with_eos"] for r in rows) / len(rows),
            "hit_token_limit_rate": sum(f["hit_token_limit"] for f in flags) / len(rows),
            "avg_new_tokens": statistics.mean(r["new_tokens"] for r in rows),
        }
        samples.extend(rows)
        print(f"{s['name']:>12} | D1 {m['distinct_1']:.3f} D2 {m['distinct_2']:.3f} rep4 {m['rep4_rate']:.3f} "
              f"flagged {m['repetition_flag_rate']:.0%} eos {m['eos_rate']:.0%} n={m['n']}")

    with open(samples_path, "w", encoding="utf-8") as f:
        for r in samples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- Phase 13: error analysis ---
    setting = next(s for s in SAMPLING_SETTINGS if s["name"] == args.error_setting)
    kw = {k: v for k, v in setting.items() if k != "name"}
    torch.manual_seed(args.seed + len(SAMPLING_SETTINGS))
    err_rows = []
    for i in range(args.n_error):
        r = generate_text(model, tok, prompts[i % len(prompts)], args.max_new_tokens, device, **kw)
        err_rows.append({"id": i, "prompt": r["prompt"], "text": r["text"], **auto_flags(r, rep_threshold)})
    err_path = write_error_csv(os.path.join(gen_dir, "error_analysis.csv"), err_rows)
    error_counts = {k: sum(r[k] for r in err_rows) for k in ("repetition", "incomplete", "hit_token_limit")}

    # --- Phase 12: performance ---
    speed = measure_speed(model, tok, device)
    print_speed(speed)
    params = count_parameters(model)
    result = {
        "checkpoint": args.ckpt,
        "step": ckpt["step"],
        "val_loss": losses["val"],
        "val_perplexity": math.exp(losses["val"]),
        "test_loss": losses["test"],
        "test_perplexity": math.exp(losses["test"]),
        "params": params,
        "model_size_mb_fp32": params["total"] * 4 / 2**20,
        "peak_mem_mb": peak_memory_mb(device),
        "precision": precision,
        "max_new_tokens": args.max_new_tokens,
        "distinct_max_words": DISTINCT_MAX_WORDS,
        "reference_real_stories": reference,
        "sampling": per_setting,
        "speed": speed,
        "error_analysis": {"setting": args.error_setting, "n": len(err_rows), "auto_flag_counts": error_counts,
                           "csv": err_path},
        "env": env_info(device),
        "training": training_info(metrics_dir),
    }
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_report(report_path, run, result, samples)
    print(f"wrote {eval_path}, {report_path} and {gen_dir}/")


def print_speed(speed: dict) -> None:
    label = "reliable" if speed["reliable"] else "NOT reliable: PC busy or threads limited, re-run with --perf_only"
    print(f"speed ({speed['threads']} threads, other CPU load {speed['other_cpu_pct']:.0f}%): {label}")
    for name, tps in speed["tokens_per_s"].items():
        print(f"{name:>12} | {tps:,.1f} tok/s")


def write_report(path: str, run: str, r: dict, samples: list[dict]) -> None:
    ref, sp = r["reference_real_stories"], r["speed"]
    L = [f"# Evaluation — {run}", "",
         f"- Checkpoint: `{r['checkpoint']}` (step {r['step']})",
         f"- Parameters: {r['params']['total']:,} total ({r['params']['non_embedding']:,} non-embedding), "
         f"{r['model_size_mb_fp32']:.1f} MB fp32",
         f"- **Test** loss **{r['test_loss']:.4f}**, perplexity **{r['test_perplexity']:.2f}** "
         f"(validation: loss {r['val_loss']:.4f}, perplexity {r['val_perplexity']:.2f})",
         f"- Device: {r['env']['device']} | torch {r['env']['torch']} | precision {r['precision']} | "
         f"peak memory of the evaluation process {r['peak_mem_mb']:.0f} MB", "",
         "## Sampling strategies", "",
         f"Up to {r['max_new_tokens']} new tokens per story. Distinct-n uses the first {r['distinct_max_words']} words "
         "of each story, in groups of one story per prompt (mean ± std across groups).", "",
         "| Setting | Stories | Distinct-1 | Distinct-2 | Rep-4 rate | Flagged repetitive | EOS rate | Hit token limit | Avg tokens |",
         "|---|---|---|---|---|---|---|---|---|",
         f"| *real stories* | {ref['n']:,} | {ref['distinct_1']:.3f} ± {ref['distinct_1_std']:.3f} | "
         f"{ref['distinct_2']:.3f} ± {ref['distinct_2_std']:.3f} | {ref['rep4_rate']:.3f} | 5% by definition | — | — | — |"]
    for name, m in r["sampling"].items():
        L.append(f"| {name} | {m['n']} | {m['distinct_1']:.3f} ± {m['distinct_1_std']:.3f} | "
                 f"{m['distinct_2']:.3f} ± {m['distinct_2_std']:.3f} | {m['rep4_rate']:.3f} | "
                 f"{m['repetition_flag_rate']:.0%} | {m['eos_rate']:.0%} | {m['hit_token_limit_rate']:.0%} | "
                 f"{m['avg_new_tokens']:.0f} |")
    L += ["", f"A story is flagged repetitive when its Rep-4 rate is above {ref['rep4_flag_threshold']:.3f}, "
              f"the {REP_FLAG_PERCENTILE}th percentile of real validation stories.", "",
          "## Generation speed", ""]
    if not sp["reliable"]:
        L += [f"**Not reliable** ({sp['threads']} threads, {sp['other_cpu_pct']:.0f}% other CPU load). "
              "Re-measure on an idle PC with `--perf_only` before reporting.", ""]
    L += [f"Fixed length: {sp['n_tokens']} tokens, {sp['repeats']} runs, {sp['threads']} threads.", "",
          "| Setting | Tokens/s | ms per token | Seconds for a 200-token story |", "|---|---|---|---|"]
    L += [f"| {k} | {v:,.1f} | {sp['ms_per_token'][k]:.2f} | {200 / v:.2f} |" for k, v in sp["tokens_per_s"].items()]
    t = r.get("training")
    if t:
        L += ["", "## Training", "",
              f"- Hardware: {t['processor']}, {t['cpu_count']} logical CPUs, {t['ram_gb']} GB RAM, "
              f"{t['threads']} threads, CPU only",
              f"- Precision {t['precision']} | batch {t['batch_size']} sequences × {t['grad_accum_steps']} accumulation "
              f"steps = effective batch {t['effective_batch_seqs']} sequences ({t['effective_batch_tokens']:,} tokens)",
              f"- {t['steps']:,} steps, {t['tokens_seen'] / 1e6:.0f}M tokens seen, data `{t['data_dir']}`",
              f"- Training time **{t['wall_clock_h']:.1f} h**, average {t['avg_tokens_per_s']:,.0f} tokens/s, "
              f"peak memory {t['peak_mem_mb']:,.0f} MB (highest value in `log.csv`)"]
    ea = r["error_analysis"]
    L += ["", f"## Error analysis ({ea['n']} stories, {ea['setting']})", "",
          *[f"- {k}: {v}/{ea['n']} ({v / ea['n']:.0%}, auto-flagged)" for k, v in ea["auto_flag_counts"].items()],
          f"- Label contradiction / grammar / entity consistency by hand in `{ea['csv']}`.", "",
          "## Example generations", ""]
    seen = set()
    for s in samples:
        if s["setting"] not in seen:
            seen.add(s["setting"])
            L += [f"**{s['setting']}**", "", "> " + s["text"].replace("\n", " "), ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
