"""Phase 11: made-up word rate per decoding setting.

Distinct-n rewards variety, but it cannot tell a rare real word from an invented one ("neverld", "oripped").
This counts words (lower-case letters only) that never appear in the first N cleaned training stories.

Usage: python scripts/madeup_words.py --run baseline --run small [--n_known 300000]
Writes results/metrics/<run>/madeup_words.json
"""
import argparse
import collections
import json
import os
import re

import pandas as pd

WORD = re.compile(r"[a-z]+")
SETTINGS = ["greedy", "temp_0.7", "temp_1.0", "top_k_20", "top_k_50", "top_p_0.90", "top_p_0.95"]


def madeup_rate(texts, known):
    total, bad, examples = 0, 0, collections.Counter()
    for t in texts:
        words = WORD.findall(t.lower())
        total += len(words)
        for w in words:
            if w not in known:
                bad += 1
                examples[w] += 1
    return bad / max(total, 1), [w for w, _ in examples.most_common(5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--n_known", type=int, default=300_000, help="training stories used to build the word list")
    ap.add_argument("--n_real", type=int, default=2_000, help="validation stories used as the reference rate")
    args = ap.parse_args()

    known = set()
    for t in pd.read_parquet("data/clean/train.parquet")["text"][: args.n_known]:
        known.update(WORD.findall(t.lower()))
    real = pd.read_parquet("data/clean/validation.parquet")["text"][: args.n_real]
    real_rate, _ = madeup_rate(real, known)
    print(f"known words: {len(known):,} | real validation stories: {real_rate:.3%}")

    for run in args.run:
        rows = [json.loads(line) for line in open(f"results/generations/{run}/samples.jsonl", encoding="utf-8")]
        out = {"known_words": len(known), "n_known_stories": args.n_known, "real_rate": real_rate, "settings": {}}
        for s in SETTINGS:
            rate, ex = madeup_rate([r["completion"] for r in rows if r["setting"] == s], known)
            out["settings"][s] = {"madeup_rate": rate, "examples": ex}
            print(f"{run:>9} {s:<11} {rate:7.3%}  e.g. {ex}")
        with open(f"results/metrics/{run}/madeup_words.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
