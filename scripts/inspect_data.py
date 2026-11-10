"""Phase 2: download TinyStories and inspect it (no tokenizer needed).

Reports, for each split: number of stories, empty / duplicate stories, words per story
(min / mean / median / p95 / max), the most common words, and a few example stories.

Usage: python scripts/inspect_data.py              (downloads to data/hf_cache on first run)
       python scripts/inspect_data.py --n 5000     (inspect only the first 5000 stories of each split)
Writes: data/inspect_stats.json
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import clean_story, load_raw  # noqa: E402


def length_stats(values):
    a = np.array(values)
    return {"min": int(a.min()), "mean": round(float(a.mean()), 1), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": int(a.max())}


def inspect(split, n, n_common_words):
    raw = load_raw(split, n)["text"]
    empty = sum(1 for t in raw if not t or not t.strip())
    stories = [clean_story(t) for t in raw if t and t.strip()]
    words = [len(s.split()) for s in stories]
    common = Counter(w.lower().strip(".,!?\"'") for s in stories[:50_000] for w in s.split())
    return {
        "stories_raw": len(raw),
        "empty": empty,
        "stories": len(stories),
        "duplicates": len(stories) - len(set(stories)),
        "words_per_story": length_stats(words),
        "chars_per_story": length_stats([len(s) for s in stories]),
        "common_words_first_50k": common.most_common(n_common_words),
        "shortest_story": min(stories, key=len),
        "examples": stories[:3],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="only the first n stories of each split")
    ap.add_argument("--n_common_words", type=int, default=20)
    args = ap.parse_args()

    stats = {split: inspect(split, args.n, args.n_common_words) for split in ("train", "validation")}
    os.makedirs("data", exist_ok=True)
    with open("data/inspect_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    for split, s in stats.items():
        w = s["words_per_story"]
        print(f"\n=== {split} ===")
        print(f"stories: {s['stories']:,}  (empty removed: {s['empty']}, duplicates: {s['duplicates']:,})")
        print(f"words per story: min {w['min']}, mean {w['mean']}, median {w['median']}, "
              f"p95 {w['p95']}, max {w['max']}")
        print("most common words:", ", ".join(f"{word}({c:,})" for word, c in s["common_words_first_50k"][:12]))
        print(f"shortest story: {s['shortest_story']!r}")
        print(f"example: {s['examples'][0][:300]}...")
    print("\nsaved data/inspect_stats.json")


if __name__ == "__main__":
    main()
