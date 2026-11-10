"""Phase 2 (cleaning): build the cleaned TinyStories splits used by every later phase.

Train split, in this order:
  1. normalise text (quotes, whitespace) and drop empty stories
  2. drop cut-off stories: fewer than --min_words words, or not ending in . ! ? or "
  3. drop exact duplicates (keep the first occurrence, original order)
  4. drop any story that also appears in the official validation split (prevents test leakage)
Official validation split: only normalised, then split into two fixed halves (seeded shuffle):
  validation -> progress checks during training and choosing model_best.pt
  test       -> final reported scores only (TinyStories has no official test split)

Usage: python scripts/clean_data.py
Writes: data/clean/{train,validation,test}.parquet, data/clean/clean_report.json
"""
import argparse
import json
import os
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CLEAN_DIR, clean_story, load_raw  # noqa: E402

ENDINGS = (".", "!", "?", '"')


def save(stories, split):
    path = os.path.join(CLEAN_DIR, f"{split}.parquet")
    pq.write_table(pa.table({"text": stories}), path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_words", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337, help="seed for the validation/test split")
    args = ap.parse_args()
    os.makedirs(CLEAN_DIR, exist_ok=True)
    report = {"min_words": args.min_words}

    # Official validation: normalise only, then split into validation / test halves.
    held_out = [clean_story(t) for t in load_raw("validation")["text"] if t and t.strip()]
    val_set = set(held_out)  # train must not overlap either half
    order = list(range(len(held_out)))
    random.Random(args.seed).shuffle(order)
    half = len(order) // 2
    val = [held_out[i] for i in sorted(order[:half])]
    test = [held_out[i] for i in sorted(order[half:])]
    report["validation"] = {"stories": len(val)}
    report["test"] = {"stories": len(test), "split_seed": args.seed}
    print(f"official validation: {len(held_out):,} stories -> validation {len(val):,} + test {len(test):,}")

    # Train: step by step, counting what each step removes.
    raw = load_raw("train")["text"]
    steps = [("raw", len(raw))]

    stories = [clean_story(t) for t in raw if t and t.strip()]
    steps.append(("after removing empty", len(stories)))

    stories = [s for s in stories if len(s.split()) >= args.min_words and s.endswith(ENDINGS)]
    steps.append((f"after removing cut-off (<{args.min_words} words or no end punctuation)", len(stories)))

    seen, unique = set(), []
    for s in stories:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    stories = unique
    steps.append(("after removing duplicates", len(stories)))

    stories = [s for s in stories if s not in val_set]
    steps.append(("after removing stories that are in validation", len(stories)))

    print("train:")
    prev = None
    for name, n in steps:
        removed = f"  (-{prev - n:,})" if prev is not None else ""
        print(f"  {name:<70} {n:>10,}{removed}")
        prev = n
    report["train_steps"] = [{"step": name, "stories": n} for name, n in steps]
    report["train"] = {"stories": len(stories)}

    assert not (set(stories) & val_set), "train/validation overlap remains"
    print("check: train, validation and test share 0 stories")

    assert not (set(val) & set(test)), "validation/test overlap"
    for split, data in (("train", stories), ("validation", val), ("test", test)):
        print(f"saved {save(data, split)}")
    with open(os.path.join(CLEAN_DIR, "clean_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"saved {os.path.join(CLEAN_DIR, 'clean_report.json')}")


if __name__ == "__main__":
    main()
