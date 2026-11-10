"""Phase 4: tokenize the cleaned splits and write train.bin / val.bin / test.bin, plus statistics.

Each story is stored as <BOS> tokens <EOS>; all stories are concatenated into a flat uint16 array.
Usage: python scripts/prepare_data.py --n_train 100000
       (pass --n_train 0 for all 1,772,187 cleaned training stories)
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_stories, token_file  # noqa: E402
from src.tokenizer import StoryTokenizer  # noqa: E402


def encode_split(tok, stories, chunk=10_000):
    lengths, parts = [], []
    for i in range(0, len(stories), chunk):
        for ids in tok.encode_batch(stories[i : i + chunk]):
            lengths.append(len(ids))
            parts.append(np.array([tok.bos_id, *ids, tok.eos_id], dtype=np.uint16))
        print(f"  encoded {min(i + chunk, len(stories)):,}/{len(stories):,}", end="\r")
    print()
    return np.concatenate(parts), lengths


def describe(lengths):
    a = np.array(lengths)
    return {"stories": int(a.size), "total": int(a.sum()), "mean": float(a.mean()), "median": float(np.median(a)),
            "min": int(a.min()), "max": int(a.max()), "p95": float(np.percentile(a, 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=100_000, help="0 = full split")
    ap.add_argument("--n_val", type=int, default=0, help="0 = full split (also used for test)")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()

    tok = StoryTokenizer(args.tokenizer)
    assert tok.vocab_size <= 65535, "uint16 storage needs vocab < 65536"
    os.makedirs(args.data_dir, exist_ok=True)
    stats = {"tokenizer": args.tokenizer, "vocab_size": tok.vocab_size, "n_train_arg": args.n_train,
             "n_val_arg": args.n_val}

    splits = [("train", "train", args.n_train), ("val", "validation", args.n_val), ("test", "test", args.n_val)]
    for split, hf_split, n in splits:
        stories = load_stories(hf_split, n or None)
        words = [len(s.split()) for s in stories]
        arr, tok_lens = encode_split(tok, stories)
        arr.tofile(token_file(args.data_dir, split))
        stats[split] = {"tokens_per_story": describe(tok_lens), "words_per_story": describe(words),
                        "total_tokens_with_specials": int(arr.size)}
        if split == "train":
            counts = Counter(w.lower().strip(".,!?\"'") for s in stories[:20_000] for w in s.split())
            stats["top_words_first_20k"] = counts.most_common(30)
            stats["shortest_story"] = min(stories, key=len)
            stats["longest_story_chars"] = max(len(s) for s in stories)
        print(f"{split}: {len(stories):,} stories, {arr.size:,} tokens -> {token_file(args.data_dir, split)}")

    with open(os.path.join(args.data_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k in ("train", "val", "test")}, indent=2))


if __name__ == "__main__":
    main()
