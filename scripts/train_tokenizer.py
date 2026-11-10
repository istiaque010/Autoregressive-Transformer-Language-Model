"""Phase 3: train a byte-level BPE tokenizer on a TinyStories subset and save data/tokenizer.json.

Usage: python scripts/train_tokenizer.py --n_stories 100000 --vocab_size 8192
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_stories  # noqa: E402
from src.tokenizer import StoryTokenizer, train_bpe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_stories", type=int, default=100_000)
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--out", default="data/tokenizer.json")
    args = ap.parse_args()

    stories = load_stories("train", args.n_stories)
    print(f"training BPE (vocab {args.vocab_size}) on {len(stories):,} stories")
    tok = train_bpe(stories, vocab_size=args.vocab_size)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)

    # Milestone check: round-trip encode/decode.
    t = StoryTokenizer(args.out)
    sample = "The little boy played with his dog."
    ids = t.encode(sample)
    print("tokens:", t.tokens(sample))
    print("ids:", ids)
    assert t.decode(ids) == sample, "round-trip failed"
    print(f"round-trip OK | vocab={t.vocab_size} | specials pad={t.pad_id} unk={t.unk_id} bos={t.bos_id} eos={t.eos_id}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
