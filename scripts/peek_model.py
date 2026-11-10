"""Look inside the Transformer for one sentence: embeddings, attention weights, and next-token predictions.

Usage:
    python scripts/peek_model.py --prompt "One day, a little"                       # untrained model
    python scripts/peek_model.py --ckpt checkpoints/small/model_best.pt --prompt "One day, a little"
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GPT, GPTConfig, load_checkpoint  # noqa: E402
from src.tokenizer import StoryTokenizer  # noqa: E402
from src.utils import load_config  # noqa: E402


def show(piece: str) -> str:
    return piece.replace("Ġ", "_")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="One day, a little")
    ap.add_argument("--ckpt", default=None, help="trained checkpoint; omit for an untrained model")
    ap.add_argument("--config", default="configs/small.yaml", help="model shape when --ckpt is omitted")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    tok = StoryTokenizer(args.tokenizer)
    if args.ckpt:
        model, _ = load_checkpoint(args.ckpt)
        label = f"trained ({args.ckpt})"
    else:
        torch.manual_seed(1337)
        model = GPT(GPTConfig(vocab_size=tok.vocab_size, **load_config(args.config)["model"])).eval()
        label = "UNTRAINED (random weights)"
    print(f"model: {label}\n")

    ids = tok.encode(args.prompt, add_bos=True)
    pieces = [show(tok.tok.id_to_token(i)) for i in ids]
    x = torch.tensor([ids])
    print("1) tokens and IDs")
    for p, i in zip(pieces, ids):
        print(f"   {p:<12} {i}")

    # 2) embedding: the 'meaning card' of each token (first 6 of its numbers)
    h = model.tok_emb(x) + model.pos_emb(torch.arange(len(ids)))
    print(f"\n2) embedding + position: each token -> {h.shape[-1]} numbers (first 6 shown)")
    for p, v in zip(pieces, h[0]):
        print(f"   {p:<12} [{', '.join(f'{n:+.3f}' for n in v[:6].tolist())}, ...]")

    # 3) attention of the first block, first head: how much each token looks at earlier tokens
    _, att = model.blocks[0].attn(model.blocks[0].ln1(h), return_attn=True)
    print("\n3) attention weights, block 1, head 1 (row = token looking, column = token looked at; each row sums to 1)")
    print("   " + " " * 12 + "".join(f"{p[:7]:>8}" for p in pieces))
    for r, p in enumerate(pieces):
        cells = "".join(f"{att[0, 0, r, c].item():8.2f}" if c <= r else f"{'-':>8}" for c in range(len(pieces)))
        print(f"   {p:<12}{cells}")
    print("   '-' = future token: hidden by the causal mask")

    # 4) final output: probabilities for the next token at every position
    logits, _ = model(x)
    probs = logits[0].softmax(-1)
    print(f"\n4) output: top-{args.top} guesses for the NEXT token after each position")
    for r, p in enumerate(pieces):
        top = probs[r].topk(args.top)
        guesses = ", ".join(f"{show(tok.tok.id_to_token(int(i)))} {float(q):.1%}" for q, i in zip(top.values, top.indices))
        print(f"   after {p:<12} -> {guesses}")


if __name__ == "__main__":
    main()
