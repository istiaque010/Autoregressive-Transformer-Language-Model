"""Phase 10 learning aid: watch generation one token at a time.

Part 1 shows, for one prompt, the model's top guesses for the next token and which of them each decoding method
is allowed to pick (temperature reshapes the probabilities; top-k and top-p cut the list).
Part 2 runs the loop "predict -> pick -> append" for a few steps and prints each step.

Usage: python scripts/show_decoding.py --ckpt checkpoints/small/model_best.pt [--prompt "Tom was sad because"]
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generate import select_next  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.tokenizer import StoryTokenizer  # noqa: E402


def show(tok, i):
    return repr(tok.decode([i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Tom was sad because")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.ckpt, "cpu")
    model.eval()
    tok = StoryTokenizer(ckpt.get("tokenizer", "data/tokenizer.json"))
    ids = tok.encode(args.prompt, add_bos=True)

    # Part 1: one step, every method
    with torch.no_grad():
        logits = model(torch.tensor([ids]))[0][0, -1]
    probs = F.softmax(logits, -1)
    order = probs.argsort(descending=True)
    cum = probs[order].cumsum(0)
    n_p90 = int((cum - probs[order] <= 0.90).sum())
    n_p95 = int((cum - probs[order] <= 0.95).sum())
    t07 = F.softmax(logits / 0.7, -1)

    print(f'Prompt: "{args.prompt}"  ->  what comes next?\n')
    print(f"{'rank':>4}  {'token':<14}{'prob':>7}{'temp 0.7':>10}{'cumul.':>8}   allowed by")
    for r in range(args.top):
        i = int(order[r])
        allowed = ["greedy"] if r == 0 else []
        allowed += ["top-k 20"] if r < 20 else []
        allowed += ["top-p 0.90"] if r < n_p90 else []
        allowed += ["top-p 0.95"] if r < n_p95 else []
        print(f"{r + 1:>4}  {show(tok, i):<14}{probs[i]:>7.1%}{t07[i]:>10.1%}{cum[r]:>8.1%}   {', '.join(allowed)}")
    print(f"\ntop-p 0.90 keeps {n_p90} tokens, top-p 0.95 keeps {n_p95}, top-k 20 keeps 20, temperature keeps all"
          f" {len(probs):,}.")

    # Part 2: the loop, greedy and top-p side by side
    for method, kw in [("greedy", {}), ("top_p", {"top_p": 0.9})]:
        g = torch.Generator().manual_seed(args.seed)
        x = torch.tensor([ids])
        print(f"\n--- the loop with {method} {kw or ''} ---")
        for step in range(args.steps):
            with torch.no_grad():
                lg = model(x)[0][:, -1, :]
            p = F.softmax(lg, -1)[0]
            nxt = select_next(lg, method=method, generator=g, **kw)
            i = int(nxt)
            rank = int((p > p[i]).sum()) + 1
            x = torch.cat([x, nxt], 1)
            print(f"step {step + 1:>2}: picked {show(tok, i):<12} prob {p[i]:>6.1%} (rank {rank})")
            if i == tok.eos_id:
                break
        print("text so far:", tok.decode(x[0].tolist()[1:]))


if __name__ == "__main__":
    main()
