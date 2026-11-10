"""Autoregressive text generation with several decoding strategies (Phase 10).

Usage:
    python -m src.generate --ckpt checkpoints/small/model_best.pt --prompt "Once upon a time" --method top_p --top_p 0.9
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import torch
import torch.nn.functional as F

from .model import GPT, load_checkpoint
from .tokenizer import StoryTokenizer
from .utils import resolve_device, set_seed


def select_next(logits: torch.Tensor, method: str = "greedy", temperature: float = 1.0,
                top_k: Optional[int] = None, top_p: Optional[float] = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """logits: (B, vocab) for the last position. Returns next token ids (B, 1)."""
    if method == "greedy":
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-8)
    if method == "top_k":
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    elif method == "top_p":
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Drop tokens once the cumulative mass *before* them already exceeds top_p (always keep the first).
        remove = (cum - F.softmax(sorted_logits, dim=-1)) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    elif method != "temperature":
        raise ValueError(f"unknown method {method}")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, eos_id: Optional[int] = None, **sample_kwargs):
    """Repeatedly predict the next token and append it. Stops early when every row emitted <EOS>.

    Returns (token ids including the prompt, number of new tokens generated).
    """
    model.eval()
    block = model.cfg.block_size
    finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
    n_new = 0
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block:])
        nxt = select_next(logits[:, -1, :], **sample_kwargs)
        idx = torch.cat([idx, nxt], dim=1)
        n_new += 1
        if eos_id is not None:
            finished |= nxt.squeeze(1) == eos_id
            if finished.all():
                break
    return idx, n_new


def generate_text(model: GPT, tok: StoryTokenizer, prompt: str, max_new_tokens: int = 200, device: str = "cpu",
                  **sample_kwargs) -> dict:
    ids = tok.encode(prompt, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    out, n_new = generate(model, x, max_new_tokens, eos_id=tok.eos_id, **sample_kwargs)
    latency = time.perf_counter() - t0
    new_ids = out[0, len(ids):].tolist()
    ended = tok.eos_id in new_ids
    if ended:
        new_ids = new_ids[: new_ids.index(tok.eos_id)]
    return {
        "prompt": prompt,
        "completion": tok.decode(new_ids),
        "text": prompt + tok.decode(new_ids),
        "new_tokens": len(new_ids),
        "ended_with_eos": ended,
        "latency_s": latency,
        "tokens_per_s": n_new / max(latency, 1e-9),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=None, help="defaults to the path stored in the checkpoint")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--method", default="top_p", choices=["greedy", "temperature", "top_k", "top_p"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    model, ckpt = load_checkpoint(args.ckpt, device)
    tok = StoryTokenizer(args.tokenizer or ckpt["tokenizer"])
    for i in range(args.num_samples):
        r = generate_text(model, tok, args.prompt, args.max_new_tokens, device, method=args.method,
                          temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        print(f"--- sample {i + 1} ({r['new_tokens']} tokens, {r['tokens_per_s']:.1f} tok/s, eos={r['ended_with_eos']}) ---")
        print(r["text"])


if __name__ == "__main__":
    main()
