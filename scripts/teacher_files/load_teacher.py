"""Load the Project 1 teacher. Works from inside this folder, wherever the folder is copied.

    from load_teacher import load_teacher
    model, tok, info = load_teacher()          # model in eval mode, on CPU
    probs = next_token_probs(model, tok, "Once upon a time")   # what the teacher thinks comes next

For distillation, the student must use the SAME tokenizer (tokenizer.json, 8,192 tokens), otherwise the
teacher's probabilities over token IDs mean nothing to it.
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.model import GPT, GPTConfig  # noqa: E402
from src.tokenizer import StoryTokenizer  # noqa: E402


def load_teacher(device: str = "cpu"):
    ckpt = torch.load(os.path.join(HERE, "teacher.pt"), map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    tok = StoryTokenizer(os.path.join(HERE, ckpt["tokenizer"]))
    info = {k: ckpt[k] for k in ("model_config", "config", "step", "best_val_loss", "source_checkpoint")}
    return model, tok, info


@torch.no_grad()
def next_token_probs(model, tok, text: str, temperature: float = 1.0) -> torch.Tensor:
    """Teacher's probability for every token (8,192) as the next token after `text`."""
    ids = torch.tensor([tok.encode(text, add_bos=True)[-model.cfg.block_size:]])
    logits, _ = model(ids)
    return torch.softmax(logits[0, -1] / temperature, dim=-1)
