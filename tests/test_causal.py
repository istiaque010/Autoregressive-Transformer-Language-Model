"""Phase 6 milestone: future tokens must not influence earlier predictions."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attention import CausalSelfAttention  # noqa: E402
from src.model import GPT, GPTConfig  # noqa: E402


def small_model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)).eval()


def test_attention_weights_are_lower_triangular():
    attn = CausalSelfAttention(n_embd=32, n_head=4, block_size=8).eval()
    _, att = attn(torch.randn(2, 8, 32), return_attn=True)
    future = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    assert torch.all(att[..., future] == 0)
    assert torch.allclose(att.sum(-1), torch.ones(2, 4, 8))


def test_changing_future_tokens_does_not_change_past_logits():
    model = small_model()
    x = torch.randint(0, 100, (1, 12))
    for t in range(1, 12):
        x2 = x.clone()
        x2[0, t:] = torch.randint(0, 100, (12 - t,))  # rewrite every token from position t onwards
        with torch.no_grad():
            a, _ = model(x)
            b, _ = model(x2)
        assert torch.allclose(a[0, :t], b[0, :t], atol=1e-5), f"position < {t} saw the future"


def test_gradient_of_past_output_wrt_future_input_is_zero():
    model = small_model()
    x = torch.randint(0, 100, (1, 10))
    emb = model.tok_emb(x).detach().requires_grad_(True)
    h = emb + model.pos_emb(torch.arange(10))
    for blk in model.blocks:
        h = blk(h)
    h[0, 4].sum().backward()
    assert torch.all(emb.grad[0, 5:] == 0)
    assert emb.grad[0, :5].abs().sum() > 0
