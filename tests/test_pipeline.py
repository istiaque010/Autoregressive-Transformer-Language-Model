"""Pipeline checks that need no download: tokenizer round-trip, shifted targets, decoding methods."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import NextTokenDataset  # noqa: E402
from src.generate import generate, select_next  # noqa: E402
from src.model import GPT, GPTConfig, count_parameters  # noqa: E402
from src.tokenizer import StoryTokenizer, train_bpe  # noqa: E402

TEXTS = ["Once upon a time there was a little dog.", "The little boy played with his dog.",
         "Lily liked to play in the park with her mom."] * 50


def test_tokenizer_round_trip(tmp_path):
    path = str(tmp_path / "tok.json")
    train_bpe(TEXTS, vocab_size=300, min_frequency=1).save(path)
    tok = StoryTokenizer(path)
    assert (tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id) == (0, 1, 2, 3)
    s = "The little boy played with his dog."
    ids = tok.encode(s, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == s


def test_targets_are_inputs_shifted_by_one():
    ds = NextTokenDataset(np.arange(50, dtype=np.uint16), block_size=8)
    assert len(ds) == 6
    x, y = ds[1]
    assert x.tolist() == list(range(8, 16))
    assert y.tolist() == list(range(9, 17))


def test_decoding_methods():
    logits = torch.tensor([[1.0, 5.0, 3.0, 0.0]])
    assert select_next(logits, "greedy").item() == 1
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        assert select_next(logits, "top_k", top_k=2, generator=g).item() in (1, 2)
        assert select_next(logits, "top_p", top_p=0.5, generator=g).item() == 1  # p(1) ≈ 0.84 > 0.5
        assert select_next(logits, "temperature", temperature=0.7, generator=g).item() in range(4)


def test_model_forward_and_generate():
    model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0))
    params = count_parameters(model)
    assert params["total"] == params["trainable"] > 0
    x = torch.randint(0, 64, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 64) and loss.item() > 0
    out, n = generate(model, x[:, :4], max_new_tokens=30, method="top_k", top_k=10)
    assert out.shape == (2, 34) and n == 30  # longer than block_size: context gets cropped
