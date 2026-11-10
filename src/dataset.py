"""TinyStories loading (Phase 2) and fixed-length next-token datasets (Phase 4)."""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

DATASET_NAME = "roneneldan/TinyStories"
CACHE_DIR = "data/hf_cache"  # keep the download inside the project (on D:)
CLEAN_DIR = "data/clean"     # cleaned splits written by scripts/clean_data.py


def load_raw(split: str, n: Optional[int] = None, cache_dir: str = CACHE_DIR):
    """Hugging Face dataset for a TinyStories split ("train" or "validation"), optionally the first `n` rows."""
    from datasets import load_dataset

    split_arg = split if n is None else f"{split}[:{n}]"
    return load_dataset(DATASET_NAME, split=split_arg, cache_dir=cache_dir)


def load_stories(split: str, n: Optional[int] = None, clean_dir: str = CLEAN_DIR) -> List[str]:
    """Load the first `n` stories of a cleaned split (made by scripts/clean_data.py).

    The order is the dataset's fixed order, so the same `n` always gives the same subset.
    """
    import pyarrow.parquet as pq

    path = os.path.join(clean_dir, f"{split}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run scripts/clean_data.py first")
    texts = pq.read_table(path, columns=["text"]).column("text")
    return (texts if n is None else texts.slice(0, n)).to_pylist()


def clean_story(text: str) -> str:
    # Normalise whitespace and the few odd unicode quotes found in TinyStories.
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return " ".join(text.split())


def token_file(data_dir: str, split: str) -> str:
    return os.path.join(data_dir, f"{split}.bin")


def load_tokens(data_dir: str, split: str) -> np.ndarray:
    return np.memmap(token_file(data_dir, split), dtype=np.uint16, mode="r")


class NextTokenDataset(Dataset):
    """Non-overlapping windows of `block_size + 1` tokens from the flat token stream.

    Stories are stored as <BOS> story <EOS> back to back, so a window may span a story
    boundary; the <EOS>/<BOS> tokens teach the model where stories end and begin.
    Returns (x, y) with y = x shifted left by one: the next-token targets.
    """

    def __init__(self, tokens: np.ndarray, block_size: int):
        self.tokens = tokens
        self.block_size = block_size
        self.n = (len(tokens) - 1) // block_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        start = i * self.block_size
        chunk = torch.from_numpy(self.tokens[start : start + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]
