"""Byte-level BPE tokenizer (Phase 3) built on Hugging Face `tokenizers`."""
from __future__ import annotations

from typing import Iterable, List

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]  # ids 0, 1, 2, 3


def train_bpe(texts: Iterable[str], vocab_size: int = 8192, min_frequency: int = 2) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=UNK))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


class StoryTokenizer:
    """Thin wrapper exposing encode/decode plus special-token ids."""

    def __init__(self, path: str):
        self.tok = Tokenizer.from_file(path)
        self.pad_id = self.tok.token_to_id(PAD)
        self.unk_id = self.tok.token_to_id(UNK)
        self.bos_id = self.tok.token_to_id(BOS)
        self.eos_id = self.tok.token_to_id(EOS)
        self.vocab_size = self.tok.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special)

    def tokens(self, text: str) -> List[str]:
        return self.tok.encode(text).tokens
