# TinyStories teacher (Project 1 → Project 2)

A 13.9M-parameter GPT-style decoder-only Transformer, written from scratch in plain PyTorch and trained on
TinyStories. It is the **teacher** for Project 2 (distillation into a smaller student).

**Check it first:** `python verify_teacher.py` (about 3 minutes; `--quick` takes about 15 s). It checks every file
against `MANIFEST.sha256`, loads the model and reproduces the test perplexity. Exit code 0 means it passed.

## Results

| | |
|---|---|
| **Test perplexity** | **5.74** (loss 1.7473), 10,995 held-out stories |
| Validation perplexity | 5.74 (loss 1.7467) |
| Baseline (5.3M params, 100K stories) | 9.68 |
| Stories with no error, hand-labelled (60, top-p 0.90) | 5/60. Contradictions 88%, character mix-ups 55%, grammar 37%, repetition 22% |
| Speed, CPU, top-p 0.95 | 141 tokens/s, 1.4 s per 200-token story |

Perplexity is per token over the test stream, in non-overlapping 256-token windows, batch 16, fp32. Stories are
joined as `<BOS> story <EOS>`, so a window can cross a story boundary. Compute it the same way in Project 2,
or the numbers won't be comparable.

**What the teacher is good and bad at:** fluent sentences and proper endings with a moral. It can't keep facts
and characters consistent across a whole story (names change, objects turn into other objects). A student
can't be expected to do better than this. Details: `results/eval_report.md`, `results/error_analysis.csv`.

## Model

| | |
|---|---|
| Architecture | Pre-LN decoder-only Transformer (GPT-2 style), learned position embeddings, GELU MLP 4×, tied input/output embeddings, causal self-attention written by hand |
| Size | 6 layers, 6 heads, width 384, context 256 tokens, dropout 0.1, bias on |
| Parameters | 13,891,584 (10,647,552 non-embedding), 53 MB fp32 |
| Tokenizer | byte-level BPE, 8,192 tokens: `<PAD>`=0, `<UNK>`=1, `<BOS>`=2, `<EOS>`=3 |
| Tokenizer SHA-256 | `158ade557bac5c94223fb23d12cd125c58b481e3146d77417e6a5eb777c0f530` |

**The student must use this exact tokenizer.** The teacher's probabilities are over these 8,192 token IDs.

## Training

All 1,772,187 cleaned TinyStories training stories, of which it saw about 44% (164M tokens).
- **Optimiser:** AdamW, lr 6e-4 with 500 warmup steps, then cosine decay to 6e-5. Weight decay 0.1, betas (0.9, 0.95), grad clip 1.0.
- **Batches:** batch 8 × 8 accumulation steps = 64 sequences (16,384 tokens) per step. 10,000 steps.
- **Run:** seed 1337, fp32, CPU only (24 threads), 9.9 h, peak memory 1.4 GB.
- **Config:** `configs/small.yaml`.

Cleaning removed stories that also appear in validation or test. The official validation set was split
in two (seed 1337) to make validation and test, 10,995 stories each.

## Use

```python
from load_teacher import load_teacher, next_token_probs
model, tok, info = load_teacher()                       # CPU, eval mode
p = next_token_probs(model, tok, "Tom was sad because")  # tensor of 8,192 probabilities
print(tok.decode([int(p.argmax())]))                     # ' he'
```

For distillation, run the teacher on a batch `x` of token IDs: `logits, _ = model(x)` gives
`(batch, length, 8192)`. Soften with a temperature and train the student to match.

Writing stories: `python -m src.generate --ckpt teacher.pt --tokenizer tokenizer.json --method top_p --top_p 0.95`.
Recommended decoding is **top-p 0.95**; avoid greedy, which loops.

## Environment

Windows 11, Intel CPU (24 logical CPUs), 63.7 GB RAM, Python 3.12.10, `torch==2.14.1+cpu`, `tokenizers==0.23.3`,
`numpy==2.5.4`. The full list is in `requirements.lock.txt`, and the hardware report is in `results/env_report.json`.

## Files

| File | What |
|---|---|
| `teacher.pt` | weights + model config + training config (no optimizer state) |
| `tokenizer.json`, `model_config.json`, `configs/` | tokenizer, model shape, training config, evaluation prompts |
| `src/` | the model, tokenizer, data and evaluation code (needed to load a plain-PyTorch model) |
| `data/val.bin`, `data/test.bin` | uint16 token IDs used for perplexity (`stats.json`: data statistics) |
| `results/` | evaluation report, training log, loss curve, generated samples, hand-labelled errors |
| `expected.json` | the scores `verify_teacher.py` checks against |
| `MANIFEST.sha256` | SHA-256 of every file |

Rebuild this folder from the project with `python scripts/package_teacher.py`.
