# Project 1 — Small Autoregressive Language Model (TinyStories)

Source plan: `docs/Project_1_Small_Autoregressive_LM_Local_PC_Plan.docx`.
This file turns that plan into steps you can follow one phase at a time.

**Goal:** build and train a small GPT-style model from scratch that writes short children's stories,
then measure how model size and decoding strategy change quality and speed.
The best trained model becomes the *teacher* for Project 2 (knowledge distillation).

**The core idea:** the model learns to predict the next token from all the previous ones.
```
Once upon          → a
Once upon a        → time
Once upon a time   → there
```
Formally: P(x₁, …, xₜ) = ∏ P(xₜ | x₍<t₎)

**Pipeline:**
TinyStories → BPE tokenizer → sequences → Transformer → causal attention → training →
perplexity → text generation → sampling experiments → error/performance analysis → save as teacher.

---

## How to use this file

1. Open PowerShell in the project folder and activate the environment:
   ```
   cd "D:\TTI\Projrcts List\project1"
   .venv\Scripts\activate
   ```
2. Work through the phases in order. Each phase has:
   - **Learn:** the idea to understand before running anything
   - **Files:** the code for this phase (read it)
   - **Run:** the commands
   - **Check:** what you should see; the phase is done when this is true
3. Tick the status box when a phase is done.

## Progress

| # | Phase | Status |
|---|---|---|
| 1 | Environment setup | ✅ Done (2026-10-15) |
| 2 | Get TinyStories | ✅ Done (2026-10-15) |
| 3 | Build the tokenizer | ✅ Done (2026-10-15) |
| 4 | Prepare training sequences | ✅ Done (2026-10-15) |
| 5 | Build the Transformer | ✅ Done (2026-10-15) |
| 6 | Causal attention | ✅ Done (2026-10-15) |
| 7 | Choose the first model size | ✅ Done (2026-10-15) |
| 8 | Train the model | ✅ Baseline and `small` trained (`small` val perplexity 5.90) |
| 9 | Perplexity | ✅ Test perplexity: baseline 9.68, `small` 5.74 |
| 10 | Text generation | ✅ Done (2026-11-06) |
| 11 | Sampling experiment | ✅ Default decoding: top-p 0.95 (2026-11-06) |
| 12 | Evaluate the model | ✅ Done (2026-11-06) |
| 13 | Error analysis | ✅ Done (2026-11-06): contradictions 88%, entity errors 55%, grammar 37% |
| 14 | Package as teacher for Project 2 | ✅ Done (2026-11-06): `teacher/`, verified outside the project |

---

## Phase 1 — Environment setup ✅

**Learn:** PyTorch is the library that holds tensors (arrays of numbers) and computes gradients.
CUDA is NVIDIA-only; AMD GPUs need ROCm. We start on CPU so we're not blocked by GPU drivers.

**Files:** `scripts/env_report.py`, `requirements.txt`, `requirements.lock.txt`

**Run:**
```
python scripts/env_report.py
```

**Check:** the smoke-test loss goes down (it went 6.26 → 0.54) and `results/metrics/env_report.json` exists.

**Result:**
- Windows 11, Python 3.12.10, PyTorch 2.14.1+cpu, in `.venv/` on `D:\` (not WSL, a deviation from the plan).
- CPU: Intel Core Ultra 7 270K Plus, 24 threads, 64 GB RAM.
- GPU: AMD Radeon RX 9060, driver 32.0.31044.16. AMD's WSL2 table lists it for ROCm 7.2.1 + PyTorch 2.9.1
  (needs Adrenalin 26.1.1 and Ubuntu 24.04). **Not set up.** Decision: stay on CPU and revisit at Phase 8
  only if training is too slow. The code picks the GPU automatically if one ever becomes available.

---

## Phase 2 — Get TinyStories

**Learn:** TinyStories is a dataset of short, simple stories made to test whether *very small* models can
learn to write coherent text. It has a `train` split (~2.1M stories) and a `validation` split.
Always look at your data before training on it. Start small (a few thousand stories) so mistakes are cheap,
then scale to 50K–100K. Keep the train/validation split fixed so results are comparable.

**Files:** `src/dataset.py` (`load_raw`, `load_stories`, `clean_story`), `scripts/inspect_data.py`,
`scripts/clean_data.py`

**Run:**
```
python scripts/inspect_data.py            # 1. download (first run) and inspect
python scripts/clean_data.py              # 2. clean -> data/clean/train, validation, test .parquet
```

**Check:** you can say how many stories there are, how long they are, and what they look like.
Train, validation and test share 0 stories. Statistics: `data/inspect_stats.json`, `data/clean/clean_report.json`.

**Result (2026-10-15):** done.

| | Train | Validation |
|---|---|---|
| Stories | 2,119,489 (230 empty removed) | 21,990 |
| Exact duplicates | 320,241 (15%) | 0 |
| Words per story (min / median / mean / p95 / max) | 5 / 153 / 175 / 349 / 963 | 13 / 151 / 171 / 332 / 841 |

Findings:
- **6,601 validation stories (30%) also appear in train.** This leaks test data into training and would make
  validation loss and perplexity look better than they really are.
- 24,531 train stories (1.2%) don't end with punctuation (cut off mid-sentence, e.g. `"Once upon a time, there"`).
- Most common words: the, and, to, a, was, he, she, they, it: a very simple vocabulary, as intended.
- Storage: raw download 955 MB in `C:\Users\home\.cache\huggingface`, processed copy 1.8 GB in `data/hf_cache`.

Cleaning (`scripts/clean_data.py`). Train:

| Step | Stories left | Removed |
|---|---|---|
| Raw | 2,119,719 | |
| Remove empty | 2,119,489 | 230 |
| Remove cut-off (< 20 words or no ending . ! ? ") | 2,094,957 | 24,532 |
| Remove duplicates | 1,778,710 | 316,247 |
| Remove stories also in validation | **1,772,187** | 6,523 |

**Test set.** TinyStories has no official test split, and the validation set is used during training to
pick `model_best.pt`, so its score is slightly optimistic. The 21,990 official validation stories (text unchanged)
are therefore split into two fixed halves (seeded shuffle, seed 1337):

| Set | Stories | Used for |
|---|---|---|
| validation | 10,995 | progress checks during training, choosing `model_best.pt` |
| test | 10,995 | **final reported scores only** (Phase 9 onwards); never used to make choices |

All later phases read the cleaned files through `load_stories()`.

---

## Phase 3 — Build the tokenizer

**Learn:** models work with numbers, not text. A tokenizer splits text into pieces (*tokens*) and maps each
to an ID. **BPE** (Byte-Pair Encoding) starts from single characters and repeatedly merges the most common
pairs, so common words become one token and rare words split into pieces:
```
"The little boy played with his dog."
→ ["The", " little", " boy", " played", " with", " his", " dog", "."]
```
Vocabulary size: 8,192 tokens. Special tokens:

| Token | ID | Meaning |
|---|---|---|
| `<PAD>` | 0 | padding |
| `<UNK>` | 1 | unknown |
| `<BOS>` | 2 | beginning of a story |
| `<EOS>` | 3 | end of a story |

**Files:** `src/tokenizer.py`, `scripts/train_tokenizer.py`

**Run:**
```
python scripts/train_tokenizer.py --n_stories 100000 --vocab_size 8192
```

**Check:** it prints the tokens for the sample sentence, and decoding the IDs gives back the exact sentence.
`data/tokenizer.json` exists.

**Result (2026-10-15):** done. Trained on the first 100,000 cleaned train stories; `data/tokenizer.json` is 543 KB.

| What | Result |
|---|---|
| Round-trip encode → decode | exact match ✅ |
| IDs 0–3 | `<PAD>` `<UNK>` `<BOS>` `<EOS>` |
| First merges learned | `h+e`, `Ġ+t`, `Ġ+a`, `Ġ+s`, `Ġ+w`, `n+d`, `Ġt+he`, … (most common pairs first) |
| `"The little boy played with his dog."` | `The │ Ġlittle │ Ġboy │ Ġplayed │ Ġwith │ Ġhis │ Ġdog │ .` → `[1607, 399, 527, 768, 347, 353, 645, 17]` |
| `"I love 😊"` | the emoji becomes 4 byte-pieces, still decodes exactly |
| Validation set | 204 tokens per story, 1.20 tokens per word, 4.25 characters per token, 0 `<UNK>` |
| Stories that fit in 256 tokens | 85% (longer ones continue into the next window) |

`Ġ` is how byte-level BPE shows a **space**: `Ġdog` means " dog" (a space followed by dog).

---

## Phase 4 — Prepare training sequences

**Learn:** every story becomes `<BOS> tokens <EOS>`, and all stories are joined into one long list of IDs.
That list is cut into fixed-length windows (length 256). The key trick is **shift by one**:
```
tokens  [BOS, The, boy, likes, dogs, EOS]
input   [BOS, The, boy, likes, dogs]
target  [The, boy, likes, dogs, EOS]
```
At every position the target is simply the next token, so one window gives 256 training examples at once.

**Files:** `scripts/prepare_data.py`, `src/dataset.py` (`NextTokenDataset`)

**Run:**
```
python scripts/prepare_data.py --n_train 100000 --n_val 5000
```

**Check:** `data/train.bin`, `data/val.bin` and `data/stats.json` exist, and a DataLoader returns `x` and `y`
where `y` is `x` shifted by one (covered by `tests/test_pipeline.py::test_targets_are_inputs_shifted_by_one`).

**Result (2026-10-15):** done. Command used: `python scripts/prepare_data.py --n_train 100000`
(first 100K cleaned train stories; all validation and test stories).

| | Train | Validation | Test |
|---|---|---|---|
| Stories | 100,000 | 10,995 | 10,995 |
| Tokens (incl. `<BOS>`/`<EOS>`) | 20,779,778 | 2,265,953 | 2,264,073 |
| Tokens per story (median / mean / max) | 179 / 206 / 1,012 | 179 / 204 / 1,019 | 178 / 204 / 999 |
| File | `data/train.bin` (40 MB) | `data/val.bin` (4.4 MB) | `data/test.bin` (4.4 MB) |
| Windows of 256 tokens | 81,171 | 8,851 | 8,844 |

- Start of `train.bin`: `[2, 421, 360, 15, 262, 399, …]` = `<BOS> One Ġday , Ġa Ġlittle …`
- Story boundary: `… Ġtogether . <EOS> <BOS> Once Ġupon …`
- Decoding story 1 from `train.bin` gives back the original text exactly.
- Window 0: `x[:6] = <BOS> One Ġday , Ġa Ġlittle` → `y[:6] = One Ġday , Ġa Ġlittle Ġgirl` (shifted by one ✅)
- One DataLoader batch: `x` and `y` both shape `(8, 256)`, type int64. Test passes.
- Each number is stored as 2 bytes (`uint16`, max 65,535), which is enough for 8,192 IDs: 20.8M tokens × 2 = 40 MB.
- Later, for the final model: `--n_train 0` tokenizes all 1,772,187 stories (~360M tokens, ~720 MB).

---

## Phase 5 — Build the Transformer

**Learn:** the model is built ourselves, not downloaded (no pretrained GPT-2).
```
Token IDs
  → Token embedding + Position embedding
  → [ Transformer block ] × N
       LayerNorm → Causal multi-head attention → + residual
       LayerNorm → Feed-forward (MLP)          → + residual
  → LayerNorm
  → Linear head
  → Logits (a score for every word in the vocabulary)
```
- **Embedding:** turns each token ID into a vector; position embedding tells the model *where* the token is.
- **Attention:** lets each token look at earlier tokens to gather context.
- **MLP:** processes each position on its own.
- **Residual (+):** adds the input back so deep models train stably.
- **Weight tying:** the output layer reuses the token embedding matrix (fewer parameters).

**Files:** `src/transformer.py` (`Block`, `MLP`), `src/model.py` (`GPT`, `GPTConfig`, `count_parameters`)

**Run:**
```
python -m pytest tests/test_pipeline.py::test_model_forward_and_generate -v
python scripts/peek_model.py --prompt "One day, a little"     # look inside: embeddings, attention, predictions
```
`peek_model.py` also takes `--ckpt checkpoints/<run>/model_best.pt` to look inside a trained model (after Phase 8).

**Check:** the model takes token IDs and returns logits of shape `(batch, length, vocab_size)` plus a loss.

**Result (2026-10-15):** done. A real batch through the target model (`small.yaml`):
```
input x                 (8, 256)            8 windows × 256 token IDs
token embedding         (8, 256, 384)       each ID → 384 numbers
+ position embedding    (8, 256, 384)
after 6 blocks          (8, 256, 384)       same shape, richer meaning
logits                  (8, 256, 8192)      a score for every token in the vocabulary, at every position
```
- Untrained loss **9.092**, close to random guessing = ln(8192) = **9.011**: correct for a model that knows nothing yet.
- Untrained guesses after "One day, a little" (true next: `Ġgirl`): `Ġmarkers`, `Ġsquirrels`, `Ġgun`, each ~0.05%.
- Weight tying confirmed (output layer reuses the token embedding matrix).

| Config | Shape | Parameters | Embeddings | Blocks + rest | Size (fp32) |
|---|---|---|---|---|---|
| smoke | 2L / 2H / 128d | 1,461,760 | 1,064,960 | 396,800 | 5.6 MB |
| baseline | 4L / 4H / 256d | 5,322,240 | 2,162,688 | 3,159,552 | 20.3 MB |
| small (target) | 6L / 6H / 384d | 13,891,584 | 3,244,032 | 10,647,552 | 53.0 MB |

**Preview (2026-10-15):** to see real output, the smoke model was trained briefly
(`python -m src.train --config configs/smoke.yaml --max_steps 1000`: 12 s on CPU, ~45,000 tokens/s, val loss
9.0 → 3.88). `peek_model.py` after "One day, a little": untrained `_crunchy 0.0%` → trained **`_girl 58.8%`,
`_boy 16.6%`, `_bird 2.7%`**. Greedy generation repeats ("She loved to play with her mom." again and again);
top-p 0.9 is varied but not yet coherent. The real training runs are in Phase 8.

---

## Phase 6 — Causal attention

**Learn:** when predicting token *t*, the model must not see tokens after *t*, otherwise it would just copy
the answer. A **causal mask** hides the future:
```
          sees token
token   1  2  3  4
  1     ✓  ✗  ✗  ✗
  2     ✓  ✓  ✗  ✗
  3     ✓  ✓  ✓  ✗
  4     ✓  ✓  ✓  ✓
```
In code: attention scores for future positions are set to −∞ before softmax, so their weight becomes 0.

**Files:** `src/attention.py` (`CausalSelfAttention`), `tests/test_causal.py`

**Run:**
```
python -m pytest tests/test_causal.py -v
```

**Check:** all three tests pass. They prove that (1) attention weights above the diagonal are 0,
(2) changing future tokens never changes earlier outputs, and (3) the gradient from an early output to
any later input is exactly 0.

**Result (2026-10-15):** done.
- `tests/test_causal.py`: **3 passed**.
- Real-sentence demo (trained smoke model): "Lily has a red ball. She **throws it to her dog.**" vs
  "… She **eats a big green apple.**" — the predictions at all 8 shared positions (`<BOS>` … `_She`) are
  identical, although every later word differs.
- Mask removed on purpose (in memory only): **all 3 tests fail** ("position < 1 saw the future"), so the tests
  really detect a model that can see the future.

---

## Phase 7 — Choose the first model size

**Learn:** don't start big. Use small models to find bugs fast, then grow.

| Config file | Layers | Heads | Embedding | Context | Batch | Purpose |
|---|---|---|---|---|---|---|
| `configs/smoke.yaml` | 2 | 2 | 128 | 128 | 4 | find bugs in minutes |
| `configs/baseline.yaml` | 4 | 4 | 256 | 256 | 8 | first real model |
| `configs/small.yaml` | 6 | 6 | 384 | 256 | 8 | the target model from the plan |

All use vocabulary 8,192 and dropout 0.1 (0 for smoke). If memory runs short, use gradient accumulation
(`grad_accum_steps`): several small batches add up their gradients before one update.

**Files:** `configs/*.yaml`, `src/model.py` (`count_parameters`)

**Run:** measure speed and memory, and estimate full training time:
```
python scripts/benchmark.py --config configs/smoke.yaml
python scripts/benchmark.py --config configs/baseline.yaml
python scripts/benchmark.py --config configs/small.yaml
```
(results appended to `results/metrics/benchmark.csv`)

**Check:** you know how many parameters each model has (the target is in the ~10–20M range), how fast it
trains on this PC, and how long a full run takes.

**Result (2026-10-15):** CPU, fp32, 24 threads, real data, forward + backward + AdamW.

| Config | Parameters | Tokens/s | Memory | Tokens per update | Planned steps | Tokens per run | Est. run time | Passes over 100K-story data |
|---|---|---|---|---|---|---|---|---|
| smoke | 1.5 M | 45,780 | 410 MB | 512 | 200 | 0.1 M | < 1 min | 0.005 |
| baseline | 5.3 M | 11,613 | 878 MB | 8,192 | 5,000 | 41 M | **~1 h** | ~2 |
| small | 13.9 M | 4,982 | 660 MB | 16,384 | 10,000 | 164 M | **~9 h** | **~8** |

- Threads: 8 → 2,936, 12 → 3,735, 16 → 4,854, **24 → 4,982 tokens/s** (default, fastest).
- Memory is small (< 1 GB of 64 GB), so RAM is not a limit.
- **Data size matters for `small`:** 164M tokens is ~8 passes over the 100K-story set (20.8M tokens), which risks
  memorising. All 1,772,187 cleaned stories ≈ 365M tokens, so the same run would see each story less than once.

**Decision:** stay on CPU (the longest run, ~9 h, fits overnight). Training plan:
1. `baseline` on the current 100K stories (~1 h, ~2 passes)
2. tokenize all 1,772,187 cleaned stories (`prepare_data.py --n_train 0`)
3. `small` (target) on all stories, overnight (~9 h, less than 1 pass)

---

## Phase 8 — Train the model

**Learn:** one training step:
```
batch → model → next-token logits → cross-entropy loss → backpropagation → AdamW update
```
- **Cross-entropy loss:** how surprised the model is by the true next token (lower is better).
- **AdamW:** the optimizer that updates the weights.
- **Learning-rate schedule:** short warmup, then cosine decay down to `min_lr`.
- **Gradient clipping:** caps very large updates so training doesn't blow up.
- **Validation loss:** loss on stories the model never trains on. If train loss falls but validation
  loss rises, the model is memorising (overfitting).

**Files:** `src/train.py`, `scripts/plot_curves.py`

**Run (in this order):**
```
python -m src.train --config configs/smoke.yaml                 # 1. loss must go down
python -m src.train --config configs/smoke.yaml --resume checkpoints/smoke/model_last.pt --max_steps 300
                                                                # 2. resume works
python -m src.train --config configs/baseline.yaml              # 3. measure tokens/s, estimate time
python -m src.train --config configs/small.yaml                 # 4. target model, if time allows
python scripts/plot_curves.py --run baseline --run small
```
**GPU decision:** after step 3, use tokens/s to estimate how long `small.yaml` will take. If it's too long on
CPU, that's when to try the AMD GPU (WSL + ROCm).

**Result — step 1, baseline (2026-10-15):** done in **62.1 min** (estimate 59 min), peak memory 719 MB.

| Step | 250 | 500 | 1,000 | 2,000 | 3,000 | 4,000 | 5,000 |
|---|---|---|---|---|---|---|---|
| Val loss | 3.94 | 3.40 | 2.93 | 2.57 | 2.41 | 2.31 | **2.27** |
| Val perplexity | 51.2 | 29.8 | 18.8 | 13.1 | 11.1 | 10.1 | **9.71** |

- Validation loss fell at every check, and stayed just below train loss (dropout is on during training only),
  so **no overfitting**. Curve: `results/figures/loss_baseline.png`.
- After "One day, a little": `_girl 63.5%`, `_boy 29.6%`, `_bird 1.9%` (preview smoke model: girl 58.8%).
- Generated stories now have characters, dialogue and an ending (`<EOS>`). Greedy still loops
  ("The monster was scared and didn't know what to do." ×4); top-p 0.9 is varied but loses track of characters
  (the boy is both "Peter" and "Timmy").
- Checkpoints: `checkpoints/baseline/model_best.pt`, `model_last.pt`, `model_step_{1000..5000}.pt`.

**Result — resume test (2026-11-04):** done. The smoke model resumed from step 1,000: learning rate continued from
the schedule, loss 4.20 at step 1,010 (not back to ~9), val loss improved 3.884 → 3.872 by step 1,100.
Note: `elapsed_s` and `train_summary.json` restart on resume, so after a resumed run add up the time of each part.

**Result — full data (2026-11-04):** done. `python scripts/prepare_data.py --n_train 0 --data_dir data/full`
- 1,772,187 stories, 372,594,089 tokens, 1,455,445 windows of 256; `data/full/train.bin` is 745 MB.
- Every story has one `<BOS>` and one `<EOS>`; highest ID is 8,191; the first 100K stories match `data/train.bin`.
- Validation and test are identical to the files used for the baseline, so scores stay comparable.
- `configs/small.yaml` now reads `data/full`. A 164M-token run covers about 0.44 of this data, with no repeats needed.
- The 100K files in `data/` are kept, so the baseline can be reproduced.

**Result — target model `small` (2026-11-04):** done in **593.7 min (9.9 h)**, no interruptions. Peak memory is
about 1,438 MB, the highest `mem_mb` in `log.csv` (the 1,193 MB in `train_summary.json` is the old "in use at the
end" value).

| Step | 500 | 1,000 | 2,000 | 4,000 | 6,000 | 8,000 | 10,000 |
|---|---|---|---|---|---|---|---|
| Val loss | 3.16 | 2.53 | 2.19 | 1.97 | 1.87 | 1.80 | **1.77** |
| Val perplexity | 23.5 | 12.6 | 8.92 | 7.20 | 6.49 | 6.07 | **5.90** |

- Validation loss fell at every check and stayed just below train loss, so **no overfitting**.
  Curve: `results/figures/loss_baseline_small.png`.
- It was still improving slowly at the end (1.778 → 1.774 over the last 500 steps).
- After "there was a little": `_girl 75.3%`, `_boy 15.7%`, `_bird 1.4%`.
- Three top-p 0.9 stories all ended on their own with a moral. Sentences are mostly sound, but the plot still drifts
  (a book becomes a "secret recipe") and it can invent words ("pollow"). This is only a first look; Phases 9 and 10
  do the proper scoring.
- Checkpoints: `checkpoints/small/model_best.pt`, `model_last.pt`, `model_step_{1000..10000}.pt`.

**Check:** validation loss goes down. Outputs:
- `checkpoints/<run>/model_step_N.pt`, `model_best.pt`, `model_last.pt`
- `results/metrics/<run>/log.csv` (train/val loss, learning rate, tokens/s, memory, time)
- `results/figures/loss_*.png`

---

## Phase 9 — Perplexity

**Learn:** perplexity = e^(validation loss). Roughly: how many tokens the model is "choosing between" on
average. Loss 3.0 → perplexity ≈ 20.1. Lower is better, but **only compare models that use the same
tokenizer and the same validation data.**

**Files:** `src/evaluate.py`

**Run:**
```
python -m src.evaluate --ckpt checkpoints/small/model_best.pt
```

**Check:** it prints loss and perplexity on **test** (the final score) and on validation (for comparison with
the training log). Both are saved in `results/metrics/<run>/eval.json`.

**Result — baseline (2026-11-04):** test loss **2.2700**, perplexity **9.68**; validation loss 2.2705, perplexity 9.68.
Report: `results/metrics/baseline/eval_report.md`.

**Result — target model `small` (2026-11-06):** test loss **1.7473**, perplexity **5.74**; validation loss 1.7467,
perplexity 5.74. Report: `results/metrics/small/eval_report.md`.
- That is **41% lower perplexity than the baseline** (9.68 → 5.74), on the same tokenizer and the same test stories.
- Test and validation agree, so choosing the best checkpoint by validation did not overfit to it.
- The training log said 5.90 because it scored only part of the validation set. 5.74 uses all of it.

| Setting (40 stories) | Distinct-2 baseline → small | Flagged repetitive baseline → small | EOS rate baseline → small |
|---|---|---|---|
| greedy (8) | 0.391 → 0.586 | 100% → 75% | 75% → 88% |
| temp 0.7 | 0.655 → 0.693 | 30% → 25% | 88% → 92% |
| temp 1.0 | 0.799 → 0.777 | 0% → 2% | 98% → 100% |
| top-k 50 | 0.768 → 0.726 | 2% → 10% | 88% → 92% |
| top-p 0.90 | 0.738 → 0.738 | 2% → 10% | 92% → 92% |
| top-p 0.95 | 0.782 → 0.742 | 0% → 2% | 95% → 92% |

Real stories: Distinct-2 0.764, 5% flagged.
- **Greedy improved most:** it still loops, but much less (Rep-4 0.49 → 0.16).
- **Sampled stories flag a bit more often (top-p 0.90: 2% → 10%, i.e. 1 → 4 of 40).** I read the flagged
  stories. Some are real loops ("They did not hear the door open. They did not hear the door open."), and some are
  the "She wished… She wished…" pattern that real TinyStories also use. The baseline flagged less partly because
  its text was looser (made-up words, broken sentences), so the same phrases came back less often. With 40 stories this
  difference is small and noisy. Phase 10 hand-labelling will tell which it is.
- **Higher Distinct is not always better:** the baseline's temp 1.0 scores 0.799 partly from nonsense words
  ("newcess", "Shely"). Read the stories, not only the numbers.
- Some stories contain `â€œ`. This is broken quote characters copied from the training data, not an evaluation bug.
- Speed was **not reliable** at first (32% "other" CPU load). It was re-measured in Phase 12, where the load check
  itself turned out to be counting our own process.

**Evaluation method (fixed 2026-11-04, tests in `tests/test_evaluate.py`):**
- Repetition flag: Rep-4 rate above the 95th percentile of real validation stories (0.057), not a guessed number.
- Token limit: 512 (fits 98.0% of real validation stories; 200 fit only 67.6%). Hitting it is counted separately.
- Distinct-1/2: first 100 words of each story, in equal groups of one story per prompt, so settings are comparable.
- Speed: fixed 200 tokens per setting; marked not reliable if other programs use more than 15% CPU (own use
  subtracted, see Phase 12) or threads are limited.
  Re-measure on an idle PC with `python -m src.evaluate --ckpt <checkpoint> --perf_only`.
- Peak memory: the true highest value for the evaluation process.
- The `small` run started before this fix, so its `train_summary.json` memory is the old "in use at the end" value;
  use the highest `mem_mb` in its `log.csv` (sampled at each log line).

---

## Phase 10 — Text generation

**Learn:** generation repeats one step: predict probabilities → pick a token → append it → repeat, until
`<EOS>` or a length limit. How you *pick* the token is the decoding method:

| Method | How it picks |
|---|---|
| Greedy | always the most likely token (repetitive) |
| Temperature | samples; low temperature = safer, high = more random |
| Top-k | samples only from the k most likely tokens |
| Top-p (nucleus) | samples from the smallest set of tokens whose probabilities add up to p |

**Files:** `src/generate.py` (`select_next`, `generate`, `generate_text`)

**Run:**
```
python -m src.generate --ckpt checkpoints/small/model_best.pt --prompt "Once upon a time" --method greedy
python -m src.generate --ckpt checkpoints/small/model_best.pt --prompt "Once upon a time" --method top_p --top_p 0.9 --num_samples 3
python scripts/show_decoding.py --ckpt checkpoints/small/model_best.pt --prompt "Tom was sad because"   # learning aid
```

**Check:** the model writes a story that continues the prompt.

**Result — `small` (2026-11-06):** done. All 4 stories continue the prompt and end with `<EOS>` on their own
(164–198 tokens, ~135–150 tokens/s on CPU).
- **Greedy is deterministic:** it gives the same Lily-and-the-dog story on every run (and the same as in Phase 9).
  It is fluent but repeats "didn't know what to do" and "Don't worry, Lily", and the ending doesn't fit:
  "We can get you a new one" (a new dog?).
- **Top-p 0.9 gives a different story each time** (fixed seed 1337, so they can be reproduced). It has more variety
  but more slips: a book becomes a "secret recipe", the invented word "pollow", "Lily, come eat me!".
- `show_decoding.py` makes the loop visible. After "Tom was sad because", the model gives `' he'` 81.5%,
  `' his'` 13.2%, `' it'` 3.3%:
  - top-p 0.90 keeps only 2 tokens here, top-p 0.95 keeps 3, top-k 20 always keeps 20.
  - Temperature 0.7 sharpens the odds (`' he'` 81.5% → 92.1%).
  - Greedy always takes rank 1: "he had no friends to play with. He wanted to play".
  - Top-p sometimes takes a less likely token (`' fly'` at rank 9, 1.7%): "he wanted to fly high in the sky".
    Those occasional rare picks are where both the variety and the mistakes come from.

---

## Phase 11 — Sampling experiment

**Learn:** same model, same prompts, different decoding, to study the quality vs diversity trade-off.

| Method | Setting |
|---|---|
| Greedy | — |
| Temperature | 0.7, 1.0 |
| Top-k | 20, 50 |
| Top-p | 0.90, 0.95 |

Measures: **Distinct-1/2** (share of unique words / word pairs: higher = more diverse), **repetition rate**
(repeated 4-word phrases), **EOS rate** (how often the story actually finishes), and **latency** / tokens per second.

**Baseline result (2026-11-04, preliminary: 8 prompts):**

| Setting | Stories | Distinct-2 | Flagged repetitive | Hit 512-token limit |
|---|---|---|---|---|
| *real stories* | 10,995 | 0.764 | 5% by definition | — |
| greedy | 8 | 0.391 | 100% | 25% |
| temp 0.7 | 40 | 0.655 | 30% | 12% |
| temp 1.0 | 40 | 0.799 | 0% | 2% |
| top-k 20 | 40 | 0.739 | 5% | 10% |
| top-k 50 | 40 | 0.768 | 2% | 12% |
| top-p 0.90 | 40 | 0.738 | 2% | 8% |
| top-p 0.95 | 40 | 0.782 | 0% | 5% |

These measure variety and repetition only, not whether a story makes sense. Read the stories in Phase 11 too.

**Files:** `src/evaluate.py`, `configs/prompts.txt`

**Run:** included in `python -m src.evaluate --ckpt ...`

**Check:** a table in `results/metrics/<run>/eval_report.md` and all samples in
`results/generations/<run>/samples.jsonl`. Read the stories yourself too: numbers don't catch everything.

**Extra measure — made-up words:** `python scripts/madeup_words.py --run baseline --run small`
(writes `results/metrics/<run>/madeup_words.json`). This counts words that never appear in 300K training
stories (23,116 known words). Distinct-n counts "oripped" as variety; this measure catches it.

**Result — `small` (2026-11-06, still preliminary: 8 prompts, 40 stories per setting):**

| Setting | Distinct-2 (real 0.764) | Flagged repetitive (real 5%) | EOS rate | Made-up words (real 0.02%) |
|---|---|---|---|---|
| greedy (8) | 0.586 | 75% | 88% | 0% |
| temp 0.7 | 0.693 | 25% | 92% | 0.01% |
| temp 1.0 | **0.777** | **2%** | **100%** | **0.34%** (16× real) |
| top-k 20 | 0.733 | 8% | 95% | 0.05% |
| top-k 50 | 0.726 | 10% | 92% | 0% |
| top-p 0.90 | 0.738 | 10% | 92% | 0% |
| top-p 0.95 | 0.742 | **2%** | 92% | 0.04% |

What reading the stories (same prompt in every setting) showed:
- **Greedy / temp 0.7 = too safe.** They take the likeliest word, so they fall into loops ("You are not nice.
  You are not nice.", "He wanted to be nice." twice). A greedy story ran to
  443 tokens and lost the plot (the dog starts trying to make Tom happy).
- **Temp 1.0 = too wild.** It has the best numbers on variety, repetition and endings, but the most broken words
  ("neverld", "alreadypow", "oripped") and odd jumps ("His favourite place was the day", a cat that was never
  introduced). The baseline's temp 1.0 is the same (0.41%).
- **Top-k / top-p = the middle.** Real words and fewer loops. Mistakes are about meaning, not spelling: a bird "scared
  of the bird", Tom throws his "blue ball" when he only had a red one, "Sam ran to catch it, but he was too fast".
- **Every setting keeps one story going and ends it with a lesson.** None of them keeps the facts consistent for a
  whole story. That is the model's limit at 13.9M parameters, not a decoding problem.

**Prompt issue found:** `In a small house near the forest` is followed by `.` 83.5% of the time (24 of 31 stories),
so the story starts with no character ("…the forest. Every day, he would…"). It's the prompt, not the model:
it reads like the end of "They lived in a small house near the forest." Kept unchanged for now so the
baseline and `small` stay comparable. A better prompt would be "In a small house near the forest, there lived".

**Choice:** **top-p 0.95** is the default for writing stories. It has the lowest repetition (tied with temp 1.0),
variety close to real stories, few made-up words, and no single weak point. Top-p 0.90 is a close second. The gap
between them (1 vs 4 flagged stories out of 40) is within noise for 8 prompts. Avoid greedy for stories.

---

## Phase 12 — Evaluate the model

**Learn:** don't report only perplexity. Report three groups:
- **Quality:** validation loss, perplexity
- **Generation:** coherence, repetition, diversity, story completion
- **Performance:** parameters, model size (MB), memory (RAM on CPU), tokens/s, latency, training time

Also record the exact device, PyTorch version, precision, batch size and effective batch size.

**Files:** `src/evaluate.py`, `results/metrics/<run>/run_info.json`, `train_summary.json`

**Check:** `eval_report.md` has all three groups.

**Fixes made in this phase (2026-11-06, tests in `tests/test_evaluate.py`, 15 pass):**
- **The "PC busy" check counted our own work.** In the first second after loading the model, our own process used
  18% CPU, so an idle PC showed "26% other load" and every speed result was marked not reliable. Now the evaluation
  subtracts its own CPU use and checks other programs both just before and during the timing. It needs both,
  because while we fill every core, other programs get squeezed out and look idle.
- **The report had no training details.** It now has a `## Training` section, read from `run_info.json`,
  `train_summary.json` and `log.csv`: hardware, precision, batch, effective batch, steps, tokens, time and peak
  memory. It also gives seconds per 200-token story (the latency).

**Result — final comparison (2026-11-06).** Speed measured on an idle PC: other programs used 12% (baseline) and
15% (`small`) of the CPU. The `small` run is right at the limit, but it matches the earlier run within 3%.

| | Baseline | `small` (target) |
|---|---|---|
| **Quality** | | |
| Test loss / perplexity | 2.2700 / 9.68 | **1.7473 / 5.74** |
| **Generation** (top-p 0.95, 40 stories) | | |
| Distinct-2 (real 0.764) | 0.782 | 0.742 |
| Flagged repetitive (real 5%) | 0% | 2% |
| Finished with `<EOS>` | 95% | 92% |
| Made-up words (real 0.02%) | 0.07% | 0.04% |
| Reading the stories | loose, broken sentences | fluent, plot drifts |
| **Performance** | | |
| Parameters / size (fp32) | 5.32M / 20.3 MB | 13.89M / 53.0 MB |
| Generation speed (top-p 0.95) | 266 tokens/s | 141 tokens/s |
| Time for a 200-token story | 0.75 s | 1.42 s |
| Evaluation peak memory | 781 MB | 941 MB |
| **Training** | | |
| Effective batch | 32 seq (8,192 tokens) | 64 seq (16,384 tokens) |
| Steps / tokens seen | 5,000 / 41M | 10,000 / 164M |
| Training time | 1.0 h | 9.9 h |
| Training peak memory | 989 MB | 1,438 MB |

Setup for both: Intel CPU, 24 logical CPUs, 63.7 GB RAM, 24 threads, PyTorch 2.14.1+cpu, fp32, no GPU.
The baseline's training peak is 989 MB from `log.csv`. The 719 MB recorded in Phase 8 was the old "in use at
the end" value.

**Trade-off:** `small` is about 2.6× bigger and about 1.9× slower at writing, and it cost 10× the training time.
In return it has 41% lower perplexity and stories that read like stories. At about 1.4 s per story on a CPU, it
is fast enough to be a teacher.

---

## Phase 13 — Error analysis

**Learn:** read 50–100 generated stories and count the kinds of mistakes:

| Error | Example |
|---|---|
| Repetition | "the boy ran and ran and ran…" |
| Contradiction | character is suddenly in another place |
| Incomplete story | stops without a real ending |
| Grammar | broken sentence |
| Entity consistency | a character's name changes |

Repetition and incomplete stories are flagged automatically. The other three you mark by hand.

**Files:** `results/generations/<run>/error_analysis.csv` (made by `src/evaluate.py`)

**Run:** open the CSV, fill in the `contradiction`, `grammar` and `entity_consistency` columns, then count.

**Check:** you can report how often each error type happens (for example "repetition: 18%").

**Result — `small`, 60 stories, top-p 0.90 (2026-11-06):** labels and a note per story are in
`results/generations/small/error_analysis.csv`. They are also kept in `scripts/error_labels_small.py`, which
re-applies them and prints the counts: `python scripts/error_labels_small.py`.

Rules used (the same for every story):
- **contradiction:** something clashes with an earlier fact in the story or doesn't follow from it.
- **grammar:** at least one broken sentence or invented word.
- **entity consistency:** a character changes name, gender or species, or the wrong character speaks or acts.

These rules are **strict**: one slip anywhere in a story counts.

| Error | Stories | Typical example |
|---|---|---|
| Contradiction | **53/60 (88%)** | "Lily was just a toy"; dad is "at work" and "coming home" in the same moment |
| Entity consistency | **33/60 (55%)** | "a little mouse named Tim. Tim was a very obedient cat"; "a new boy… His name was Lily" |
| Grammar | 22/60 (37%) | "He liked to be park tomorrow"; "a big grapes"; "pandasplum" |
| Repetition (auto) | 13/60 (22%) | "He liked his mom. He liked his mom." |
| Incomplete (auto) | 4/60 (7%) | all 4 were cut at the 512-token limit, not bad endings |

- **No error of any kind:** 5/60 (ids 0, 6, 12, 40, 54). Error types per story: none 5, one 14, two 29, three 12.
- **The main weakness is meaning, not language.** Most sentences are fine on their own, and only 37% have a
  grammar slip. But the model can't keep track of *who* is who and *what* is where for a whole story.
- Common patterns:
  - **The same word for two characters**: "The cat was happy to see the cat".
  - **Who owns which object flips**: Tom's ball becomes Sue's.
  - **Objects change into other objects**: key → penny → coin, bee → butterfly.
  - **Wrong speaker**: Lily says "My name is Max".
  - **Characters who appear from nowhere**: "Ben and Mia" when Mia was never introduced.
- **Answer to the Phase 9 question (more repetition flags than the baseline):** of 13 flagged stories, **11 are
  real loops** ("I'm sorry, dad." ×3; a stream of "You are welcome… you are my friend") and 2 are false alarms.
  One is a short story. The other is the name confusion above (every animal is "the cat"). The flag also missed
  3 mild loops ("They could not breathe." ×2, "She wished…" ×3). So the flag is mostly right, and the extra
  repetition is real.
- **The 4 "incomplete" stories all hit the 512-token limit**, and those 4 are also the longest and most repetitive.
  Long stories are where the model loses its way.
- **The prompt problem from Phase 11 shows here too:** all 7 "In a small house near the forest" stories go on
  with ".", so they start with no proper opening ("…the forest. As she walked…").
- **Caveats:** I labelled these once, alone (no second labeller), so treat the numbers as ±10% and spot-check a
  few rows yourself. The baseline's 60 stories are not labelled, so the two models can't be compared here yet.

---

## Phase 14 — Package as teacher for Project 2

**Learn:** Project 2 trains a smaller *student* model to copy this *teacher*. To compare them fairly later,
everything about the teacher must be saved and reproducible.

**Save:** best checkpoint, model config, training config, tokenizer, `requirements.lock.txt`, hardware report,
random seeds, evaluation prompts, evaluation scripts, and a `README.md` with results.
Back up the final files outside the project folder.

**Check:** someone else could load the teacher and reproduce your perplexity from these files alone.

**Files:** `scripts/package_teacher.py` (builds the package), `scripts/teacher_files/` (`README.md`,
`load_teacher.py` and `verify_teacher.py`, which go into the package)

**Run:**
```
python scripts/package_teacher.py          # rebuilds teacher/ from the project files
python teacher/verify_teacher.py           # ~3 min; --quick ~15 s
```

**Result (2026-11-06):** done. `teacher/` has 36 files, 63 MB:
- `teacher.pt` is **53 MB**: the training checkpoint was 159 MB, but the optimizer state is only needed to resume
  training, so it is left out.
- Tokenizer (SHA-256 `158ade55…c0f530`), `configs/small.yaml`, `model_config.json`, prompts.
- `src/`: the model is plain PyTorch, so its class is needed to load it.
- `data/val.bin` and `test.bin`: the 745 MB `train.bin` isn't needed to reproduce perplexity.
- Requirements lock, hardware report, results, `expected.json`, `README.md`, and `MANIFEST.sha256` with the
  SHA-256 of every file.
- `load_teacher.py` gives `load_teacher()` and `next_token_probs()` for Project 2.

**Check passed from outside the project:** the package was copied to
`C:\Users\home\Backups\project1_teacher_2026-11-06\` and `verify_teacher.py` was run there, with Python in
isolated mode (`-I`) and from another folder, so the project files couldn't be used. The results:
- every file matches the manifest
- 13,891,584 parameters
- **test loss 1.7473 / perplexity 5.74 and val 1.7467 / 5.74, exactly as in Phase 9**
- the story ends by itself

The README examples (`next_token_probs` → `' he'` 81.5%, `src.generate --ckpt teacher.pt`) also work from the copy.

**Backup:** `C:\Users\home\Backups\project1_teacher_2026-11-06\` (65 MB) holds the teacher package plus
`project_snapshot/` (configs, scripts, src, tests, results, docs, INSTRUCTIONS.md, README.md). It is
**not** backed up there: `data/` (it can be rebuilt with Phases 2–4) and `checkpoints/` (only needed to resume training).

⚠️ C: and D: are on the **same physical disk**, so this backup doesn't protect against that disk failing.
Copy the backup folder to the USB drive (F:) or to cloud storage as well.

Also added: `README.md` at the project root (summary, layout, how to run).

---

## Reference

**Folder layout**
```
configs/      smoke.yaml, baseline.yaml, small.yaml, prompts.txt
scripts/      env_report.py, inspect_data.py, clean_data.py, train_tokenizer.py, prepare_data.py,
              benchmark.py, peek_model.py, plot_curves.py, show_decoding.py, madeup_words.py,
              error_labels_small.py, package_teacher.py, teacher_files/
src/          tokenizer.py, dataset.py, attention.py, transformer.py, model.py,
              train.py, generate.py, evaluate.py, utils.py
tests/        test_causal.py, test_pipeline.py, test_evaluate.py
teacher/      self-contained teacher package for Project 2 (Phase 14, rebuilt by package_teacher.py)
data/         hf_cache/, clean/ (Phase 2); tokenizer.json, train.bin, val.bin, test.bin, stats.json (Phases 3–4);
              full/ (same files with all stories, Phase 8)
checkpoints/  <run>/model_step_N.pt, model_best.pt, model_last.pt (created in Phase 8)
results/      metrics/, generations/, figures/                    (created in Phases 8–13)
```

**Rules**
- Run everything from the project folder with `.venv` active.
- Seeds are fixed (`seed` in each config) so runs are repeatable.
- On Windows keep `num_workers: 0` in the configs.
- Plain PyTorch only for the model (no `transformers` model classes).
- Never compare perplexities across different tokenizers.
