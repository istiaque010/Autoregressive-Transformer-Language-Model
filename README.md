# A Small Autoregressive Transformer Language Model Built from Scratch for Short Story Generation

### A Small Language Model Built from Scratch for Children's Story Generation

*Project 1*

A GPT-style decoder-only Transformer written in plain PyTorch, with a byte-level BPE tokenizer and a
hand-written causal attention. It was trained on TinyStories on a CPU and evaluated for perplexity, sampling and
errors. It ends as the **teacher** for Project 2 (distillation). The step-by-step guide with every result is in
[`INSTRUCTIONS.md`](INSTRUCTIONS.md).

## Results

| | Baseline | Target (`small`, the teacher) |
|---|---|---|
| Parameters | 5.32M | 13.89M |
| Training data | 100K stories | all 1.77M stories (164M tokens seen) |
| Training time (CPU) | 1.0 h | 9.9 h |
| **Test perplexity** | 9.68 | **5.74** |
| Story speed, top-p 0.95 | 0.75 s / 200 tokens | 1.42 s / 200 tokens |

- **Decoding:** top-p 0.95 is the best of the 7 settings tested. Greedy loops, and temperature 1.0 makes up words.
- **Errors, 60 hand-labelled stories:** contradictions 88%, character mix-ups 55%, grammar 37%, repetition 22%.
  Sentences are fluent, but facts and characters drift over a whole story.

## Layout

```
configs/      smoke.yaml, baseline.yaml, small.yaml, prompts.txt
src/          tokenizer, dataset, attention, transformer, model, train, generate, evaluate, utils
scripts/      data prep, tokenizer, benchmark, plots, inspection, made-up words, error labels, packaging
tests/        causal mask, pipeline, evaluation metrics (15 tests)
results/      metrics/, generations/, figures/
teacher/      self-contained teacher package for Project 2 (see teacher/README.md)
```

## Run

```
python -m pytest -q tests
python -m src.train --config configs/small.yaml
python -m src.evaluate --ckpt checkpoints/small/model_best.pt
python scripts/package_teacher.py && python teacher/verify_teacher.py
```

Environment: Windows 11, Python 3.12.10, `torch==2.14.1+cpu`. The full list is in `requirements.lock.txt`.
