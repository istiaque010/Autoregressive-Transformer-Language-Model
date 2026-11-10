"""Check that this teacher package is complete and reproduces its scores, using only files in this folder.

1. every file matches MANIFEST.sha256 (nothing missing or changed)
2. the model loads and has the expected number of parameters
3. test (and validation) perplexity match expected.json
4. it writes a story with the recommended decoding

Usage (from anywhere): python verify_teacher.py [--quick]
--quick scores the first 100 test batches only (about 15 s instead of about 3 minutes) and checks loosely.
Exit code 0 = all checks passed.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_teacher import load_teacher  # noqa: E402
from src.dataset import NextTokenDataset, load_tokens  # noqa: E402
from src.generate import generate_text  # noqa: E402
from src.model import count_parameters  # noqa: E402
from src.train import evaluate_loss  # noqa: E402
from src.utils import set_seed  # noqa: E402


def check_manifest():
    bad = []
    with open(os.path.join(HERE, "MANIFEST.sha256"), encoding="utf-8") as f:
        for line in f:
            digest, rel = line.rstrip("\n").split("  ", 1)
            p = os.path.join(HERE, rel)
            if not os.path.exists(p):
                bad.append(f"missing {rel}")
                continue
            h = hashlib.sha256()
            with open(p, "rb") as g:
                for chunk in iter(lambda: g.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != digest:
                bad.append(f"changed {rel}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    with open(os.path.join(HERE, "expected.json"), encoding="utf-8") as f:
        exp = json.load(f)
    ok = True

    bad = check_manifest()
    print(f"[1] manifest: {'OK' if not bad else 'FAILED ' + ', '.join(bad)}")
    ok &= not bad

    torch.set_num_threads(os.cpu_count() or 1)
    model, tok, info = load_teacher()
    n = count_parameters(model)["total"]
    print(f"[2] model: {n:,} parameters, step {info['step']}, vocab {tok.vocab_size} "
          f"{'OK' if n == exp['params'] else 'FAILED (expected %d)' % exp['params']}")
    ok &= n == exp["params"]

    max_batches = 100 if args.quick else None
    tol = 0.05 if args.quick else 1e-3  # full run must match to 3 decimals of loss
    for split in ("test",) if args.quick else ("test", "val"):
        ds = NextTokenDataset(load_tokens(os.path.join(HERE, "data"), split), model.cfg.block_size)
        loss = evaluate_loss(model, DataLoader(ds, batch_size=exp["eval_batch_size"], shuffle=False), "cpu",
                             exp["precision"], max_batches)
        want = exp[f"{split}_loss"]
        good = abs(loss - want) <= tol
        print(f"[3] {split}: loss {loss:.4f}, perplexity {math.exp(loss):.2f} "
              f"(expected {want:.4f} / {math.exp(want):.2f}{', quick: first 100 batches' if args.quick else ''}) "
              f"{'OK' if good else 'FAILED'}")
        ok &= good

    set_seed(1337)
    d = exp["recommended_decoding"]
    out = generate_text(model, tok, "Once upon a time", max_new_tokens=200, method=d["method"],
                        top_p=d["top_p"], temperature=d["temperature"])
    print(f"[4] story ({out['new_tokens']} tokens, ended by itself: {out['ended_with_eos']}):\n    {out['text']}")
    ok &= out["new_tokens"] > 0

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
