"""Statistics and figures for the article
"Decoding Strategies and Error Analysis for Small Autoregressive Transformer Language Models in Story Generation".

Every number in main.tex comes from the project's saved results or from this script (written to
analysis/paper_stats.json). Nothing is trained here; the two checkpoints are only used for scoring.

Run from the project root:
    python project1_paper/analysis/paper_analysis.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from src.evaluate import repeated_ngram_rate, words  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.tokenizer import StoryTokenizer  # noqa: E402

OUT = os.path.join(ROOT, "project1_paper")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
RUNS = ["baseline", "small"]
SETTINGS = ["greedy", "temp_0.7", "temp_1.0", "top_k_20", "top_k_50", "top_p_0.90", "top_p_0.95"]
LABEL = {"greedy": "Greedy", "temp_0.7": "T=0.7", "temp_1.0": "T=1.0", "top_k_20": "k=20", "top_k_50": "k=50",
         "top_p_0.90": "p=0.90", "top_p_0.95": "p=0.95"}
RNG = np.random.default_rng(1337)
B = 5000  # bootstrap resamples
torch.set_num_threads(os.cpu_count() or 1)

# Validated categorical palette (light mode): slot 1 blue, slot 2 orange, slot 3 aqua, slot 4 yellow, slot 5 magenta.
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#1f2328", "#5d6470", "#e1e0d9"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
COL = 3.5  # IEEE single-column width (inches)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- statistics helpers
def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_two_sided(a, b, c, d):
    """Fisher exact test for the 2x2 table [[a, b], [c, d]] (two-sided, sum of tables no more likely)."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d

    def p(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)

    p0 = p(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9)))


def boot_mean(x):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff_p(x, y):
    """Two-sided bootstrap p-value for mean(x) - mean(y) = 0 (pooled-null resampling)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = x.mean() - y.mean()
    pool = np.concatenate([x, y])
    ix = RNG.integers(0, len(pool), (B, len(x)))
    iy = RNG.integers(0, len(pool), (B, len(y)))
    d = pool[ix].mean(1) - pool[iy].mean(1)
    return float((np.abs(d) >= abs(obs)).mean())


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    r = float(np.corrcoef(rx, ry)[0, 1])
    # permutation p-value
    perm = np.array([np.corrcoef(rx, RNG.permutation(ry))[0, 1] for _ in range(B)])
    return r, float((np.abs(perm) >= abs(r)).mean())


# ---------------------------------------------------------------- load results
ev = {r: json.load(open(f"results/metrics/{r}/eval.json")) for r in RUNS}
mw = {r: json.load(open(f"results/metrics/{r}/madeup_words.json")) for r in RUNS}
samples = {r: [json.loads(l) for l in open(f"results/generations/{r}/samples.jsonl", encoding="utf-8")] for r in RUNS}
thr = ev["small"]["reference_real_stories"]["rep4_flag_threshold"]
stats = {"rep4_threshold": thr, "bootstrap_resamples": B}

# known vocabulary for made-up words (same definition as scripts/madeup_words.py)
WORD = re.compile(r"[a-z]+")
known = set()
for t in pd.read_parquet("data/clean/train.parquet")["text"][:300000]:
    known.update(WORD.findall(t.lower()))
stats["known_words"] = len(known)

# ---------------------------------------------------------------- per-story decoding statistics
per = {}
for r in RUNS:
    per[r] = {}
    for s in SETTINGS:
        rows = [x for x in samples[r] if x["setting"] == s]
        rep = [repeated_ngram_rate(x["completion"]) for x in rows]
        mw_bad, mw_tot = 0, 0
        for x in rows:
            ws = WORD.findall(x["completion"].lower())
            mw_tot += len(ws)
            mw_bad += sum(w not in known for w in ws)
        per[r][s] = {
            "n": len(rows),
            "rep4": rep,
            "flag": [float(v > thr) for v in rep],
            "eos": [float(x["ended_with_eos"]) for x in rows],
            "len": [x["new_tokens"] for x in rows],
            "madeup": (mw_bad, mw_tot),
        }

dec = {}
for r in RUNS:
    dec[r] = {}
    for s in SETTINGS:
        p = per[r][s]
        n = p["n"]
        fk = int(sum(p["flag"]))
        ek = int(sum(p["eos"]))
        dec[r][s] = {
            "n": n,
            "distinct_1": ev[r]["sampling"][s]["distinct_1"], "distinct_1_std": ev[r]["sampling"][s]["distinct_1_std"],
            "distinct_2": ev[r]["sampling"][s]["distinct_2"], "distinct_2_std": ev[r]["sampling"][s]["distinct_2_std"],
            "rep4_mean_ci": boot_mean(p["rep4"]),
            "flag_k": fk, "flag_rate": fk / n, "flag_wilson": wilson(fk, n),
            "eos_k": ek, "eos_rate": ek / n, "eos_wilson": wilson(ek, n),
            "len_mean_ci": boot_mean(p["len"]),
            "madeup_bad": p["madeup"][0], "madeup_words": p["madeup"][1],
            "madeup_rate": p["madeup"][0] / max(p["madeup"][1], 1),
            "madeup_wilson": wilson(*p["madeup"]),
            "tokens_per_s": ev[r]["speed"]["tokens_per_s"][s],
            "ms_per_token": ev[r]["speed"]["ms_per_token"][s],
        }
stats["decoding"] = dec

# model-size effect at equal decoding: Fisher test on flag rate, bootstrap on Rep-4
stats["size_effect"] = {}
for s in SETTINGS:
    if s == "greedy":
        continue
    a, c = dec["small"][s]["flag_k"], dec["baseline"][s]["flag_k"]
    n1, n2 = dec["small"][s]["n"], dec["baseline"][s]["n"]
    stats["size_effect"][s] = {
        "flag_small": a, "flag_base": c, "fisher_p": fisher_two_sided(a, n1 - a, c, n2 - c),
        "rep4_p": boot_diff_p(per["small"][s]["rep4"], per["baseline"][s]["rep4"]),
    }
# within the small model: each setting vs top-p 0.95 (flag rate, made-up words)
stats["vs_top_p_095"] = {}
ref = dec["small"]["top_p_0.95"]
for s in SETTINGS:
    if s in ("greedy", "top_p_0.95"):
        continue
    d = dec["small"][s]
    stats["vs_top_p_095"][s] = {
        "flag_fisher_p": fisher_two_sided(d["flag_k"], d["n"] - d["flag_k"], ref["flag_k"], ref["n"] - ref["flag_k"]),
        "madeup_fisher_p": fisher_two_sided(d["madeup_bad"], d["madeup_words"] - d["madeup_bad"],
                                            ref["madeup_bad"], ref["madeup_words"] - ref["madeup_bad"]),
    }
# made-up words, temperature 1.0 vs real stories (pooled over both models)
real_texts = pd.read_parquet("data/clean/validation.parquet")["text"][:2000]
rb = rt = 0
for t in real_texts:
    ws = WORD.findall(t.lower())
    rt += len(ws)
    rb += sum(w not in known for w in ws)
stats["real_madeup"] = {"bad": rb, "words": rt, "rate": rb / rt, "wilson": wilson(rb, rt)}

# ---------------------------------------------------------------- likelihood of generated text, nucleus size
tok = StoryTokenizer("data/tokenizer.json")
models = {r: load_checkpoint(f"checkpoints/{r}/model_best.pt")[0].eval() for r in RUNS}


@torch.no_grad()
def nll_of(model, prompt, completion):
    """Mean negative log-likelihood (nats/token) of the completion tokens given <BOS> + prompt."""
    p_ids = tok.encode(prompt, add_bos=True)
    full = tok.encode(prompt + completion, add_bos=True)[: model.cfg.block_size]
    if len(full) <= len(p_ids):
        return None
    x = torch.tensor([full[:-1]])
    y = torch.tensor(full[1:])
    logits, _ = model(x)
    lp = torch.log_softmax(logits[0], -1)
    tgt = lp[torch.arange(len(y)), y]
    return float(-tgt[len(p_ids) - 1:].mean())


test = pd.read_parquet("data/clean/test.parquet")["text"].sample(200, random_state=1337).tolist()
stats["self_nll"] = {}
for r in RUNS:
    m = models[r]
    out = {}
    for s in SETTINGS:
        v = [nll_of(m, x["prompt"], x["completion"]) for x in samples[r] if x["setting"] == s]
        v = [a for a in v if a is not None]
        out[s] = boot_mean(v)
    # real test stories: prompt = first 5 words, completion = the rest (same conditioning as the generations)
    v = []
    for t in test:
        w = t.split(" ")
        if len(w) > 8:
            v.append(nll_of(m, " ".join(w[:5]), " " + " ".join(w[5:])))
    out["real"] = boot_mean([a for a in v if a is not None])
    stats["self_nll"][r] = out


@torch.no_grad()
def nucleus_sizes(model, texts, ps=(0.90, 0.95)):
    sizes = {p: [] for p in ps}
    topk_mass = {20: [], 50: []}
    for t in texts:
        ids = tok.encode(t, add_bos=True, add_eos=True)[: model.cfg.block_size]
        logits, _ = model(torch.tensor([ids[:-1]]))
        pr = torch.softmax(logits[0], -1)
        sp, _ = torch.sort(pr, -1, descending=True)
        cum = sp.cumsum(-1)
        for p in ps:
            sizes[p] += ((cum - sp) < p).sum(-1).tolist()  # smallest prefix whose mass reaches p
        for k in topk_mass:
            topk_mass[k] += cum[:, k - 1].tolist()
    return sizes, topk_mass


ns, tkm = nucleus_sizes(models["small"], test[:100])
stats["nucleus_small"] = {
    str(p): {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
             "frac_le_1": float(np.mean(np.array(v) <= 1)), "frac_gt_20": float(np.mean(np.array(v) > 20)),
             "frac_gt_50": float(np.mean(np.array(v) > 50)), "n_positions": len(v)} for p, v in ns.items()}
stats["topk_mass_small"] = {str(k): {"median": float(np.median(v)), "frac_below_0.9": float(np.mean(np.array(v) < 0.9))}
                            for k, v in tkm.items()}

# ---------------------------------------------------------------- error analysis (small, 60 stories, top-p 0.90)
rows = list(csv.DictReader(open("results/generations/small/error_analysis.csv", encoding="utf-8")))
n = len(rows)
yes = lambda v: str(v).strip().lower() in ("yes", "true", "1")  # noqa: E731
types = {"contradiction": "contradiction", "entity": "entity_consistency", "grammar": "grammar",
         "repetition (auto)": "repetition", "incomplete (auto)": "incomplete"}
err = {}
for name, col in types.items():
    k = sum(yes(r[col]) for r in rows)
    err[name] = {"k": k, "n": n, "rate": k / n, "wilson": wilson(k, n)}
hand = ["contradiction", "entity_consistency", "grammar"]
counts = [sum(yes(r[c]) for c in hand) for r in rows]
co = {a: {b: sum(yes(r[a]) and yes(r[b]) for r in rows) for b in hand} for a in hand}
flagged = [r for r in rows if yes(r["repetition"])]
real_loops = sum(("rep: real" in r["notes"]) for r in flagged)
false_alarm = sum(("false alarm" in r["notes"]) for r in flagged)
missed = sum(("rep: missed" in r["notes"]) for r in rows)
words_len = [len(words(r["text"])) for r in rows]
rho, rho_p = spearman(words_len, counts)
any_err = sum(c > 0 for c in counts)
by_prompt = {}
for r in rows:
    d = by_prompt.setdefault(r["prompt"], [0, 0])
    d[0] += sum(yes(r[c]) for c in hand)
    d[1] += 1
long_ = [c for c, w in zip(counts, words_len) if w >= np.median(words_len)]
short = [c for c, w in zip(counts, words_len) if w < np.median(words_len)]
stats["errors_small"] = {
    "n": n, "types": err, "error_count_hist": {str(i): counts.count(i) for i in range(4)},
    "any_error": any_err, "clean_ids": [int(r["id"]) for r, c in zip(rows, counts) if c == 0],
    "cooccurrence": co,
    "rep_flag": {"flagged": len(flagged), "real": real_loops, "false_alarm": false_alarm, "missed": missed,
                 "precision": real_loops / len(flagged), "recall": real_loops / (real_loops + missed)},
    "length_vs_errors_spearman": {"rho": rho, "p_perm": rho_p},
    "median_words": float(np.median(words_len)),
    "mean_errors_long": float(np.mean(long_)), "mean_errors_short": float(np.mean(short)),
    "long_vs_short_p": boot_diff_p(long_, short),
    "by_prompt": {k: {"errors": v[0], "stories": v[1], "per_story": v[0] / v[1]} for k, v in by_prompt.items()},
}
brows = list(csv.DictReader(open("results/generations/baseline/error_analysis.csv", encoding="utf-8")))
stats["errors_baseline_auto"] = {"n": len(brows), "repetition": sum(yes(r["repetition"]) for r in brows),
                                 "incomplete": sum(yes(r["incomplete"]) for r in brows)}
a, c = sum(yes(r["repetition"]) for r in rows), stats["errors_baseline_auto"]["repetition"]
stats["errors_rep_size_fisher_p"] = fisher_two_sided(a, n - a, c, len(brows) - c)

# ---------------------------------------------------------------- training curves
curves = {}
for r in RUNS:
    df = pd.read_csv(f"results/metrics/{r}/log.csv")
    curves[r] = df
tok_per_step = {"baseline": 8192, "small": 16384}

json.dump(stats, open(os.path.join(OUT, "analysis", "paper_stats.json"), "w"), indent=1)

# ================================================================= FIGURES
# Fig. 1: architecture and pipeline
fig, ax = plt.subplots(figsize=(COL, 3.15))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10.6)
ax.axis("off")


def box(x, y, w, h, text, fc="#ffffff", ec="#c3c2b7", bold=False, fs=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED))


box(0.35, 9.55, 4.7, 0.75, r"$\langle$BOS$\rangle\, x_1 \ldots x_{t-1}$ (BPE, $|V|{=}8192$)", fs=6.5)
arrow(2.7, 9.55, 2.7, 9.15)
box(0.35, 8.35, 4.7, 0.8, "Token emb. + position emb.\n" + r"$\mathbf{h}^{(0)}_i = \mathbf{E}[x_i] + \mathbf{P}[i]$")
arrow(2.7, 8.35, 2.7, 7.95)
ax.add_patch(FancyBboxPatch((0.35, 3.05), 4.7, 4.85, boxstyle="round,pad=0.02,rounding_size=0.25", fc="#f3f7fd",
                            ec=C1, lw=0.9, ls="--"))
ax.text(4.95, 7.62, r"$\times L$", ha="right", va="center", fontsize=8, color=C1, fontweight="bold")
box(0.75, 6.85, 3.9, 0.6, "LayerNorm")
arrow(2.7, 6.85, 2.7, 6.55)
box(0.55, 5.75, 4.3, 0.8, "Causal multi-head\nself-attention ($H$ heads)", fc="#e6eefc", ec=C1)
arrow(2.7, 5.75, 2.7, 5.4)
ax.text(2.85, 5.2, r"$\oplus$ residual", fontsize=6.5, color=MUTED, va="center")
box(0.75, 4.35, 3.9, 0.6, "LayerNorm")
arrow(2.7, 4.35, 2.7, 4.05)
box(0.55, 3.25, 4.3, 0.8, "MLP  $d \\rightarrow 4d \\rightarrow d$\n(GELU)", fc="#fdeee6", ec=C2)
arrow(2.7, 3.05, 2.7, 2.65)
box(0.35, 1.85, 4.7, 0.8, "LayerNorm, tied output head\n" + r"$\mathbf{z}_t = \mathbf{E}\,\mathrm{LN}(\mathbf{h}^{(L)}_t)$")
arrow(2.7, 1.85, 2.7, 1.45)
box(0.35, 0.6, 4.7, 0.85, r"$p_\theta(x_t \mid x_{<t}) = \mathrm{softmax}(\mathbf{z}_t / T)$")
# right column (read bottom-up): perplexity, decoding, metrics, annotation
box(5.6, 0.6, 4.1, 1.4, "Perplexity on held-out\ntest set (10,995 stories)", fs=6.5)
box(5.6, 2.7, 4.1, 1.9, "Decoding rule $\\mathcal{D}$\ngreedy | temperature $T$\ntop-$k$ | top-$p$ (nucleus)", fc="#eaf7f1", ec=C3, fs=6.5)
box(5.6, 5.3, 4.1, 2.2, "Automatic metrics\nDistinct-$n$, Rep-4 flag,\nEOS rate, made-up words,\nself-NLL, tokens/s", fs=6.5)
box(5.6, 8.2, 4.1, 2.1, "Error annotation\ncontradiction, entity,\ngrammar (+ auto flags)", fc="#fdf4dc", ec=C4, fs=6.5)
ax.annotate("", xy=(5.6, 1.3), xytext=(4.8, 1.0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED))
ax.annotate("", xy=(5.6, 3.6), xytext=(4.8, 1.25), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED))
arrow(7.65, 4.6, 7.65, 5.3)
arrow(7.65, 7.5, 7.65, 8.2)
save(fig, "fig_architecture")

# Fig. 2: learning curves vs tokens seen
fig, ax = plt.subplots(figsize=(COL, 2.2))
for r, col, lab in (("baseline", C2, "Baseline (5.3M)"), ("small", C1, "Small (13.9M)")):
    df = curves[r]
    t = df["step"] * tok_per_step[r] / 1e6
    tr = df.dropna(subset=["train_loss"])
    ax.plot(tr["step"] * tok_per_step[r] / 1e6, tr["train_loss"], color=col, lw=0.8, alpha=0.35)
    va = df.dropna(subset=["val_loss"])
    ax.plot(va["step"] * tok_per_step[r] / 1e6, va["val_loss"], color=col, lw=1.6, marker="o", ms=2.5, label=lab)
ax.axhline(math.log(8192), color=MUTED, lw=0.6, ls="--")
ax.text(160, math.log(8192) - 0.35, r"uniform guess $\ln|V| = 9.01$", ha="right", fontsize=6.5, color=MUTED)
ax.set_xlabel("Training tokens seen (millions)")
ax.set_ylabel("Cross-entropy (nats/token)")
ax.set_ylim(1.4, 9.6)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86))
ax.text(0.99, 0.5, "thick: validation\nthin: training", transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)
save(fig, "fig_learning_curves")

# Fig. 3: diversity vs repetition trade-off
fig, ax = plt.subplots(figsize=(COL, 2.6))
real = ev["small"]["reference_real_stories"]
for r, col, mk, lab in (("baseline", C2, "s", "Baseline (5.3M)"), ("small", C1, "o", "Small (13.9M)")):
    xs = [dec[r][s]["distinct_2"] for s in SETTINGS]
    ys = [100 * dec[r][s]["rep4_mean_ci"][0] for s in SETTINGS]
    lo = [100 * (dec[r][s]["rep4_mean_ci"][0] - dec[r][s]["rep4_mean_ci"][1]) for s in SETTINGS]
    hi = [100 * (dec[r][s]["rep4_mean_ci"][2] - dec[r][s]["rep4_mean_ci"][0]) for s in SETTINGS]
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=col, ms=4, mec="white", mew=0.6, elinewidth=0.6, capsize=1.5,
                label=lab, zorder=3)
    for s, x, y in zip(SETTINGS, xs, ys):
        if s in ("greedy", "temp_0.7", "temp_1.0"):
            ax.annotate(LABEL[s], (x, y), xytext=(5, -10) if r == "baseline" else (5, 2), textcoords="offset points", fontsize=6, color=INK)
ax.scatter([real["distinct_2"]], [100 * real["rep4_rate"]], marker="*", s=70, color=INK, zorder=4,
           label="Human-written stories")
ax.annotate("top-$k$ / top-$p$\n(k=20, 50; p=0.90, 0.95)", xy=(0.733, 2.6), xytext=(0.6, 0.55), fontsize=6,
            color=INK, arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
ax.set_yscale("log")
ax.set_ylim(0.3, 120)
ax.set_xlabel("Distinct-2 (higher = more diverse)")
ax.set_ylabel("Repeated 4-grams per story (%)")
ax.legend(frameon=False, loc="lower left")
save(fig, "fig_tradeoff")

# Fig. 4: self-NLL of generated text (likelihood under the generating model)
fig, ax = plt.subplots(figsize=(COL, 2.2))
xs = np.arange(len(SETTINGS) + 1)
labs = [LABEL[s] for s in SETTINGS] + ["Human"]
w = 0.38
for i, (r, col, lab) in enumerate((("baseline", C2, "Baseline"), ("small", C1, "Small"))):
    vals = [stats["self_nll"][r][s] for s in SETTINGS + ["real"]]
    m = [v[0] for v in vals]
    e = [[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]]
    ax.bar(xs + (i - 0.5) * w, m, w - 0.04, color=col, label=lab, yerr=e, error_kw=dict(lw=0.6, capsize=1.5))
ax.set_xticks(xs)
ax.set_xticklabels(labs, rotation=35, ha="right")
ax.set_ylabel("NLL under own model (nats/token)")
ax.legend(frameon=False, loc="upper left")
save(fig, "fig_self_nll")

# Fig. 5: nucleus size distribution (small model, real test text)
fig, ax = plt.subplots(figsize=(COL, 2.1))
for p, col in ((0.90, C3), (0.95, C1)):
    v = np.sort(np.array(ns[p]))
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=1.4, label=f"top-$p$, $p={p:.2f}$")
for k, ha, f in ((20, "right", 0.93), (50, "left", 1.08)):
    ax.axvline(k, color=MUTED, lw=0.6, ls="--")
    ax.text(k * f, 0.05, f"$k={k}$", fontsize=6.5, color=MUTED, ha=ha)
ax.set_xscale("log")
ax.set_xlabel("Tokens kept at a decoding step (nucleus size, log scale)")
ax.set_ylabel("Cumulative fraction of steps")
ax.legend(frameon=False, loc="lower right")
save(fig, "fig_nucleus")

# Fig. 6: error rates with Wilson intervals + co-occurrence
fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL * 2, 2.0), gridspec_kw=dict(width_ratios=[1.6, 1]))
names = list(err.keys())
rates = [100 * err[k]["rate"] for k in names]
lo = [100 * (err[k]["rate"] - err[k]["wilson"][0]) for k in names]
hi = [100 * (err[k]["wilson"][1] - err[k]["rate"]) for k in names]
cols = [C1, C1, C1, MUTED, MUTED]
y = np.arange(len(names))[::-1]
a1.barh(y, rates, color=cols, height=0.6, xerr=[lo, hi], error_kw=dict(lw=0.6, capsize=2))
for yi, rv, k in zip(y, rates, names):
    a1.text(rv + hi[names.index(k)] + 1.5, yi, f"{rv:.0f}% ({err[k]['k']}/{n})", va="center", fontsize=6.5, color=INK)
a1.set_yticks(y)
a1.set_yticklabels(names)
a1.set_xlim(0, 112)
a1.set_xlabel("Stories with at least one error (%), 95% Wilson CI")
a1.grid(axis="y", visible=False)
lab3 = ["contra.", "entity", "grammar"]
M = np.array([[co[a][b] for b in hand] for a in hand])
a2.imshow(M, cmap="Blues", vmin=0, vmax=n)
for i in range(3):
    for j in range(3):
        a2.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=7, color="white" if M[i, j] > 30 else INK)
a2.set_xticks(range(3))
a2.set_yticks(range(3))
a2.set_xticklabels(lab3)
a2.set_yticklabels(lab3)
a2.grid(False)
a2.set_title("Co-occurrence (stories)", fontsize=7)
save(fig, "fig_errors")

print(json.dumps({"self_nll_small": stats["self_nll"]["small"], "nucleus": stats["nucleus_small"],
                  "topk_mass": stats["topk_mass_small"], "errors": {k: v["k"] for k, v in err.items()},
                  "rep_flag": stats["errors_small"]["rep_flag"],
                  "spearman": stats["errors_small"]["length_vs_errors_spearman"],
                  "long_short": [stats["errors_small"]["mean_errors_long"], stats["errors_small"]["mean_errors_short"],
                                 stats["errors_small"]["long_vs_short_p"]],
                  "size_effect": stats["size_effect"], "vs_p095": stats["vs_top_p_095"],
                  "real_madeup": stats["real_madeup"], "rep_size_p": stats["errors_rep_size_fisher_p"]}, indent=1))
