"""Plot training/validation loss and learning rate from results/metrics/<run>/log.csv.

Usage: python scripts/plot_curves.py --run small [--run baseline ...]
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_log(run):
    with open(os.path.join("results", "metrics", run, "log.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    args = ap.parse_args()

    fig, (ax_loss, ax_lr) = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in args.run:
        rows = read_log(run)
        steps = [int(r["step"]) for r in rows]
        (line,) = ax_loss.plot(steps, [float(r["train_loss"]) for r in rows], alpha=0.6, label=f"{run} train")
        val = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
        if val:
            ax_loss.plot(*zip(*val), "o-", color=line.get_color(), label=f"{run} val")
        ax_lr.plot(steps, [float(r["lr"]) for r in rows], color=line.get_color(), label=run)
    ax_loss.set(xlabel="step", ylabel="cross-entropy loss", title="Loss")
    ax_lr.set(xlabel="step", ylabel="learning rate", title="Learning rate")
    for ax in (ax_loss, ax_lr):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out = os.path.join("results", "figures", f"loss_{'_'.join(args.run)}.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
