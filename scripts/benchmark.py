"""Benchmark the registered encoders on an RDKit descriptor target.

Trains every encoder returned by ``available_encoders`` under one budget and
writes a metrics table, the learning-curves hero figure, and a per-encoder parity
figure. Runs fully offline on the committed sample by default.

Usage:
    # Offline, committed sample (a few hundred public SMILES):
    python scripts/benchmark.py --target logp --epochs 60

    # On a downloaded SMILES CSV:
    python scripts/benchmark.py --csv data/smiles.csv --target logp --epochs 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from molprop.data import build_dataset, load_sample, load_smiles_csv, random_split
from molprop.models import available_encoders, build_encoder
from molprop.train import Trainer, set_seed

# Validated categorical palette (dataviz skill): blue, orange, violet. Worst
# adjacent CVD dE 96.7, all >= 3:1 contrast on the light surface.
ENCODER_COLORS = ["#2a78d6", "#eb6834", "#4a3aa7"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None, help="optional SMILES CSV; omit to use the sample")
    parser.add_argument("--target", default="logp", choices=["logp", "tpsa", "mw"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-bits", type=int, default=1024)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(0)
    if args.csv:
        smiles = load_smiles_csv(args.csv)
        dataset = build_dataset(smiles, target=args.target, n_bits=args.n_bits)
        source = args.csv
    else:
        dataset = load_sample(target=args.target, n_bits=args.n_bits)
        source = "committed sample"
    train_set, val_set, test_set = random_split(dataset, seed=0)
    vocab = len(dataset.tokenizer)
    print(
        f"Source: {source} | {len(dataset)} molecules, vocab={vocab}, target={args.target} "
        f"(train={len(train_set)}, val={len(val_set)}, test={len(test_set)})"
    )

    names = available_encoders()
    results: dict[str, dict[str, float]] = {}
    trainers: dict[str, Trainer] = {}
    for name in names:
        set_seed(0)
        model = build_encoder(name, vocab_size=vocab, n_bits=args.n_bits)
        trainer = Trainer(model, lr=1e-3, weight_decay=1e-5, patience=args.patience)
        trainer.fit(train_set, val_set, epochs=args.epochs, batch_size=args.batch_size, verbose=False)
        results[name] = trainer.evaluate(test_set)
        trainers[name] = trainer

    print("\nTest-set results:")
    header = f"{'encoder':<18}{'RMSE':>10}{'MAE':>10}{'R2':>10}{'best_ep':>9}"
    print(header)
    print("-" * len(header))
    for name in names:
        m = results[name]
        print(
            f"{name:<18}{m['rmse']:>10.4f}{m['mae']:>10.4f}{m['r2']:>10.4f}"
            f"{trainers[name].best_epoch:>9d}"
        )
    payload = {
        "target": args.target,
        "source": source,
        "n_molecules": len(dataset),
        "split": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
        "epochs": args.epochs,
        "results": results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))

    _plot_learning_curves(names, trainers, args.target, out_dir / "learning_curves.png")
    _plot_parity(names, trainers, test_set, args.target, out_dir / "parity.png")

    best = min(results, key=lambda k: results[k]["rmse"])
    print(f"\nBest encoder: {best} (RMSE {results[best]['rmse']:.4f}). Wrote figures to {out_dir}/")
    return 0


def _plot_learning_curves(names, trainers, target, path) -> None:
    """Train and validation loss (standardized MSE) for every encoder, one axis."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _style_axes(ax)
    for name, color in zip(names, ENCODER_COLORS, strict=False):
        tr = trainers[name]
        epochs = range(1, len(tr.history["train"]) + 1)
        std = tr.target_std or 1.0
        val_mse = [(r / std) ** 2 for r in tr.history["val"]]
        ax.plot(epochs, tr.history["train"], color=color, linewidth=2, label=f"{name} (train)")
        ax.plot(
            epochs, val_mse, color=color, linewidth=2, linestyle="--", label=f"{name} (val)"
        )
    ax.set_yscale("log")
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("standardized MSE (log scale)", color=INK)
    ax.set_title(f"Learning curves by encoder ({target})", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=8, ncol=1, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def _plot_parity(names, trainers, test_set, target, path) -> None:
    """One parity panel per encoder: predicted vs true on the test split."""
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2), squeeze=False)
    truth = test_set.targets
    lo = float(truth.min())
    hi = float(truth.max())
    for ax, name, color in zip(axes[0], names, ENCODER_COLORS, strict=False):
        _style_axes(ax)
        preds = trainers[name].predict(test_set)
        lo_i = min(lo, float(preds.min()))
        hi_i = max(hi, float(preds.max()))
        ax.plot([lo_i, hi_i], [lo_i, hi_i], color=MUTED, linewidth=1, linestyle="--", zorder=1)
        ax.scatter(truth, preds, color=color, alpha=0.6, s=20, edgecolor="white", linewidth=0.4, zorder=2)
        r2 = trainers[name].evaluate(test_set)["r2"]
        ax.set_title(f"{name}\nR2 = {r2:.3f}", color=INK, fontsize=10)
        ax.set_xlabel("true", color=INK)
        ax.set_ylabel("predicted", color=INK)
    fig.suptitle(f"Parity on the test split ({target})", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
