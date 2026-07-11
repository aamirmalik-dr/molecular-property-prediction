"""Benchmark three molecular representations on an RDKit descriptor target.

Trains FingerprintMLP, SmilesCNN, and SmilesRNN under one budget and writes a
metrics table, a training-curve figure, and a parity plot.

Usage:
    # Offline, built-in SMILES:
    python scripts/benchmark.py --target logp --epochs 40

    # On a downloaded SMILES CSV:
    python scripts/benchmark.py --csv data/smiles.csv --target logp --epochs 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from molprop.data import build_dataset, load_smiles_csv, random_split
from molprop.models import FingerprintMLP, SmilesCNN, SmilesRNN
from molprop.train import Trainer, set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None, help="optional SMILES CSV")
    parser.add_argument("--target", default="logp", choices=["logp", "tpsa", "mw"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--n-bits", type=int, default=1024)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    smiles = load_smiles_csv(args.csv) if args.csv else None
    set_seed(0)
    dataset = build_dataset(smiles, target=args.target, n_bits=args.n_bits)
    train_set, val_set, test_set = random_split(dataset, seed=0)
    vocab = len(dataset.tokenizer)
    print(
        f"Dataset: {len(dataset)} molecules, vocab={vocab}, target={args.target} "
        f"(train={len(train_set)}, val={len(val_set)}, test={len(test_set)})"
    )

    factories = {
        "FingerprintMLP": lambda: FingerprintMLP(n_bits=args.n_bits),
        "SmilesCNN": lambda: SmilesCNN(vocab),
        "SmilesRNN": lambda: SmilesRNN(vocab),
    }

    results: dict[str, dict[str, float]] = {}
    trainers: dict[str, Trainer] = {}
    for name, make in factories.items():
        set_seed(0)
        trainer = Trainer(make(), lr=1e-3, weight_decay=1e-5)
        trainer.fit(train_set, val_set, epochs=args.epochs, verbose=False)
        results[name] = trainer.evaluate(test_set)
        trainers[name] = trainer

    print("\nTest-set results:")
    header = f"{'representation':<16}{'RMSE':>10}{'MAE':>10}{'R2':>10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<16}{m['rmse']:>10.4f}{m['mae']:>10.4f}{m['r2']:>10.4f}")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    plt.figure(figsize=(7, 4.5))
    for name, trainer in trainers.items():
        plt.plot(trainer.history["val"], label=name)
    plt.xlabel("epoch")
    plt.ylabel("validation RMSE")
    plt.title(f"Representation comparison ({args.target})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "val_rmse.png", dpi=120)
    plt.close()

    best = min(results, key=lambda k: results[k]["rmse"])
    preds = trainers[best].predict(test_set)
    plt.figure(figsize=(5, 5))
    plt.scatter(test_set.targets, preds, alpha=0.6, s=18)
    lo = min(test_set.targets.min(), preds.min())
    hi = max(test_set.targets.max(), preds.max())
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("true")
    plt.ylabel("predicted")
    plt.title(f"{best} parity (test, {args.target})")
    plt.tight_layout()
    plt.savefig(out_dir / "parity.png", dpi=120)
    plt.close()

    print(f"\nBest representation: {best}. Wrote metrics and figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
