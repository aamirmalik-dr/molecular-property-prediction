"""Compare every registered encoder on the committed sample.

Trains each encoder in the registry under one budget and prints a ranked table.
This is the scripted core of what scripts/benchmark.py does, minus the figures.

Run:
    python examples/compare_encoders.py
"""

from __future__ import annotations

from molprop import (
    Trainer,
    available_encoders,
    build_encoder,
    load_sample,
    random_split,
    set_seed,
)


def main() -> None:
    set_seed(0)
    dataset = load_sample(target="logp")
    train_set, val_set, test_set = random_split(dataset)
    vocab = len(dataset.tokenizer)

    rows = []
    for name in available_encoders():
        set_seed(0)
        model = build_encoder(name, vocab_size=vocab, n_bits=1024)
        trainer = Trainer(model, lr=1e-3, weight_decay=1e-5, patience=10)
        trainer.fit(train_set, val_set, epochs=60, batch_size=16, verbose=False)
        rows.append((name, trainer.evaluate(test_set)))

    rows.sort(key=lambda r: r[1]["rmse"])
    print(f"{'encoder':<18}{'RMSE':>10}{'MAE':>10}{'R2':>10}")
    for name, m in rows:
        print(f"{name:<18}{m['rmse']:>10.4f}{m['mae']:>10.4f}{m['r2']:>10.4f}")


if __name__ == "__main__":
    main()
