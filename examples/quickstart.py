"""Offline quickstart: train one encoder on the committed sample.

Runs in well under a minute on a CPU with no network access. It loads the
packaged sample of a few hundred public SMILES, computes the logP target with
RDKit, and trains the LSTM encoder with early stopping.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

from molprop import Trainer, build_encoder, load_sample, random_split, set_seed


def main() -> None:
    set_seed(0)
    dataset = load_sample(target="logp")
    train_set, val_set, test_set = random_split(dataset)
    print(f"{len(dataset)} molecules, vocab={len(dataset.tokenizer)}")

    model = build_encoder("smiles_rnn", vocab_size=len(dataset.tokenizer), n_bits=1024)
    trainer = Trainer(model, lr=1e-3, weight_decay=1e-5, patience=10)
    trainer.fit(train_set, val_set, epochs=60, batch_size=16, verbose=False)

    metrics = trainer.evaluate(test_set)
    print(f"best epoch: {trainer.best_epoch} (stopped at {trainer.stopped_epoch})")
    print("test metrics:", {k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
