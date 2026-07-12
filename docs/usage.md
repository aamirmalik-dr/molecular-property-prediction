# Usage

`molprop` is a small library for comparing molecular representations on a
regression target. The public API is importable and typed, and it is designed to
be extended with your own encoder. This page is the reference for the surface
you interact with.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## The 30-second quickstart

`load_sample` reads the packaged sample of a few hundred public SMILES and
computes the target with RDKit, so this runs with no network access.

```python
from molprop import load_sample, random_split, build_encoder, Trainer, set_seed

set_seed(0)
dataset = load_sample(target="logp")
train_set, val_set, test_set = random_split(dataset)

model = build_encoder("smiles_rnn", vocab_size=len(dataset.tokenizer), n_bits=1024)
trainer = Trainer(model, patience=10).fit(train_set, val_set, epochs=60, verbose=False)
print(trainer.evaluate(test_set))
```

## Data

- `load_sample(target="logp", n_bits=1024)` returns a `PropertyDataset` over the
  committed sample. Fully offline.
- `build_dataset(smiles, target="logp", n_bits=1024)` builds a dataset from any
  list of SMILES. Invalid SMILES are skipped. The target is an RDKit descriptor,
  one of `"logp"`, `"tpsa"`, `"mw"`.
- `load_smiles_csv(path, column=None)` reads a SMILES column from a CSV.
- `random_split(dataset, fractions=(0.8, 0.1, 0.1), seed=0)` returns train, val,
  and test splits that share the training tokenizer.

A `PropertyDataset` holds the SMILES, their token ids, Morgan fingerprints, the
scalar targets, and the fitted `SmilesTokenizer`.

## Encoders

Three encoders ship in the registry:

| Name | Class | Input |
|------|-------|-------|
| `fingerprint_mlp` | `FingerprintMLP` | Morgan fingerprint |
| `smiles_cnn` | `SmilesCNN` | tokenized SMILES |
| `smiles_rnn` | `SmilesRNN` | tokenized SMILES |

- `available_encoders()` lists the registered names.
- `build_encoder(name, *, vocab_size, n_bits, **kwargs)` instantiates one from the
  dataset dimensions. Extra keyword arguments are passed to the encoder, for
  example `build_encoder("smiles_rnn", vocab_size=32, n_bits=1024, hidden_dim=256)`.

Every encoder is a `MolEncoder`: an `nn.Module` whose `forward` takes a `Batch`
and returns a tensor of shape `(batch_size,)`.

## Training

`Trainer` standardizes the target with the training-set statistics, optimizes an
MSE loss with Adam, and de-standardizes predictions.

```python
trainer = Trainer(model, lr=1e-3, weight_decay=1e-5, patience=10)
trainer.fit(train_set, val_set, epochs=60, batch_size=16)

trainer.history        # {"train": [...], "val": [...]} per epoch
trainer.best_epoch     # epoch of the lowest validation RMSE
trainer.stopped_epoch  # epoch where training actually stopped
trainer.predict(test_set)
trainer.evaluate(test_set)   # {"rmse": ..., "mae": ..., "r2": ...}
```

Passing `patience > 0` enables early stopping on the validation RMSE. When
training stops, the best-scoring weights are restored automatically.

## Adding your own encoder

Subclass `MolEncoder`, implement `forward` and `from_dataset_spec`, and register
it. It is then reachable by name and trainable by the same `Trainer`.

```python
import torch
import torch.nn as nn
from molprop import MolEncoder, register_encoder, build_encoder
from molprop.data import Batch


@register_encoder("mean_embed")
class MeanEmbedEncoder(MolEncoder):
    def __init__(self, vocab_size: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.head = nn.Linear(embed_dim, 1)

    @classmethod
    def from_dataset_spec(cls, *, vocab_size, n_bits, embed_dim=64, **kwargs):
        return cls(vocab_size=vocab_size, embed_dim=embed_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        mask = (batch.ids != 0).unsqueeze(-1).float()
        pooled = (self.embed(batch.ids) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


model = build_encoder("mean_embed", vocab_size=32, n_bits=1024)
```

See `examples/custom_encoder.py` for a runnable version.

## Examples and scripts

- `examples/quickstart.py`: train one encoder offline on the sample.
- `examples/custom_encoder.py`: register and train a new encoder.
- `examples/compare_encoders.py`: rank every registered encoder.
- `scripts/benchmark.py`: full comparison with the learning-curves and parity
  figures written to `results/`.
- `scripts/download_data.py`: fetch the larger public SMILES set for `--csv`.
