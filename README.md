# Molecular property prediction: representation comparison

One PyTorch codebase that predicts a molecular property from three different
representations of the same molecules, so the representations can be compared on
an equal footing:

- **FingerprintMLP**: a multilayer perceptron on Morgan (ECFP-like) fingerprints.
- **SmilesCNN**: a 1D convolutional network on character-tokenized SMILES.
- **SmilesRNN**: a bidirectional LSTM on character-tokenized SMILES.

All three share the same dataset, splits, target standardization, training loop,
and metrics. The regression target is an RDKit descriptor (logP, TPSA, or
molecular weight) computed directly from structure, which means the benchmark is
fully reproducible from SMILES alone with no dataset license to manage.

## What it does

- A character-level SMILES tokenizer with a fitted vocabulary (`tokenizer.py`).
- Morgan fingerprints and RDKit descriptor targets (`featurize.py`).
- A batching layer that pads variable-length SMILES and packs them for the RNN
  so padding does not leak into the recurrent states (`data.py`, `models.py`).
- A single `Trainer` and a benchmark script that reports RMSE, MAE, and R^2 and
  writes learning-curve and parity figures.

## What it does not do

- The models are compact and CPU-friendly; they are demonstrations of the
  representations, not tuned production models.
- Only single-target regression is implemented.
- Targets are RDKit descriptors, not experimentally measured properties. Swap in
  a measured-property CSV to benchmark on lab data.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Run

Fully offline (built-in SMILES set):

```bash
python scripts/benchmark.py --target logp --epochs 40
```

On a larger downloaded SMILES set:

```bash
python scripts/download_data.py --out data/smiles.csv
python scripts/benchmark.py --csv data/smiles.csv --target logp --epochs 40
```

## Results

Target logP, computed by RDKit on 1128 public SMILES (the MoleculeNet ESOL SMILES
column), random 80/10/10 split, 40 epochs, single CPU, seed 0. Produced by
`scripts/benchmark.py` in this repository.

| Representation |   RMSE |    MAE |    R^2 |
|----------------|-------:|-------:|-------:|
| SmilesRNN      | 0.2866 | 0.1986 | 0.9742 |
| SmilesCNN      | 0.5121 | 0.3621 | 0.9176 |
| FingerprintMLP | 0.7442 | 0.5061 | 0.8259 |

On this target the sequence models learn the near-additive structure of logP
from raw SMILES and beat the fixed fingerprint baseline; the LSTM edges out the
CNN. The ranking depends on the target: try `--target tpsa` to see it change.

## Layout

```
src/molprop/     tokenizer, featurize, data, models, train
scripts/         download_data.py, benchmark.py
notebooks/       demo.ipynb (executed)
tests/           pytest suite for tokenizer, data, and models
data/            gitignored; see data/README.md
```

## Tests

```bash
pytest -q
ruff check src tests scripts
```

## License

MIT, see [LICENSE](LICENSE).

## Author

Aamir Malik. [GitHub](https://github.com/aamirmalik-dr) ·
[LinkedIn](https://linkedin.com/in/dr-aamirmalik)
