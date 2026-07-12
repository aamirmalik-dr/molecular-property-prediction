# Model card: molprop encoders

This card describes the three encoders in `molprop` and the data they are trained
on. They are compact demonstration models for comparing molecular
representations, not tuned production predictors.

## Overview

One PyTorch codebase provides three interchangeable encoders that map a molecule
to a single regression value:

| Encoder | Input representation | Architecture |
|---------|----------------------|--------------|
| `fingerprint_mlp` | 1024-bit Morgan (ECFP-like) fingerprint | 2-hidden-layer MLP with dropout |
| `smiles_cnn` | character-tokenized SMILES | embedding, 3 dilated 1D convolutions, global max pool, linear head |
| `smiles_rnn` | character-tokenized SMILES | embedding, bidirectional LSTM, concat of final states, linear head |

All three share the same dataset, splits, target standardization, training loop
(`Trainer`), and metrics, so the comparison is on an equal footing. Each is a
`MolEncoder` and any user-registered encoder trains through the same path.

## Data

- **Source**: public SMILES strings. The committed sample (`data/sample_smiles.csv`,
  300 molecules) is carved from the public MoleculeNet ESOL (Delaney) SMILES
  column. The full set (1128 molecules) is fetched by `scripts/download_data.py`.
- **Target**: an RDKit descriptor computed directly from structure, one of logP
  (default), TPSA, or molecular weight. Because the target is computed, not
  measured, every benchmark is reproducible from SMILES alone with no dataset
  license to manage.
- **Features**: Morgan fingerprints (radius 2, 1024 bits) for the MLP, and a
  character-level SMILES vocabulary fit on the training split for the sequence
  models. Padding is index 0 and packed out of the RNN so it does not leak into
  the recurrent states.
- **Splits**: random 80/10/10 train/validation/test, seed 0, with the training
  tokenizer reused on the validation and test splits.

## Training

Targets are standardized with the training-set mean and standard deviation,
optimized with Adam (learning rate 1e-3, weight decay 1e-5) under an MSE loss, and
de-standardized at prediction time. Early stopping on validation RMSE restores the
best-scoring weights. Everything runs on CPU.

## Intended use

- Teaching and portfolio demonstration of how the input representation, not just
  the model, drives regression quality on molecular data.
- A small, extensible starting point: register a new `MolEncoder` and benchmark it
  against the three baselines with no other code changes.

## Out of scope

- Not a tuned or production property predictor. The models are intentionally small.
- Single-target regression only. No classification, no multitask, no uncertainty.
- The default target is an RDKit descriptor, not an experimentally measured
  property. To model lab data, supply a measured-property CSV and adapt the target.
- No conformer, 3D, or graph-based representations; the point is a fingerprint vs
  sequence comparison.

## Measured results

Target logP, seed 0, 60 epochs, single CPU. Produced by `scripts/benchmark.py` in
this repository and written to `results/metrics.json`.

**Committed sample (300 molecules, offline quickstart):**

| Encoder | RMSE | MAE | R^2 |
|---------|-----:|----:|----:|
| smiles_rnn | 0.6196 | 0.4732 | 0.8848 |
| smiles_cnn | 0.7073 | 0.5525 | 0.8499 |
| fingerprint_mlp | 1.0545 | 0.8183 | 0.6665 |

The sequence models learn the near-additive structure of logP from raw SMILES and
beat the fixed fingerprint baseline, with the LSTM ahead of the CNN. The ranking
depends on the target; try `--target tpsa` to see it shift.
