"""Compare molecular representations for property prediction.

One PyTorch codebase, three encoders over the same molecules:

* ``FingerprintMLP``  - a multilayer perceptron on Morgan fingerprints,
* ``SmilesCNN``       - a 1D convolutional network on tokenized SMILES,
* ``SmilesRNN``       - an LSTM on tokenized SMILES.

Targets are RDKit descriptors (logP, TPSA, molecular weight) computed on any
public SMILES set, so every benchmark is fully reproducible with no dataset
license to worry about.
"""

from molprop.data import (
    PropertyDataset,
    build_dataset,
    collate_batch,
    random_split,
)
from molprop.featurize import morgan_fingerprint
from molprop.models import FingerprintMLP, SmilesCNN, SmilesRNN
from molprop.tokenizer import SmilesTokenizer
from molprop.train import Trainer, regression_metrics, set_seed

__all__ = [
    "PropertyDataset",
    "build_dataset",
    "collate_batch",
    "random_split",
    "morgan_fingerprint",
    "FingerprintMLP",
    "SmilesCNN",
    "SmilesRNN",
    "SmilesTokenizer",
    "Trainer",
    "regression_metrics",
    "set_seed",
]

__version__ = "0.1.0"
