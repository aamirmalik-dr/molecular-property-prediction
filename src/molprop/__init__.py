"""molprop: a small library for comparing molecular representations.

One PyTorch codebase, three interchangeable encoders over the same molecules:

* ``FingerprintMLP``  - a multilayer perceptron on Morgan fingerprints,
* ``SmilesCNN``       - a 1D convolutional network on tokenized SMILES,
* ``SmilesRNN``       - an LSTM on tokenized SMILES.

The public API is designed to be extended: any :class:`MolEncoder` you register
trains through the same :class:`Trainer`. Targets are RDKit descriptors (logP,
TPSA, molecular weight) computed on any public SMILES set, so every benchmark is
fully reproducible with no dataset license to worry about.

Quickstart::

    from molprop import load_sample, random_split, build_encoder, Trainer, set_seed

    set_seed(0)
    dataset = load_sample(target="logp")
    train_set, val_set, test_set = random_split(dataset)
    model = build_encoder("smiles_rnn", vocab_size=len(dataset.tokenizer), n_bits=1024)
    trainer = Trainer(model, patience=5).fit(train_set, val_set, epochs=40, verbose=False)
    print(trainer.evaluate(test_set))
"""

from molprop.data import (
    Batch,
    PropertyDataset,
    build_dataset,
    collate_batch,
    load_sample,
    load_smiles_csv,
    random_split,
)
from molprop.featurize import compute_target, morgan_fingerprint
from molprop.models import (
    FingerprintMLP,
    MolEncoder,
    SmilesCNN,
    SmilesRNN,
    available_encoders,
    build_encoder,
    register_encoder,
)
from molprop.tokenizer import SmilesTokenizer
from molprop.train import Trainer, regression_metrics, set_seed

__all__ = [
    "Batch",
    "PropertyDataset",
    "build_dataset",
    "collate_batch",
    "load_sample",
    "load_smiles_csv",
    "random_split",
    "compute_target",
    "morgan_fingerprint",
    "MolEncoder",
    "FingerprintMLP",
    "SmilesCNN",
    "SmilesRNN",
    "available_encoders",
    "build_encoder",
    "register_encoder",
    "SmilesTokenizer",
    "Trainer",
    "regression_metrics",
    "set_seed",
]

__version__ = "0.2.0"
