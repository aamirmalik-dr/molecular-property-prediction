import torch

from molprop.data import build_dataset, collate_batch, random_split
from molprop.models import FingerprintMLP, SmilesCNN, SmilesRNN
from molprop.train import Trainer, regression_metrics


def _batch(n_bits=128):
    ds = build_dataset(["CCO", "CCN", "c1ccccc1", "CC(=O)O"], target="logp", n_bits=n_bits)
    return ds, collate_batch([ds[i] for i in range(len(ds))])


def test_fingerprint_mlp_forward():
    ds, batch = _batch()
    model = FingerprintMLP(n_bits=128, hidden_dim=32)
    assert model(batch).shape == (len(ds),)


def test_smiles_cnn_forward():
    ds, batch = _batch()
    model = SmilesCNN(len(ds.tokenizer), embed_dim=16, channels=32)
    assert model(batch).shape == (len(ds),)


def test_smiles_rnn_forward():
    ds, batch = _batch()
    model = SmilesRNN(len(ds.tokenizer), embed_dim=16, hidden_dim=32)
    assert model(batch).shape == (len(ds),)


def test_metrics_perfect():
    import numpy as np

    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["rmse"] == 0.0 and m["r2"] == 1.0


def test_trainer_reduces_loss():
    torch.manual_seed(0)
    ds = build_dataset(target="logp", n_bits=256)
    tr, va, te = random_split(ds, seed=0)
    trainer = Trainer(FingerprintMLP(n_bits=256, hidden_dim=64))
    trainer.fit(tr, va, epochs=20, batch_size=16, verbose=False)
    assert trainer.history["train"][-1] < trainer.history["train"][0]
