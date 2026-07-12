"""Tests for the public library API: registry, Trainer, and load_sample."""

from __future__ import annotations

import pytest
import torch

from molprop import (
    MolEncoder,
    Trainer,
    available_encoders,
    build_encoder,
    load_sample,
    random_split,
    register_encoder,
    set_seed,
)
from molprop.data import Batch, build_dataset, collate_batch


def test_builtin_encoders_registered():
    names = available_encoders()
    assert {"fingerprint_mlp", "smiles_cnn", "smiles_rnn"} <= set(names)


def test_build_encoder_returns_mol_encoder():
    ds = build_dataset(["CCO", "CCN", "c1ccccc1"], target="logp", n_bits=128)
    for name in available_encoders():
        model = build_encoder(name, vocab_size=len(ds.tokenizer), n_bits=128)
        assert isinstance(model, MolEncoder)
        batch = collate_batch([ds[i] for i in range(len(ds))])
        assert model(batch).shape == (len(ds),)


def test_build_encoder_unknown_name_raises():
    with pytest.raises(KeyError):
        build_encoder("does_not_exist", vocab_size=10, n_bits=64)


def test_register_duplicate_name_raises():
    with pytest.raises(ValueError):

        @register_encoder("fingerprint_mlp")
        class _Dup(MolEncoder):  # pragma: no cover - registration fails first
            @classmethod
            def from_dataset_spec(cls, *, vocab_size, n_bits, **kwargs):
                return cls()

            def forward(self, batch):
                return torch.zeros(1)


def test_user_can_register_and_train_a_new_encoder():
    @register_encoder("mean_fp")
    class MeanFP(MolEncoder):
        """Trivial encoder: a learned linear map over the fingerprint mean."""

        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Linear(1, 1)

        @classmethod
        def from_dataset_spec(cls, *, vocab_size: int, n_bits: int, **kwargs) -> MeanFP:
            return cls()

        def forward(self, batch: Batch) -> torch.Tensor:
            return self.w(batch.fp.mean(dim=-1, keepdim=True)).squeeze(-1)

    assert "mean_fp" in available_encoders()
    ds = build_dataset(target="logp", n_bits=128)
    tr, va, te = random_split(ds, seed=0)
    trainer = Trainer(build_encoder("mean_fp", vocab_size=len(ds.tokenizer), n_bits=128))
    trainer.fit(tr, va, epochs=5, verbose=False)
    assert len(trainer.history["train"]) == 5


def test_trainer_history_and_early_stopping():
    set_seed(0)
    ds = build_dataset(target="logp", n_bits=256)
    tr, va, te = random_split(ds, seed=0)
    trainer = Trainer(
        build_encoder("fingerprint_mlp", vocab_size=len(ds.tokenizer), n_bits=256),
        patience=3,
    )
    trainer.fit(tr, va, epochs=100, batch_size=16, verbose=False)
    assert len(trainer.history["train"]) == len(trainer.history["val"])
    assert trainer.stopped_epoch <= 100
    assert 1 <= trainer.best_epoch <= trainer.stopped_epoch
    # Early stopping should trigger well before the 100-epoch cap on this task.
    assert trainer.stopped_epoch < 100


def test_trainer_without_val_records_no_best_epoch():
    ds = build_dataset(["CCO", "CCN", "CCC", "CCCC"], target="logp", n_bits=64)
    trainer = Trainer(build_encoder("fingerprint_mlp", vocab_size=len(ds.tokenizer), n_bits=64))
    trainer.fit(ds, None, epochs=3, verbose=False)
    assert trainer.best_epoch == -1
    assert len(trainer.history["train"]) == 3


def test_load_sample_offline():
    ds = load_sample(target="logp", n_bits=128)
    assert len(ds) > 100
    assert ds.fingerprints.shape == (len(ds), 128)
    assert ds.targets.shape == (len(ds),)
