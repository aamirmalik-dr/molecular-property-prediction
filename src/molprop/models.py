"""Encoders for molecular property regression and the pluggable API.

Every encoder is a :class:`MolEncoder`: an ``nn.Module`` whose ``forward``
consumes a :class:`~molprop.data.Batch` and returns one prediction per molecule,
so any encoder is interchangeable inside :class:`~molprop.train.Trainer`.

To add your own encoder, subclass :class:`MolEncoder`, implement ``forward`` and
the :meth:`MolEncoder.from_dataset_spec` factory, and register it::

    from molprop.models import MolEncoder, register_encoder

    @register_encoder("my_encoder")
    class MyEncoder(MolEncoder):
        @classmethod
        def from_dataset_spec(cls, *, vocab_size, n_bits, **kwargs):
            return cls(**kwargs)

        def forward(self, batch):
            ...

Once registered it is reachable by name through :func:`build_encoder` and shows
up in :func:`available_encoders`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from molprop.data import Batch


class MolEncoder(nn.Module, ABC):
    """Abstract base for a molecular encoder.

    A concrete encoder maps a padded :class:`~molprop.data.Batch` to a rank-1
    tensor of shape ``(batch_size,)`` holding one scalar prediction per molecule.
    Subclasses must implement :meth:`forward` and :meth:`from_dataset_spec` so the
    registry can build them from dataset-derived dimensions.
    """

    @classmethod
    @abstractmethod
    def from_dataset_spec(cls, *, vocab_size: int, n_bits: int, **kwargs: object) -> MolEncoder:
        """Build the encoder from dataset dimensions.

        Args:
            vocab_size: Size of the fitted SMILES tokenizer vocabulary.
            n_bits: Length of the Morgan fingerprint vector.
            **kwargs: Encoder-specific hyperparameters passed to ``__init__``.

        Returns:
            An initialized encoder instance.
        """

    @abstractmethod
    def forward(self, batch: Batch) -> torch.Tensor:
        """Return one prediction per molecule, shape ``(batch_size,)``."""


_ENCODER_REGISTRY: dict[str, type[MolEncoder]] = {}


def register_encoder(name: str):
    """Class decorator that registers a :class:`MolEncoder` under ``name``.

    Args:
        name: Unique key used by :func:`build_encoder` and shown in
            :func:`available_encoders`.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def _register(cls: type[MolEncoder]) -> type[MolEncoder]:
        if name in _ENCODER_REGISTRY:
            raise ValueError(f"encoder {name!r} is already registered")
        _ENCODER_REGISTRY[name] = cls
        return cls

    return _register


def available_encoders() -> list[str]:
    """Return the sorted names of all registered encoders."""
    return sorted(_ENCODER_REGISTRY)


def build_encoder(name: str, *, vocab_size: int, n_bits: int, **kwargs: object) -> MolEncoder:
    """Instantiate a registered encoder by name from dataset dimensions.

    Args:
        name: A key from :func:`available_encoders`.
        vocab_size: Size of the fitted tokenizer vocabulary.
        n_bits: Length of the Morgan fingerprint vector.
        **kwargs: Encoder-specific hyperparameters.

    Returns:
        An initialized :class:`MolEncoder`.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in _ENCODER_REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; choose from {available_encoders()}")
    return _ENCODER_REGISTRY[name].from_dataset_spec(
        vocab_size=vocab_size, n_bits=n_bits, **kwargs
    )


@register_encoder("fingerprint_mlp")
class FingerprintMLP(MolEncoder):
    """A multilayer perceptron over Morgan fingerprints."""

    def __init__(self, n_bits: int = 1024, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @classmethod
    def from_dataset_spec(
        cls, *, vocab_size: int, n_bits: int, hidden_dim: int = 256, **kwargs: object
    ) -> FingerprintMLP:
        return cls(n_bits=n_bits, hidden_dim=hidden_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.net(batch.fp).squeeze(-1)


@register_encoder("smiles_cnn")
class SmilesCNN(MolEncoder):
    """A 1D convolutional network over tokenized SMILES.

    An embedding maps token ids to vectors, three convolutions with increasing
    dilation capture local motifs, and global max pooling produces a molecule
    vector for the regression head.
    """

    def __init__(
        self, vocab_size: int, embed_dim: int = 64, channels: int = 128
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(
                embed_dim if i == 0 else channels,
                channels,
                kernel_size=3,
                padding=d,
                dilation=d,
            )
            for i, d in enumerate([1, 2, 4])
        )
        self.head = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, 1)
        )

    @classmethod
    def from_dataset_spec(
        cls,
        *,
        vocab_size: int,
        n_bits: int,
        embed_dim: int = 64,
        channels: int = 128,
        **kwargs: object,
    ) -> SmilesCNN:
        return cls(vocab_size=vocab_size, embed_dim=embed_dim, channels=channels)

    def forward(self, batch: Batch) -> torch.Tensor:
        # (B, T, E) -> (B, E, T) for Conv1d.
        x = self.embed(batch.ids).transpose(1, 2)
        for conv in self.convs:
            x = torch.relu(conv(x))
        pooled = x.max(dim=-1).values
        return self.head(pooled).squeeze(-1)


@register_encoder("smiles_rnn")
class SmilesRNN(MolEncoder):
    """A bidirectional LSTM over tokenized SMILES.

    Packing is used so padding does not affect the recurrent states; the final
    forward and backward hidden states are concatenated for the regression head.
    """

    def __init__(
        self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    @classmethod
    def from_dataset_spec(
        cls,
        *,
        vocab_size: int,
        n_bits: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        **kwargs: object,
    ) -> SmilesRNN:
        return cls(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        embedded = self.embed(batch.ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            batch.lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.rnn(packed)
        # h_n: (num_directions, B, hidden) -> concat the two directions.
        final = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(final).squeeze(-1)
