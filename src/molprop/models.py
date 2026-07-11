"""Three encoders for molecular property regression.

Each model consumes a :class:`~molprop.data.Batch` and returns one prediction
per molecule, so they are interchangeable inside the training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molprop.data import Batch


class FingerprintMLP(nn.Module):
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

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.net(batch.fp).squeeze(-1)


class SmilesCNN(nn.Module):
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

    def forward(self, batch: Batch) -> torch.Tensor:
        # (B, T, E) -> (B, E, T) for Conv1d.
        x = self.embed(batch.ids).transpose(1, 2)
        for conv in self.convs:
            x = torch.relu(conv(x))
        pooled = x.max(dim=-1).values
        return self.head(pooled).squeeze(-1)


class SmilesRNN(nn.Module):
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
