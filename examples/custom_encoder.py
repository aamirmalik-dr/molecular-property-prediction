"""Add your own encoder and train it through the shared Trainer.

This is the extension point of the library: subclass MolEncoder, register it,
and it is immediately usable by name through build_encoder and trainable by the
same Trainer as the built-in encoders. Here we add a small mean-pooled embedding
encoder over tokenized SMILES.

Run:
    python examples/custom_encoder.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

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
from molprop.data import Batch


@register_encoder("mean_embed")
class MeanEmbedEncoder(MolEncoder):
    """Average the token embeddings, then a two-layer regression head."""

    def __init__(self, vocab_size: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    @classmethod
    def from_dataset_spec(
        cls, *, vocab_size: int, n_bits: int, embed_dim: int = 64, **kwargs: object
    ) -> MeanEmbedEncoder:
        return cls(vocab_size=vocab_size, embed_dim=embed_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        emb = self.embed(batch.ids)  # (B, T, E)
        mask = (batch.ids != 0).unsqueeze(-1).float()
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


def main() -> None:
    print("registered encoders:", available_encoders())
    set_seed(0)
    dataset = load_sample(target="logp")
    train_set, val_set, test_set = random_split(dataset)

    model = build_encoder("mean_embed", vocab_size=len(dataset.tokenizer), n_bits=1024)
    trainer = Trainer(model, patience=10)
    trainer.fit(train_set, val_set, epochs=60, batch_size=16, verbose=False)
    print("mean_embed test metrics:", {k: round(v, 4) for k, v in trainer.evaluate(test_set).items()})


if __name__ == "__main__":
    main()
