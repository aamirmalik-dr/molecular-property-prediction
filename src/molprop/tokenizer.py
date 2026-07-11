"""A small character-level tokenizer for SMILES strings.

SMILES are tokenized per character, which is enough for the sequence models
here. The tokenizer builds its vocabulary from a training corpus and reserves
index 0 for padding and index 1 for unknown characters.
"""

from __future__ import annotations

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class SmilesTokenizer:
    """Character-level vocabulary and encoder for SMILES strings.

    Attributes:
        stoi: Mapping from character to integer id.
        itos: Reverse mapping from id to character.
    """

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.itos: dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self.stoi)

    def fit(self, smiles: list[str]) -> SmilesTokenizer:
        """Build the vocabulary from a list of SMILES strings.

        Characters are added in sorted order for determinism.
        """
        chars = sorted({ch for s in smiles for ch in s})
        for ch in chars:
            if ch not in self.stoi:
                idx = len(self.stoi)
                self.stoi[ch] = idx
                self.itos[idx] = ch
        return self

    def encode(self, smiles: str, max_len: int | None = None) -> list[int]:
        """Encode one SMILES string to a list of token ids.

        Args:
            smiles: The SMILES string.
            max_len: If given, truncate to this length.

        Returns:
            A list of integer ids, unknown characters mapped to ``unk_id``.
        """
        ids = [self.stoi.get(ch, self.unk_id) for ch in smiles]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to a string, dropping padding."""
        return "".join(self.itos.get(i, UNK_TOKEN) for i in ids if i != self.pad_id)
