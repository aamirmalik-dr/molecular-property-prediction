"""Dataset, batching, splits, and data sources for property prediction."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem

from molprop.featurize import compute_target, morgan_fingerprint
from molprop.tokenizer import SmilesTokenizer

# A built-in set of valid SMILES for offline runs and tests. Literal molecule
# strings, not a redistributed dataset.
BUILTIN_SMILES: list[str] = [
    "CCO", "CCN", "CCC", "CCCC", "CCCCC", "CCCCCC", "c1ccccc1", "c1ccccc1O",
    "c1ccccc1N", "CC(=O)O", "CC(=O)N", "CCOC(=O)C", "CC(C)O", "CC(C)C", "CCOCC",
    "CNC", "C1CCCCC1", "C1CCCC1", "c1ccncc1", "c1ccc2ccccc2c1", "CC(=O)Nc1ccccc1",
    "CCN(CC)CC", "OCC(O)CO", "CC(N)C(=O)O", "Cc1ccccc1", "Clc1ccccc1",
    "Brc1ccccc1", "Fc1ccccc1", "Oc1ccc(O)cc1", "Nc1ccc(N)cc1", "CC#N", "C=CC=C",
    "C#CC", "CCS", "CSC", "CC(=O)C", "CCC(=O)O", "CCCO", "CCCCO", "COC",
    "Cc1ccc(C)cc1", "c1ccc(cc1)C(=O)O", "NCCO", "OCCO", "CC(O)CO", "Cc1ccncc1",
    "Cn1cccc1", "c1cc[nH]c1", "c1ccoc1", "c1ccsc1", "CC(C)(C)O", "CCCCCCC",
    "CCCCCCCC", "CCCCCCCCO", "c1ccc(cc1)N", "c1ccc(cc1)Cl", "c1ccc(cc1)O",
    "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CCn1cnc2c1c(=O)n(C)c(=O)n2C",
    "OC(=O)c1ccccc1O", "NC(=O)c1ccccc1", "COc1ccccc1", "COc1ccc(cc1)C=O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CCCCCCCCCCCC", "OCCCCO", "NCCCCN", "CC(C)CC(C)C",
    "c1ccc(cc1)c1ccccc1", "O=C(O)CCC(=O)O", "O=C(O)/C=C/C(=O)O", "CC(=O)CC(=O)C",
    "CCOC(=O)CC", "CN(C)C=O", "CS(=O)(=O)C", "c1ccc(cc1)S", "CC(Cl)Cl", "ClCCl",
    "ClC(Cl)Cl", "FC(F)F", "CC(=O)Cl", "CCOCCO", "OCCOCCO", "c1ccc2[nH]ccc2c1",
    "c1ccc2ncccc2c1", "Cc1ccc(cc1)S(=O)(=O)N", "CC(N)Cc1ccccc1", "Oc1ccc(cc1)CCN",
    "CCCCN", "CCCCCN", "CC(C)(C)c1ccccc1", "c1ccc(cc1)C=C", "C=Cc1ccccc1",
    "Cc1ccccc1C", "Cc1cccc(C)c1", "Oc1ccccc1O", "Nc1ccccc1N", "CCc1ccccc1",
]


@dataclass
class PropertyDataset(torch.utils.data.Dataset):
    """SMILES paired with token ids, a fingerprint, and a scalar target."""

    smiles: list[str]
    token_ids: list[list[int]]
    fingerprints: np.ndarray
    targets: np.ndarray
    tokenizer: SmilesTokenizer

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
        return {
            "ids": self.token_ids[idx],
            "fp": self.fingerprints[idx],
            "y": float(self.targets[idx]),
        }


@dataclass
class Batch:
    """A padded batch shared by all three encoders."""

    ids: torch.Tensor       # (B, T) padded token ids
    lengths: torch.Tensor   # (B,) true sequence lengths
    fp: torch.Tensor        # (B, n_bits)
    y: torch.Tensor         # (B,)

    def to(self, device: str | torch.device) -> Batch:
        return Batch(
            self.ids.to(device),
            self.lengths.to(device),
            self.fp.to(device),
            self.y.to(device),
        )


def collate_batch(items: list[dict]) -> Batch:
    """Pad token id sequences to the batch maximum and stack tensors."""
    lengths = [len(it["ids"]) for it in items]
    max_len = max(lengths) if lengths else 1
    ids = np.zeros((len(items), max_len), dtype=np.int64)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = it["ids"]
    fp = np.stack([it["fp"] for it in items], axis=0)
    y = np.asarray([it["y"] for it in items], dtype=np.float32)
    return Batch(
        ids=torch.from_numpy(ids),
        lengths=torch.tensor(lengths, dtype=torch.long).clamp(min=1),
        fp=torch.from_numpy(fp),
        y=torch.from_numpy(y),
    )


def build_dataset(
    smiles: list[str] | None = None,
    target: str = "logp",
    n_bits: int = 1024,
    tokenizer: SmilesTokenizer | None = None,
) -> PropertyDataset:
    """Build a dataset from SMILES with an RDKit descriptor target.

    Invalid SMILES are skipped. A tokenizer is fit on the kept molecules unless
    one is supplied (useful to reuse a train vocabulary on a test split).
    """
    smiles = list(smiles) if smiles is not None else list(BUILTIN_SMILES)
    kept: list[str] = []
    targets: list[float] = []
    fps: list[np.ndarray] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        kept.append(smi)
        targets.append(compute_target(smi, target))
        fps.append(morgan_fingerprint(smi, n_bits=n_bits))

    if tokenizer is None:
        tokenizer = SmilesTokenizer().fit(kept)
    token_ids = [tokenizer.encode(s) for s in kept]
    return PropertyDataset(
        smiles=kept,
        token_ids=token_ids,
        fingerprints=np.stack(fps, axis=0),
        targets=np.asarray(targets, dtype=np.float32),
        tokenizer=tokenizer,
    )


def load_smiles_csv(csv_path: str, column: str | None = None) -> list[str]:
    """Read a SMILES column from a CSV file.

    Args:
        csv_path: Path to the CSV.
        column: Column name; if None, the first column whose header contains
            "smiles" (case-insensitive) is used.

    Raises:
        KeyError: If no suitable column is found.
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if column is None:
            for name in fields:
                if "smiles" in name.lower():
                    column = name
                    break
        if column is None or column not in fields:
            raise KeyError(f"no SMILES column found in {fields}")
        return [row[column] for row in reader if row[column]]


def random_split(
    dataset: PropertyDataset,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[PropertyDataset, PropertyDataset, PropertyDataset]:
    """Split into train/val/test. The train tokenizer is reused on all splits.

    Raises:
        ValueError: If fractions do not sum to 1.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1.0")
    n = len(dataset)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(fractions[0] * n)
    n_va = int(fractions[1] * n)
    splits = (perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :])
    out = []
    for idx in splits:
        out.append(
            PropertyDataset(
                smiles=[dataset.smiles[i] for i in idx],
                token_ids=[dataset.token_ids[i] for i in idx],
                fingerprints=dataset.fingerprints[idx],
                targets=dataset.targets[idx],
                tokenizer=dataset.tokenizer,
            )
        )
    return out[0], out[1], out[2]
