"""Molecular featurizers: Morgan fingerprints and RDKit descriptor targets."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, DataStructs, Descriptors, rdFingerprintGenerator

# Descriptors usable as regression targets, all computable by RDKit on any SMILES.
TARGET_FUNCTIONS = {
    "logp": Crippen.MolLogP,
    "tpsa": Descriptors.TPSA,
    "mw": Descriptors.MolWt,
}


def morgan_fingerprint(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    """Compute a binary Morgan fingerprint for a SMILES string.

    Raises:
        ValueError: If the SMILES cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_target(smiles: str, target: str) -> float:
    """Compute an RDKit descriptor target for a SMILES string.

    Raises:
        ValueError: If the target is unknown or the SMILES cannot be parsed.
    """
    if target not in TARGET_FUNCTIONS:
        raise ValueError(f"unknown target {target!r}; choose from {list(TARGET_FUNCTIONS)}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return float(TARGET_FUNCTIONS[target](mol))
