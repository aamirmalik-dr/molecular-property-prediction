# Data

This directory is gitignored. No datasets are committed.

The regression target is computed by RDKit from each molecule's structure
(logP, TPSA, or molecular weight), so only SMILES strings are needed and every
benchmark is reproducible with no dataset license concern.

## Downloaded SMILES

`scripts/download_data.py` fetches the SMILES column of the public MoleculeNet
ESOL (Delaney) file and writes a one-column CSV:

```bash
python scripts/download_data.py --out data/smiles.csv
```

If the mirrors are unreachable it writes the library's built-in SMILES list
instead, and says so.

## Offline

The unit tests and the default `benchmark.py` invocation need no download at
all: they build the dataset from a built-in list of valid SMILES and compute the
target with RDKit.
