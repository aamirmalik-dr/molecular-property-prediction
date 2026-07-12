# Data

The regression target is computed by RDKit from each molecule's structure
(logP, TPSA, or molecular weight), so only SMILES strings are needed and every
benchmark is reproducible with no dataset license concern.

## Committed sample

`sample_smiles.csv` (300 rows) is committed and license-clean: the SMILES are a
carved subset of the public MoleculeNet ESOL (Delaney) SMILES column, and the
`logp` column is computed by RDKit (`Crippen.MolLogP`). It is a public subset,
not synthetic. `molprop.load_sample()` reads a copy of this file bundled inside
the package and drives the fully offline quickstart, examples, and tests.

Everything else in this directory (full downloads) stays gitignored.

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
