"""Download a public SMILES set to benchmark representations on.

The regression target is computed by RDKit at load time (logP, TPSA, or MW), so
only SMILES strings are needed. This script fetches the SMILES column of the
public MoleculeNet ESOL (Delaney) file and writes a one-column CSV. If the
mirrors are unreachable it falls back to the library's built-in SMILES list.

Usage:
    python scripts/download_data.py --out data/smiles.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import requests

from molprop.data import BUILTIN_SMILES

URLS = [
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
]


def fetch_smiles() -> list[str] | None:
    for url in URLS:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"  mirror failed ({url}): {exc}")
            continue
        reader = csv.DictReader(io.StringIO(resp.text))
        col = next((c for c in (reader.fieldnames or []) if "smiles" in c.lower()), None)
        if col is None:
            continue
        smiles = [row[col] for row in reader if row[col]]
        print(f"Downloaded {len(smiles)} SMILES from {url}")
        return smiles
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/smiles.csv")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    smiles = fetch_smiles()
    if smiles is None:
        print("All mirrors failed; writing the built-in SMILES list instead.")
        smiles = list(BUILTIN_SMILES)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles"])
        for s in smiles:
            writer.writerow([s])
    print(f"Wrote {len(smiles)} SMILES -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
