import pytest
import torch

from molprop.data import build_dataset, collate_batch, load_smiles_csv, random_split
from molprop.tokenizer import SmilesTokenizer


def test_tokenizer_roundtrip():
    tok = SmilesTokenizer().fit(["CCO", "c1ccccc1"])
    ids = tok.encode("CCO")
    assert tok.decode(ids) == "CCO"
    assert tok.pad_id == 0 and tok.unk_id == 1


def test_tokenizer_unknown_char_maps_to_unk():
    tok = SmilesTokenizer().fit(["CCO"])
    ids = tok.encode("CXO")  # X not in vocab
    assert tok.unk_id in ids


def test_tokenizer_truncation():
    tok = SmilesTokenizer().fit(["CCCCCC"])
    assert len(tok.encode("CCCCCC", max_len=3)) == 3


def test_build_dataset_shapes():
    ds = build_dataset(["CCO", "CCN", "c1ccccc1"], target="logp", n_bits=256)
    assert len(ds) == 3
    assert ds.fingerprints.shape == (3, 256)
    assert ds.targets.shape == (3,)


def test_build_dataset_skips_invalid():
    ds = build_dataset(["CCO", "not_a_mol", "CCN"], target="tpsa", n_bits=128)
    assert len(ds) == 2


def test_collate_pads_to_max_length():
    ds = build_dataset(["C", "CCCCCCC"], target="logp", n_bits=64)
    batch = collate_batch([ds[0], ds[1]])
    assert batch.ids.shape[0] == 2
    assert batch.ids.shape[1] == max(len(ds[0]["ids"]), len(ds[1]["ids"]))
    assert batch.fp.shape == (2, 64)
    assert torch.all(batch.lengths >= 1)


def test_random_split_shares_tokenizer():
    ds = build_dataset(target="logp", n_bits=128)
    tr, va, te = random_split(ds, seed=1)
    assert tr.tokenizer is ds.tokenizer
    assert va.tokenizer is ds.tokenizer
    assert len(tr) + len(va) + len(te) == len(ds)


def test_load_smiles_csv(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("name,smiles\na,CCO\nb,CCN\n", encoding="utf-8")
    smiles = load_smiles_csv(str(p))
    assert smiles == ["CCO", "CCN"]


def test_load_smiles_csv_missing_column(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_smiles_csv(str(p))
