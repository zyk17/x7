"""mmap JSONL 数据集与内存版逐样本一致性。"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nn import PolicyJsonlDataset, PolicyJsonlMmapDataset
from nn import build_jsonl_index, index_sampler_is_complete


def test_mmap_dataset_pickle_small_state(tmp_path: Path) -> None:
    """spawn worker 会 pickle Dataset；不得把整份 JSONL memmap 绑进 pickle。"""
    src = ROOT / "data" / "smock_train.jsonl"
    vocab_path = ROOT / "data" / "smock_vocab.json"
    if not src.is_file() or not vocab_path.is_file():
        pytest.skip("需要 data/smock_train.jsonl 与 data/smock_vocab.json")

    small = tmp_path / "tiny.jsonl"
    small.write_text(src.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    move_to_idx = {m: i for i, m in enumerate(vocab["moves"])}
    index_dir = tmp_path / "idx"
    build_jsonl_index(small, move_to_idx, index_dir, weight_by_fen=False)
    assert index_sampler_is_complete(index_dir)

    ds = PolicyJsonlMmapDataset(small, index_dir, move_to_idx, for_training=False)
    raw = pickle.dumps(ds)
    assert len(raw) < 2_000_000, f"pickle 过大: {len(raw)} bytes"


def test_mmap_matches_ram_rows(tmp_path: Path) -> None:
    src = ROOT / "data" / "smock_train.jsonl"
    vocab_path = ROOT / "data" / "smock_vocab.json"
    if not src.is_file() or not vocab_path.is_file():
        pytest.skip("需要 data/smock_train.jsonl 与 data/smock_vocab.json")

    small = tmp_path / "tiny.jsonl"
    lines = src.read_text(encoding="utf-8").splitlines()[:8]
    small.write_text("\n".join(lines) + "\n", encoding="utf-8")

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    move_to_idx = {m: i for i, m in enumerate(vocab["moves"])}

    index_dir = tmp_path / "idx"
    n = build_jsonl_index(small, move_to_idx, index_dir, weight_by_fen=False)
    assert n >= 1
    assert index_sampler_is_complete(index_dir)

    ram = PolicyJsonlDataset(small, move_to_idx, for_training=False)
    mmap_ds = PolicyJsonlMmapDataset(
        small, index_dir, move_to_idx, for_training=False
    )
    assert len(ram) == len(mmap_ds) == n

    for i in range(n):
        a = ram[i]
        b = mmap_ds[i]
        assert len(a) == len(b) == 4
        for ta, tb in zip(a, b):
            assert torch.equal(ta, tb)


def test_mmap_val_meta_matches(tmp_path: Path) -> None:
    src = ROOT / "data" / "smock_val.jsonl"
    vocab_path = ROOT / "data" / "smock_vocab.json"
    if not src.is_file() or not vocab_path.is_file():
        pytest.skip("需要 data/smock_val.jsonl 与 data/smock_vocab.json")

    small = tmp_path / "tiny_val.jsonl"
    lines = src.read_text(encoding="utf-8").splitlines()[:12]
    small.write_text("\n".join(lines) + "\n", encoding="utf-8")

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    move_to_idx = {m: i for i, m in enumerate(vocab["moves"])}

    index_dir = tmp_path / "idxv"
    build_jsonl_index(small, move_to_idx, index_dir, weight_by_fen=False)

    ram = PolicyJsonlDataset(
        small, move_to_idx, for_training=False, with_row_meta=True
    )
    mmap_ds = PolicyJsonlMmapDataset(
        small, index_dir, move_to_idx, for_training=False, with_row_meta=True
    )
    assert mmap_ds.pgn_source_vocab == ram.pgn_source_vocab

    for i in range(len(ram)):
        a = ram[i]
        b = mmap_ds[i]
        assert len(a) == len(b) == 6
        for ta, tb in zip(a, b):
            assert torch.equal(ta, tb)
