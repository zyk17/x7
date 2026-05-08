import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nn import PolicyJsonlMmapDataset
from nn.dataset_packed import PolicyPackedMmapDataset
from nn import build_jsonl_index, index_sampler_is_complete
from nn.materialize_pack import materialize_pack
from nn.policy_pack import PACK_META


def test_packed_matches_mmap(tmp_path: Path) -> None:
    src = ROOT / "data" / "smock_train.jsonl"
    vocab_path = ROOT / "data" / "smock_vocab.json"
    if not src.is_file() or not vocab_path.is_file():
        pytest.skip("需要 data/smock_train.jsonl 与 data/smock_vocab.json")

    small = tmp_path / "tiny.jsonl"
    lines = src.read_text(encoding="utf-8").splitlines()[:6]
    small.write_text("\n".join(lines) + "\n", encoding="utf-8")

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    move_to_idx = {m: i for i, m in enumerate(vocab["moves"])}

    idx_dir = tmp_path / "idx"
    build_jsonl_index(small, move_to_idx, idx_dir, weight_by_fen=False)
    assert index_sampler_is_complete(idx_dir)

    pack_dir = tmp_path / "pack"
    materialize_pack(small, idx_dir, vocab_path, pack_dir, show_progress=False)

    mmap_ds = PolicyJsonlMmapDataset(small, idx_dir, move_to_idx, for_training=False)
    pack_ds = PolicyPackedMmapDataset(pack_dir, move_to_idx, for_training=False)
    assert len(mmap_ds) == len(pack_ds)

    for i in range(len(mmap_ds)):
        a = mmap_ds[i]
        b = pack_ds[i]
        assert len(a) == len(b) == 4
        for ta, tb in zip(a, b):
            assert torch.equal(ta, tb)


def test_pack_rejects_vocab_fingerprint_mismatch(tmp_path: Path) -> None:
    src = ROOT / "data" / "smock_train.jsonl"
    vocab_path = ROOT / "data" / "smock_vocab.json"
    if not src.is_file() or not vocab_path.is_file():
        pytest.skip("需要 data/smock_train.jsonl 与 data/smock_vocab.json")

    small = tmp_path / "tiny.jsonl"
    small.write_text(src.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    move_to_idx = {m: i for i, m in enumerate(vocab["moves"])}

    idx_dir = tmp_path / "idx"
    build_jsonl_index(small, move_to_idx, idx_dir, weight_by_fen=False)
    pack_dir = tmp_path / "pack"
    materialize_pack(small, idx_dir, vocab_path, pack_dir, show_progress=False)

    meta = json.loads((pack_dir / PACK_META).read_text(encoding="utf-8"))
    meta["vocab_sha256"] = "0" * 64
    (pack_dir / PACK_META).write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="词表与训练包不一致"):
        PolicyPackedMmapDataset(pack_dir, move_to_idx, for_training=False)


def test_pack_rejects_missing_vocab_sha(tmp_path: Path) -> None:
    src = ROOT / "data" / "smock_train.jsonl"
    vocab_path = ROOT / "data" / "smock_vocab.json"
    if not src.is_file() or not vocab_path.is_file():
        pytest.skip("需要 data/smock_train.jsonl 与 data/smock_vocab.json")

    small = tmp_path / "tiny.jsonl"
    small.write_text(src.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    move_to_idx = {m: i for i, m in enumerate(vocab["moves"])}

    idx_dir = tmp_path / "idx2"
    build_jsonl_index(small, move_to_idx, idx_dir, weight_by_fen=False)
    pack_dir = tmp_path / "pack2"
    materialize_pack(small, idx_dir, vocab_path, pack_dir, show_progress=False)

    meta = json.loads((pack_dir / PACK_META).read_text(encoding="utf-8"))
    del meta["vocab_sha256"]
    (pack_dir / PACK_META).write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="vocab_sha256"):
        PolicyPackedMmapDataset(pack_dir, move_to_idx, for_training=False)
