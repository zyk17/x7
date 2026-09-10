from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nn.train_config import load_train_config


def test_load_train_config_uses_fixed_contract_and_model_defaults(tmp_path: Path) -> None:
    path = tmp_path / "x7.yaml"
    path.write_text(
        """
dataset: {px0_version: "710"}
model: {}
training: {out: data/x7.pt}
""",
        encoding="utf-8",
    )
    args = load_train_config(path)
    assert (args.in_planes, args.num_moves) == (124, 2062)
    assert (args.width, args.blocks, args.heads, args.ffn_channels) == (512, 12, 16, 768)


def test_load_train_config_rejects_retired_model_variants(tmp_path: Path) -> None:
    path = tmp_path / "old.yaml"
    path.write_text(
        """
dataset: {px0_version: "710"}
model: {kind: retired}
training: {out: data/x7.pt}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="未知字段"):
        load_train_config(path)
