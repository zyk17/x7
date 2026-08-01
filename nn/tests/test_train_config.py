from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nn.train_config import load_train_config


def test_load_train_config_uses_fixed_px0_contract(tmp_path: Path) -> None:
    path = tmp_path / "x7.yaml"
    path.write_text(
        """
name: x7_test
dataset:
  px0_version: "710"
model:
  width: 256
  blocks: 12
  bottleneck_channels: 160
training:
  out: data/x7.pt
  lr: 0.0003
  final_value_loss_weight: 0.6
  root_wdl_loss_weight: 0.6
""",
        encoding="utf-8",
    )

    args = load_train_config(path)

    assert args.name == "x7_test"
    assert args.px0_version == "710"
    assert args.in_planes == 124
    assert args.num_moves == 2062
    assert args.out == Path("data/x7.pt")
    assert args.lr == 0.0003
    assert args.final_value_loss_weight == 0.6
    assert args.root_wdl_loss_weight == 0.6
    assert args.soft_policy_temperature == 4.0
    assert args.moves_left_loss_weight == 0.5
    assert args.width == 256
    assert args.blocks == 12
    assert args.bottleneck_channels == 160
    assert args.min_lr_scale == 0.1
    assert args.validation_samples == 8192
    assert args.shuffle_size == 4096


def test_load_train_config_rejects_unknown_training_option(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
dataset: {px0_version: "710"}
model: {}
training: {out: data/x7.pt, legacy_resume: true}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="未知字段"):
        load_train_config(path)


def test_load_train_config_defaults_to_v3_attentionbody(tmp_path: Path) -> None:
    path = tmp_path / "default.yaml"
    path.write_text(
        """
dataset: {px0_version: "710"}
model: {}
training: {out: data/x7.pt}
""",
        encoding="utf-8",
    )
    args = load_train_config(path)
    assert (args.model_kind, args.blocks, args.width, args.bottleneck_channels) == (
        "x7_v3_attentionbody",
        12,
        512,
        256,
    )


def test_load_transformer_v3_config(tmp_path: Path) -> None:
    path = tmp_path / "transformer.yaml"
    path.write_text(
        """
name: v3
dataset: {px0_version: "677"}
model: {kind: x7_v3_attentionbody, width: 32, blocks: 2, heads: 4, ffn_channels: 96}
training: {out: data/v3.pt}
""",
        encoding="utf-8",
    )
    args = load_train_config(path)
    assert (args.model_kind, args.width, args.blocks, args.heads, args.ffn_channels) == (
        "x7_v3_attentionbody",
        32,
        2,
        4,
        96,
    )


def test_load_train_config_accepts_adamw_weight_decay(tmp_path: Path) -> None:
    path = tmp_path / "adamw.yaml"
    path.write_text(
        """
dataset: {px0_version: "710"}
model: {}
training: {out: data/x7.pt, weight_decay: 0.0001}
""",
        encoding="utf-8",
    )

    assert load_train_config(path).weight_decay == 0.0001
