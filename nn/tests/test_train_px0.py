from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train import (
    OPTIMIZER_KIND,
    build_optimizer,
    build_dataset_configs,
    compute_loss_terms,
    learning_rate_at_step,
    validate_args,
    validate_existing_optimizer_checkpoint,
    validate_existing_output_checkpoint,
)


def test_checkpoint_dimensions_must_match() -> None:
    checkpoint = {
        "width": 96,
        "blocks": 10,
        "heads": 4,
        "ffn_channels": 144,
        "moves_left_head": True,
        "auxiliary_heads": True,
    }
    validate_existing_output_checkpoint(checkpoint, width=96, blocks=10, heads=4, ffn_channels=144)
    with pytest.raises(SystemExit, match="width/blocks/heads/ffn_channels"):
        validate_existing_output_checkpoint(checkpoint, width=96, blocks=10, heads=8, ffn_channels=144)


def test_optimizer_and_schedule() -> None:
    import torch

    optimizer = build_optimizer(torch.nn.Linear(2, 1), learning_rate=0.001, weight_decay=0.0001)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.0001, 0.0]
    assert OPTIMIZER_KIND == "adamw"
    assert learning_rate_at_step(1, total_steps=1_000, lr=1e-3, warmup_steps=250, min_lr_scale=0.1) == 4e-6
    assert learning_rate_at_step(1_000, total_steps=1_000, lr=1e-3, warmup_steps=250, min_lr_scale=0.1) == 1e-4


def test_training_loss_and_argument_validation() -> None:
    import torch

    class Args:
        width = 32
        blocks = 2
        heads = 4
        ffn_channels = 96
        px0_version = "710"
        px0_val_ratio = 0.1
        soft_policy_weight = 8.0
        soft_policy_temperature = 4.0
        final_value_loss_weight = 0.6
        root_wdl_loss_weight = 0.6
        moves_left_loss_weight = 0.5
        steps = 1_000
        warmup_steps = 250
        shuffle_size = 4096
        full_validation_every = 200_000
        lr = 3e-4
        min_lr_scale = 0.1
        weight_decay = 1e-4
        init_from = None

    validate_args(Args())
    terms = compute_loss_terms(
        (torch.zeros((1, 3)), torch.zeros((1, 3)), torch.zeros((1, 1)), torch.zeros((1, 3)), torch.zeros((1, 3))),
        raw_policy=torch.tensor([[-1.0, 0.2, 0.8]]),
        winner_wdl=torch.tensor([[0.0, 1.0, 0.0]]),
        root_wdl=torch.tensor([[0.0, 1.0, 0.0]]),
        plies_left=torch.zeros((1, 1)),
        final_value_loss_weight=0.6,
        moves_left_loss_weight=0.5,
        soft_policy_weight=8.0,
        soft_policy_temperature=4.0,
        root_wdl_loss_weight=0.6,
    )
    assert torch.isfinite(terms["total"])


def test_resume_rejects_other_optimizer() -> None:
    with pytest.raises(SystemExit, match="AdamW"):
        validate_existing_optimizer_checkpoint({"optimizer_kind": "sgd"})


def test_build_dataset_configs_uses_prepared_data_loader(monkeypatch, tmp_path: Path) -> None:
    import train
    from nn.px0_kaggle import PreparedPx0Version

    prepared = PreparedPx0Version("710", tmp_path, [], tmp_path / "train.json", tmp_path / "val.json")

    def load(*_args, **_kwargs):
        return prepared, prepared.val_manifest

    monkeypatch.setattr(train, "load_prepared_px0_training_data", load)

    class Args:
        px0_version = "710"
        px0_root = tmp_path
        px0_val_ratio = 0.1
        px0_seed = 42
        shuffle_size = 4096

    train, val, full, _ = build_dataset_configs(Args())
    assert (train.sample_rate, val.sample_rate, full.sample_rate) == (32, 32, 1)
