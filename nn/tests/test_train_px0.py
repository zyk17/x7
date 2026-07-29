from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from train_checkpoint import learning_rate_at_step
from train_px0 import (
    OPTIMIZER_KIND,
    build_optimizer,
    build_dataset_configs,
    resolve_device,
    validate_args,
    validate_existing_output_checkpoint,
    validate_existing_optimizer_checkpoint,
)


def test_validate_existing_output_checkpoint_accepts_matching_v2_dimensions() -> None:
    validate_existing_output_checkpoint(
        {
            "trunk_kind": "x7_v2_bottleneck_gbroadcast",
            "width": 96,
            "blocks": 10,
            "bottleneck_channels": 64,
            "moves_left_head": True,
            "auxiliary_heads": True,
        },
        width=96,
        blocks=10,
        bottleneck_channels=64,
    )


def test_validate_existing_output_checkpoint_rejects_missing_bottleneck_width() -> None:
    with pytest.raises(SystemExit, match="width/blocks/bottleneck_channels"):
        validate_existing_output_checkpoint(
            {
                "trunk_kind": "x7_v2_bottleneck_gbroadcast",
                "width": 96,
                "blocks": 10,
                "moves_left_head": True,
                "auxiliary_heads": True,
            },
            width=96,
            blocks=10,
            bottleneck_channels=42,
        )


def test_validate_existing_output_checkpoint_rejects_mismatched_dimensions() -> None:
    with pytest.raises(SystemExit, match="width/blocks/bottleneck_channels"):
        validate_existing_output_checkpoint(
            {
                "trunk_kind": "x7_v2_bottleneck_gbroadcast",
                "width": 96,
                "blocks": 10,
                "bottleneck_channels": 64,
                "moves_left_head": True,
                "auxiliary_heads": True,
            },
            width=96,
            blocks=10,
            bottleneck_channels=48,
        )


def test_resolve_device_accepts_explicit_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_optimizer_uses_adamw_and_excludes_bias_from_weight_decay() -> None:
    import torch

    model = torch.nn.Linear(2, 1)
    optimizer = build_optimizer(model, learning_rate=0.001, weight_decay=0.0001)
    assert optimizer.defaults["weight_decay"] == 0.01
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.0001, 0.0]
    assert OPTIMIZER_KIND == "adamw"


def test_resume_rejects_sgd_checkpoint_state() -> None:
    with pytest.raises(SystemExit, match="AdamW"):
        validate_existing_optimizer_checkpoint({"optimizer_kind": "sgd_nesterov"})


def test_cosine_learning_rate_has_warmup_and_floor() -> None:
    assert learning_rate_at_step(1, total_steps=1_000, lr=1e-3, warmup_steps=250, min_lr_scale=0.1) == 4e-6
    assert learning_rate_at_step(250, total_steps=1_000, lr=1e-3, warmup_steps=250, min_lr_scale=0.1) == 1e-3
    assert learning_rate_at_step(1_000, total_steps=1_000, lr=1e-3, warmup_steps=250, min_lr_scale=0.1) == 1e-4
    assert learning_rate_at_step(2_000, total_steps=1_000, lr=1e-3, warmup_steps=250, min_lr_scale=0.1) == 1e-4


def test_validate_args_rejects_invalid_adamw_schedule() -> None:
    class Args:
        width = 256
        blocks = 12
        bottleneck_channels = 112
        px0_version = "710"
        px0_val_ratio = 0.1
        validation_samples = 8192
        validation_source_files = 64
        soft_policy_weight = 8.0
        soft_policy_temperature = 4.0
        final_value_loss_weight = 0.6
        root_wdl_loss_weight = 0.6
        moves_left_loss_weight = 1.0
        steps = 1_000
        warmup_steps = 250
        shuffle_size = 4096
        lr = 0.0
        min_lr_scale = 0.1
        weight_decay = 1e-4
        init_from = None

    with pytest.raises(SystemExit, match="AdamW"):
        validate_args(Args())


def test_validate_args_allows_non_baseline_v2_dimensions() -> None:
    class Args:
        width = 96
        blocks = 10
        bottleneck_channels = 48
        px0_version = "710"
        px0_val_ratio = 0.1
        validation_samples = 8192
        validation_source_files = 64
        soft_policy_weight = 8.0
        soft_policy_temperature = 4.0
        final_value_loss_weight = 0.6
        root_wdl_loss_weight = 0.6
        moves_left_loss_weight = 1.0
        steps = 1_000
        warmup_steps = 250
        shuffle_size = 4096
        lr = 3e-4
        min_lr_scale = 0.1
        weight_decay = 1e-4
        init_from = None

    validate_args(Args())


def test_build_dataset_configs_uses_prepared_data_loader(monkeypatch, tmp_path: Path) -> None:
    import train_px0
    from nn.px0_kaggle import PreparedPx0Version

    prepared = PreparedPx0Version(
        version="710",
        version_dir=tmp_path,
        chunk_files=[],
        train_manifest=tmp_path / "train.json",
        val_manifest=tmp_path / "val.json",
    )
    validation_manifest = tmp_path / "validation.json"
    captured = {}

    def _load(version, **kwargs):
        captured["version"] = version
        captured.update(kwargs)
        return prepared, validation_manifest

    monkeypatch.setattr(train_px0, "load_prepared_px0_training_data", _load)

    class Args:
        px0_version = "710"
        px0_root = tmp_path
        px0_val_ratio = 0.1
        px0_seed = 42
        validation_samples = 8192
        validation_source_files = 0
        shuffle_size = 4096

    train_cfg, val_cfg, manifest = build_dataset_configs(Args())
    assert captured["version"] == "710"
    assert captured["validation_samples"] == 8192
    assert train_cfg.file_list_path == prepared.train_manifest
    assert val_cfg.sample_list_path == validation_manifest
    assert manifest == validation_manifest
