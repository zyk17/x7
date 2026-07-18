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
    next_qmix_phase,
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
        },
        width=96,
        blocks=10,
        bottleneck_channels=64,
    )


def test_validate_existing_output_checkpoint_derives_legacy_bottleneck_width() -> None:
    validate_existing_output_checkpoint(
        {
            "trunk_kind": "x7_v2_bottleneck_gbroadcast",
            "width": 96,
            "blocks": 10,
            "moves_left_head": True,
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
            },
            width=96,
            blocks=10,
            bottleneck_channels=48,
        )


def test_resolve_device_accepts_explicit_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_optimizer_matches_pxzero_sgd_nesterov() -> None:
    import torch

    model = torch.nn.Linear(2, 1)
    optimizer = build_optimizer(model, learning_rate=0.001)
    assert optimizer.defaults["momentum"] == 0.9
    assert optimizer.defaults["nesterov"] is True
    assert optimizer.defaults["weight_decay"] == 0
    assert OPTIMIZER_KIND == "sgd_nesterov"


def test_resume_rejects_adamw_checkpoint_state() -> None:
    with pytest.raises(SystemExit, match=r"SGD\+Nesterov"):
        validate_existing_optimizer_checkpoint({"optimizer_kind": "adamw"})


def test_qmix_change_starts_a_new_phase() -> None:
    changed, phase_start, history = next_qmix_phase(
        {"q_ratio": 0.0, "completed_steps": 200_000},
        q_ratio=0.75,
        completed_steps=200_000,
    )
    assert changed
    assert phase_start == 200_000
    assert history == [
        {"start_step": 0, "q_ratio": 0.0},
        {"start_step": 200_000, "q_ratio": 0.75},
    ]


def test_piecewise_learning_rate_has_phase_local_warmup() -> None:
    values = (1e-3, 3e-4, 1e-4)
    boundaries = (140_000, 180_000)
    assert learning_rate_at_step(1, values=values, boundaries=boundaries, warmup_steps=250) == 4e-6
    assert learning_rate_at_step(250, values=values, boundaries=boundaries, warmup_steps=250) == 1e-3
    assert learning_rate_at_step(140_001, values=values, boundaries=boundaries, warmup_steps=250) == 3e-4
    assert learning_rate_at_step(180_001, values=values, boundaries=boundaries, warmup_steps=250) == 1e-4


def test_validate_args_rejects_invalid_lr_schedule() -> None:
    class Args:
        width = 256
        blocks = 12
        bottleneck_channels = 112
        px0_version = "710"
        px0_val_ratio = 0.1
        validation_samples = 8192
        validation_source_files = 64
        q_ratio = 0.0
        moves_left_loss_weight = 1.0
        warmup_steps = 250
        shuffle_size = 4096
        lr_values = (1e-3,)
        lr_boundaries = (100,)
        init_from = None

    with pytest.raises(SystemExit, match="lr_values"):
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
        q_ratio = 0.0
        moves_left_loss_weight = 1.0
        warmup_steps = 250
        shuffle_size = 4096
        lr_values = (1e-3,)
        lr_boundaries = ()
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
