"""PX0 主线训练的单一 YAML 配置读取。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .model_common import CNN_TRUNK_KIND, TRANSFORMER_TRUNK_KIND
from typing import Any

import yaml

from nn.px0_kaggle import DEFAULT_PX0_ROOT


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是 YAML mapping")
    return value


def _reject_unknown(section: dict[str, Any], *, name: str, allowed: set[str]) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"{name} 包含未知字段: {', '.join(unknown)}")


def _required(section: dict[str, Any], key: str, *, name: str) -> Any:
    if key not in section:
        raise ValueError(f"{name}.{key} 是必填字段")
    return section[key]


def load_train_config(path: Path | str) -> argparse.Namespace:
    """Load the fixed PX0 training surface from one YAML file.

    The section layout follows pxzero-training's `dataset/training/model`
    configuration convention. Paths intentionally stay relative to the process
    working directory, so the copied `nn/` directory remains self-contained.
    Reference: pxzero-training `tf/train.py:110-126`, `tf/configs/example.yaml:4-31`.
    """
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"无法读取训练配置: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 语法错误: {config_path}") from exc

    config = _mapping(raw, name="root")
    _reject_unknown(config, name="root", allowed={"name", "dataset", "model", "training"})
    dataset = _mapping(_required(config, "dataset", name="root"), name="dataset")
    model = _mapping(_required(config, "model", name="root"), name="model")
    training = _mapping(_required(config, "training", name="root"), name="training")

    _reject_unknown(
        dataset,
        name="dataset",
        allowed={
            "px0_version",
            "px0_root",
            "val_ratio",
            "validation_samples",
            "validation_source_files",
            "seed",
            "force_download",
        },
    )
    _reject_unknown(
        model,
        name="model",
        allowed={"kind", "width", "blocks", "bottleneck_channels", "heads", "ffn_channels"},
    )
    _reject_unknown(
        training,
        name="training",
        allowed={
            "out",
            "init_from",
            "batch_size",
            "steps",
            "eval_every",
            "shuffle_size",
            "warmup_steps",
            "lr",
            "min_lr_scale",
            "weight_decay",
            "final_value_loss_weight",
            "root_wdl_loss_weight",
            "moves_left_loss_weight",
            "soft_policy_weight",
            "soft_policy_temperature",
            "amp",
            "device",
            "num_workers",
        },
    )

    model_kind = str(model.get("kind", TRANSFORMER_TRUNK_KIND))
    if model_kind not in (CNN_TRUNK_KIND, TRANSFORMER_TRUNK_KIND):
        raise ValueError(f"model.kind 只支持 {CNN_TRUNK_KIND} 或 {TRANSFORMER_TRUNK_KIND}")
    width = int(model.get("width", 512 if model_kind == TRANSFORMER_TRUNK_KIND else 384))
    return argparse.Namespace(
        config_path=config_path.resolve(),
        name=str(config.get("name", config_path.stem)),
        px0_version=str(_required(dataset, "px0_version", name="dataset")),
        px0_root=Path(dataset.get("px0_root", DEFAULT_PX0_ROOT)),
        px0_val_ratio=float(dataset.get("val_ratio", 0.1)),
        validation_samples=int(dataset.get("validation_samples", 8192)),
        validation_source_files=int(dataset.get("validation_source_files", 0)),
        px0_seed=int(dataset.get("seed", 42)),
        px0_force_download=bool(dataset.get("force_download", False)),
        out=Path(_required(training, "out", name="training")),
        init_from=Path(training["init_from"]) if training.get("init_from") else None,
        width=width,
        blocks=int(model.get("blocks", 12 if model_kind == TRANSFORMER_TRUNK_KIND else 15)),
        bottleneck_channels=int(model.get("bottleneck_channels", width // 2)),
        model_kind=model_kind,
        heads=int(model.get("heads", 16)),
        ffn_channels=int(model.get("ffn_channels", width * 3 // 2)),
        in_planes=124,
        num_moves=2062,
        batch_size=int(training.get("batch_size", 256)),
        steps=int(training.get("steps", 200_000)),
        eval_every=int(training.get("eval_every", 1_000)),
        shuffle_size=int(training.get("shuffle_size", 4_096)),
        warmup_steps=int(training.get("warmup_steps", 250)),
        lr=float(training.get("lr", 3e-4)),
        min_lr_scale=float(training.get("min_lr_scale", 0.1)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        final_value_loss_weight=float(training.get("final_value_loss_weight", 0.6)),
        root_wdl_loss_weight=float(training.get("root_wdl_loss_weight", 0.6)),
        moves_left_loss_weight=float(training.get("moves_left_loss_weight", 0.5)),
        soft_policy_weight=float(training.get("soft_policy_weight", 8.0)),
        soft_policy_temperature=float(training.get("soft_policy_temperature", 4.0)),
        amp=bool(training.get("amp", True)),
        device=str(training.get("device", "cuda")),
        num_workers=training.get("num_workers"),
    )
