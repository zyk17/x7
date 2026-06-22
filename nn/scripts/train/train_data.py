"""训练数据源解析与 DataLoader 构造。"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from nn.dataset_batch import collate_xrsh_samples
from nn.dataset_xrsh import MixedPolicyXrshDataset, PolicyXrshDataset
from nn.xrsh_io import xrsh_dir_is_complete

from train_common import GameGroupedBatchSampler, TRAIN_SEED, default_num_workers

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_train_mix(path: Path) -> list[tuple[Path, float]]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if "train_mix" in spec:
        spec = spec["train_mix"]
    sources = spec.get("sources") or spec.get("train_sources")
    if not sources:
        raise ValueError(f"{path} 缺少 sources 列表")
    out: list[tuple[Path, float]] = []
    for item in sources:
        d = Path(item["dir"])
        if not d.is_absolute():
            d = (PROJECT_ROOT / d).resolve()
        w = float(item.get("weight", 1.0))
        out.append((d, w))
    return out


def validate_train_args(parser: ArgumentParser, args: Namespace) -> None:
    if args.train_mix is None and args.train_dir is None:
        parser.error("需要 --train-dir 或 --train-mix")
    if args.train_mix is not None and args.train_dir is not None:
        parser.error("--train-dir 与 --train-mix 不可同时使用")
    if not xrsh_dir_is_complete(args.val_dir):
        raise FileNotFoundError(f"--val-dir 不完整: {args.val_dir}")


def build_datasets(
    args: Namespace,
    move_to_idx: dict[str, int],
    *,
    value_head: bool,
    search_policy_head: bool,
) -> tuple[PolicyXrshDataset | MixedPolicyXrshDataset, PolicyXrshDataset, str]:
    if args.train_mix is not None:
        mix_sources = load_train_mix(args.train_mix)
        for d, _ in mix_sources:
            if not xrsh_dir_is_complete(d):
                raise FileNotFoundError(f"train-mix 目录不完整: {d}")
        train_ds: PolicyXrshDataset | MixedPolicyXrshDataset = MixedPolicyXrshDataset(
            mix_sources,
            move_to_idx,
            for_training=True,
            with_value_labels=value_head,
            with_search_labels=search_policy_head,
            storage_mode=args.train_dataset_mode,
        )
        train_note = str(args.train_mix)
    else:
        assert args.train_dir is not None
        if not xrsh_dir_is_complete(args.train_dir):
            raise FileNotFoundError(f"--train-dir 不完整: {args.train_dir}")
        train_ds = PolicyXrshDataset(
            args.train_dir,
            move_to_idx,
            for_training=True,
            with_value_labels=value_head,
            with_search_labels=search_policy_head,
            storage_mode=args.train_dataset_mode,
        )
        train_note = str(args.train_dir)

    val_ds = PolicyXrshDataset(
        args.val_dir,
        move_to_idx,
        with_row_meta=True,
        with_value_labels=value_head,
        with_search_labels=search_policy_head,
        storage_mode=args.val_dataset_mode,
    )
    return train_ds, val_ds, train_note


def _loader_kw(device_type: str, num_workers: int) -> dict[str, Any]:
    out: dict[str, Any] = dict(
        num_workers=num_workers,
        pin_memory=device_type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    if num_workers == 0:
        out.pop("prefetch_factor", None)
        out.pop("persistent_workers", None)
    return out


def build_loaders(
    args: Namespace,
    train_ds: PolicyXrshDataset | MixedPolicyXrshDataset,
    val_ds: PolicyXrshDataset,
    *,
    device_type: str,
) -> tuple[DataLoader, DataLoader, GameGroupedBatchSampler]:
    train_bs = GameGroupedBatchSampler(
        batch_size=args.batch_size,
        row_group_ids=train_ds.row_group_ids,
        seed=TRAIN_SEED,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_bs,
        collate_fn=collate_xrsh_samples,
        **_loader_kw(device_type, int(args.train_num_workers)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_xrsh_samples,
        **_loader_kw(device_type, int(args.val_num_workers)),
    )
    return train_loader, val_loader, train_bs
