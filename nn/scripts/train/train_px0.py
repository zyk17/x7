#!/usr/bin/env python3
"""PX0 主线训练入口。"""

from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

NN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NN_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nn.dataset_px0 import Px0ChunkDataset, Px0DatasetConfig
from nn.px0_kaggle import load_prepared_px0_training_data
from nn.train_config import load_train_config
from nn.model import (
    PolicyResNet,
    mix_wdl_targets,
    moves_left_loss,
    soft_policy_cross_entropy,
    value_q_mse_from_wdl,
    value_wdl_cross_entropy,
)

from train_checkpoint import learning_rate_at_step, save_checkpoint, set_optimizer_learning_rate
from train_common import TRAIN_SEED, default_num_workers


OPTIMIZER_KIND = "sgd_nesterov"


def build_optimizer(model: PolicyResNet, *, learning_rate: float) -> SGD:
    """Match pxzero-training tfprocess.py:404-417: SGD, momentum 0.9, Nesterov."""
    return SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train small policy/value model on PX0 v6 chunks")
    ap.add_argument("--config", type=Path, required=True, help="唯一训练配置 YAML")
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if int(args.width) < 4 or int(args.width) % 2 != 0:
        raise SystemExit("model.width 须为不小于 4 的偶数")
    if int(args.blocks) < 3:
        raise SystemExit("model.blocks 须至少为 3，才能放置两次 Global Broadcast")
    if int(args.bottleneck_channels) < 1:
        raise SystemExit("model.bottleneck_channels 须为正整数")
    if not str(args.px0_version).strip():
        raise SystemExit("dataset.px0_version 不能为空")
    if not (0.0 < float(args.px0_val_ratio) < 1.0):
        raise SystemExit("dataset.val_ratio 须在 (0,1) 内")
    if int(args.validation_samples) < 3:
        raise SystemExit("dataset.validation_samples 须至少为 3")
    if int(args.validation_source_files) < 0:
        raise SystemExit("dataset.validation_source_files 须为非负整数")
    if not (0.0 <= float(args.q_ratio) <= 1.0):
        raise SystemExit("--q-ratio 须在 [0,1] 内")
    if float(args.moves_left_loss_weight) < 0.0:
        raise SystemExit("training.moves_left_loss_weight 须非负")
    if int(args.warmup_steps) < 0:
        raise SystemExit("training.warmup_steps 须非负")
    if int(args.shuffle_size) < 0:
        raise SystemExit("training.shuffle_size 须为非负整数")
    if not args.lr_values or any(value <= 0.0 for value in args.lr_values):
        raise SystemExit("training.lr_values 须为非空正数列表")
    if len(args.lr_values) != len(args.lr_boundaries) + 1:
        raise SystemExit("training.lr_values 长度必须等于 lr_boundaries 长度加一")
    invalid_boundaries = any(boundary <= 0 for boundary in args.lr_boundaries)
    sorted_unique_boundaries = tuple(sorted(set(args.lr_boundaries)))
    if invalid_boundaries or tuple(args.lr_boundaries) != sorted_unique_boundaries:
        raise SystemExit("training.lr_boundaries 须为严格递增的正整数列表")
    if args.init_from is not None and not Path(args.init_from).is_file():
        raise SystemExit(f"--init-from 文件不存在: {args.init_from}")


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit training device without silently changing a run."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("training.device=cuda，但 torch.cuda.is_available() 为 False；改为 cpu 或 auto 才会回退")
    return device


def take_logits_and_value(
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not isinstance(output, tuple):
        raise TypeError("train_px0 requires value_head=True")
    if len(output) == 2:
        logits, value = output
        return logits, value, None
    if len(output) == 3:
        logits, value, moves_left = output
        return logits, value, moves_left
    raise TypeError(f"unexpected model output len={len(output)}")


def checkpoint_bottleneck_channels(ckpt: dict) -> int:
    """Derive the original v2 default for checkpoints without explicit bt metadata."""
    return int(ckpt.get("bottleneck_channels", int(ckpt.get("width", 0)) * 7 // 16))


def validate_existing_output_checkpoint(
    ckpt: dict, *, width: int, blocks: int, bottleneck_channels: int
) -> None:
    if str(ckpt.get("trunk_kind")) != "x7_v2_bottleneck_gbroadcast":
        raise SystemExit("--out 已存在，但不是当前 x7_v2_bottleneck_gbroadcast 架构；请换新输出文件重新训练")
    if (
        int(ckpt.get("width", 0)) != width
        or int(ckpt.get("blocks", 0)) != blocks
        or checkpoint_bottleneck_channels(ckpt) != bottleneck_channels
    ):
        raise SystemExit(
            "checkpoint 的 width/blocks/bottleneck_channels 与当前 YAML 不一致；请使用相同配置续训，"
            "或换新输出文件开始实验"
        )
    if ckpt.get("moves_left_head") is not None and not bool(ckpt.get("moves_left_head", False)):
        raise SystemExit("--out 已存在，但不含当前 moves_left 辅助头；请换新输出文件重新训练")


def validate_existing_optimizer_checkpoint(ckpt: dict) -> None:
    if str(ckpt.get("optimizer_kind", "")) != OPTIMIZER_KIND:
        raise SystemExit(
            "--out 已存在，但 optimizer 不是当前 SGD+Nesterov；优化器 state 不兼容，请换新输出文件或用 --init-from"
        )


def next_qmix_phase(
    checkpoint: dict,
    *,
    q_ratio: float,
    completed_steps: int,
) -> tuple[bool, int, list[dict[str, float | int]]]:
    """Keep one checkpoint while making a qMix target change an explicit phase."""
    previous_q_ratio = checkpoint.get("q_ratio")
    previous_phase_start = int(checkpoint.get("phase_start_step", 0))
    raw_history = checkpoint.get("q_ratio_history", [])
    history = list(raw_history) if isinstance(raw_history, list) else []
    if not history and previous_q_ratio is not None:
        history.append(
            {
                "start_step": previous_phase_start,
                "q_ratio": float(previous_q_ratio),
            }
        )

    changed = previous_q_ratio is not None and not math.isclose(float(previous_q_ratio), float(q_ratio), abs_tol=1e-8)
    if changed:
        previous_phase_start = completed_steps
        history.append({"start_step": completed_steps, "q_ratio": float(q_ratio)})
    return changed, previous_phase_start, history


def run_val(
    model: PolicyResNet,
    loader: DataLoader,
    *,
    device: torch.device,
    value_loss_weight: float,
    moves_left_loss_weight: float,
    q_ratio: float,
) -> tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_policy = 0.0
    total_value_ce = 0.0
    total_value_q_mse = 0.0
    total_moves_left = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            boards = batch["board"].to(device=device, dtype=torch.float32)
            raw_policy = batch["policy"].to(device=device, dtype=torch.float32)
            winner_wdl = batch["winner_wdl"].to(device=device, dtype=torch.float32)
            search_wdl = batch["search_wdl"].to(device=device, dtype=torch.float32)
            plies_left = batch["plies_left"].to(device=device, dtype=torch.float32)
            legal_mask = raw_policy >= 0
            target_policy = raw_policy.clamp_min(0.0)
            target_value = mix_wdl_targets(winner_wdl, search_wdl, q_ratio=float(q_ratio))
            target_q = target_value[:, 0] - target_value[:, 2]
            output = model(boards)
            policy_logits, pred_value, pred_moves_left = take_logits_and_value(output)
            policy_loss = soft_policy_cross_entropy(policy_logits, target_policy, legal_mask)
            value_ce = value_wdl_cross_entropy(pred_value, target_value)
            value_q = value_q_mse_from_wdl(pred_value, target_q)
            moves_left = (
                moves_left_loss(pred_moves_left, plies_left)
                if pred_moves_left is not None
                else torch.zeros((), device=device)
            )
            loss = policy_loss + float(value_loss_weight) * value_ce + float(moves_left_loss_weight) * moves_left
            total_loss += float(loss.item())
            total_policy += float(policy_loss.item())
            total_value_ce += float(value_ce.item())
            total_value_q_mse += float(value_q.item())
            total_moves_left += float(moves_left.item())
            batches += 1
    if batches < 1:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    return (
        total_loss / batches,
        total_policy / batches,
        total_value_ce / batches,
        total_value_q_mse / batches,
        total_moves_left / batches,
    )


def build_dataset_configs(args: argparse.Namespace) -> tuple[Px0DatasetConfig, Px0DatasetConfig, Path]:
    prepared, validation_manifest = load_prepared_px0_training_data(
        args.px0_version,
        root=args.px0_root,
        val_ratio=float(args.px0_val_ratio),
        seed=int(args.px0_seed),
        validation_samples=int(args.validation_samples),
        validation_source_files=int(args.validation_source_files),
    )
    train_cfg = Px0DatasetConfig(
        file_list_path=prepared.train_manifest,
        shuffle_files=True,
        shuffle_size=int(args.shuffle_size),
        verify_files=False,
    )
    val_cfg = Px0DatasetConfig(
        sample_list_path=validation_manifest,
        shuffle_files=False,
    )
    return train_cfg, val_cfg, validation_manifest


def main() -> None:
    cli = build_parser().parse_args()
    try:
        args = load_train_config(cli.config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    validate_args(args)
    random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)

    device = resolve_device(args.device)
    print(f"torch {torch.__version__} | cuda.is_available={torch.cuda.is_available()} | device={device}")

    try:
        train_cfg, val_cfg, validation_manifest = build_dataset_configs(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    train_ds = Px0ChunkDataset(train_cfg)
    val_ds = Px0ChunkDataset(val_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        num_workers=default_num_workers() if args.num_workers is None else int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        num_workers=max(
            0,
            min(2, default_num_workers() if args.num_workers is None else int(args.num_workers)),
        ),
        pin_memory=device.type == "cuda",
    )

    model = PolicyResNet(
        in_planes=int(args.in_planes),
        width=int(args.width),
        num_blocks=int(args.blocks),
        num_moves=int(args.num_moves),
        bottleneck_channels=int(args.bottleneck_channels),
        value_head=True,
        moves_left_head=True,
    ).to(device)
    opt = build_optimizer(model, learning_rate=float(args.lr_values[0]))
    start_step = 0
    phase_start_step = 0
    best_val = float("inf")
    qmix_changed = False
    q_ratio_history: list[dict[str, float | int]] = [{"start_step": 0, "q_ratio": float(args.q_ratio)}]
    if args.out.is_file() and args.init_from is not None:
        raise SystemExit("--out 已存在时不要再传 --init-from；默认会直接从 --out 续训")

    if args.out.is_file():
        ckpt = torch.load(args.out, map_location=device)
        validate_existing_output_checkpoint(
            ckpt,
            width=int(args.width),
            blocks=int(args.blocks),
            bottleneck_channels=int(args.bottleneck_channels),
        )
        validate_existing_optimizer_checkpoint(ckpt)
        model.load_state_dict(ckpt["model"], strict=True)
        start_step = int(ckpt.get("completed_steps", 0))
        qmix_changed, phase_start_step, q_ratio_history = next_qmix_phase(
            ckpt,
            q_ratio=float(args.q_ratio),
            completed_steps=start_step,
        )
        if qmix_changed:
            # A different value target starts a fresh optimization/validation phase.
            opt = build_optimizer(model, learning_rate=float(args.lr_values[0]))
            print(
                f"qMix phase change at step={start_step}: "
                f"{float(ckpt['q_ratio']):.3f} -> {float(args.q_ratio):.3f}; "
                "reset optimizer, learning-rate phase, and best validation"
            )
        else:
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
            best_val = float(ckpt.get("best_val_loss", float("inf")))
        if start_step >= int(args.steps):
            raise SystemExit(
                f"--out 已完成到 step={start_step}，当前 --steps={int(args.steps)}；请增大 --steps 或换新文件名"
            )
        print(f"resume from {args.out} | completed_steps={start_step}")
    elif args.init_from is not None:
        ckpt = torch.load(args.init_from, map_location=device)
        validate_existing_output_checkpoint(
            ckpt,
            width=int(args.width),
            blocks=int(args.blocks),
            bottleneck_channels=int(args.bottleneck_channels),
        )
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"init from {args.init_from} | start new phase with q_ratio={float(args.q_ratio):.3f}")

    phase_steps = int(args.steps) - phase_start_step
    completed_phase_steps = start_step - phase_start_step
    if phase_steps <= 0 or completed_phase_steps < 0:
        raise SystemExit("当前 qMix phase 的 steps 必须大于其起始 step")
    if args.lr_boundaries and args.lr_boundaries[-1] >= phase_steps:
        raise SystemExit("training.lr_boundaries 必须小于当前 qMix phase 的总 step")

    print(
        f"px0: train_files={len(train_ds.files)} val_files={len(val_ds.files)} "
        f"batch_size={int(args.batch_size)} steps={int(args.steps)} "
        f"width={int(args.width)} blocks={int(args.blocks)} bt={int(args.bottleneck_channels)} "
        f"optimizer={OPTIMIZER_KIND} "
        f"q_ratio={float(args.q_ratio):.3f} phase_start={phase_start_step} "
        f"shuffle_size={int(args.shuffle_size)} validation_samples={int(args.validation_samples)}"
    )
    print(
        f"px0_kaggle: version={args.px0_version} root={args.px0_root.resolve()} "
        f"val_ratio={float(args.px0_val_ratio):.3f} validation_manifest={validation_manifest} "
        f"config={args.config_path}"
    )

    train_iter = iter(train_loader)
    for step in range(start_step + 1, int(args.steps) + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        boards = batch["board"].to(device=device, dtype=torch.float32)
        raw_policy = batch["policy"].to(device=device, dtype=torch.float32)
        winner_wdl = batch["winner_wdl"].to(device=device, dtype=torch.float32)
        search_wdl = batch["search_wdl"].to(device=device, dtype=torch.float32)
        plies_left = batch["plies_left"].to(device=device, dtype=torch.float32)
        legal_mask = raw_policy >= 0
        target_policy = raw_policy.clamp_min(0.0)
        target_value = mix_wdl_targets(winner_wdl, search_wdl, q_ratio=float(args.q_ratio))
        target_q = target_value[:, 0] - target_value[:, 2]

        phase_step = step - phase_start_step
        set_optimizer_learning_rate(
            opt,
            learning_rate_at_step(
                phase_step,
                values=args.lr_values,
                boundaries=args.lr_boundaries,
                warmup_steps=int(args.warmup_steps),
            ),
        )
        opt.zero_grad(set_to_none=True)
        output = model(boards)
        policy_logits, pred_value, pred_moves_left = take_logits_and_value(output)
        policy_loss = soft_policy_cross_entropy(policy_logits, target_policy, legal_mask)
        value_ce = value_wdl_cross_entropy(pred_value, target_value)
        value_q = value_q_mse_from_wdl(pred_value, target_q)
        moves_left = moves_left_loss(pred_moves_left, plies_left)
        loss = policy_loss + float(args.value_loss_weight) * value_ce + float(args.moves_left_loss_weight) * moves_left
        loss.backward()
        opt.step()
        if (
            step == 1
            or (qmix_changed and step == start_step + 1)
            or step % max(1, int(args.eval_every)) == 0
            or step == int(args.steps)
        ):
            val_loss, val_policy, val_value_ce, val_value_q_mse, val_moves_left = run_val(
                model,
                val_loader,
                device=device,
                value_loss_weight=float(args.value_loss_weight),
                moves_left_loss_weight=float(args.moves_left_loss_weight),
                q_ratio=float(args.q_ratio),
            )
            print(
                f"step {step}/{int(args.steps)} "
                f"train_loss={loss.item():.4f} "
                f"train_policy={policy_loss.item():.4f} "
                f"train_value_ce={value_ce.item():.4f} "
                f"train_value_q_mse={value_q.item():.4f} "
                f"train_moves_left={moves_left.item():.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_policy={val_policy:.4f} "
                f"val_value_ce={val_value_ce:.4f} "
                f"val_value_q_mse={val_value_q_mse:.4f} "
                f"val_moves_left={val_moves_left:.4f} "
                f"lr={opt.param_groups[0]['lr']:.2e}"
            )
            payload = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "optimizer_kind": OPTIMIZER_KIND,
                "width": int(args.width),
                "blocks": int(args.blocks),
                "bottleneck_channels": int(args.bottleneck_channels),
                "in_planes": int(args.in_planes),
                "n_moves": int(args.num_moves),
                "format": "px0_v6",
                "value_head": True,
                "moves_left_head": True,
                "trunk_kind": str(getattr(model, "trunk_kind", "x7_v2_bottleneck_gbroadcast")),
                "value_head_format": "wdl",
                "value_target_kind": "qmix_wdl",
                "q_ratio": float(args.q_ratio),
                "phase_start_step": phase_start_step,
                "q_ratio_history": q_ratio_history,
                "completed_steps": step,
                "best_val_loss": min(best_val, val_loss),
                "config_path": str(args.config_path),
                "run_name": str(args.name),
                "px0_version": str(args.px0_version),
                "px0_root": str(args.px0_root.resolve()),
                "px0_val_ratio": float(args.px0_val_ratio),
                "px0_seed": int(args.px0_seed),
                "validation_samples": int(args.validation_samples),
                "validation_source_files": int(args.validation_source_files),
                "validation_manifest": str(validation_manifest),
                "train_files": [str(p) for p in train_ds.files],
                "val_files": [str(p) for p in val_ds.files],
                "batch_size": int(args.batch_size),
                "shuffle_size": int(args.shuffle_size),
                "warmup_steps": int(args.warmup_steps),
                "lr_values": list(args.lr_values),
                "lr_boundaries": list(args.lr_boundaries),
                "value_loss_weight": float(args.value_loss_weight),
                "value_q_metric_target": "qmix_wdl",
                "moves_left_loss_weight": float(args.moves_left_loss_weight),
                "last_val_value_ce": float(val_value_ce),
                "last_val_value_q_mse": float(val_value_q_mse),
                "last_val_moves_left": float(val_moves_left),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            save_checkpoint(payload, args.out)
            if math.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                best_out = args.out.with_name(args.out.stem + ".best" + args.out.suffix)
                save_checkpoint(payload, best_out)


if __name__ == "__main__":
    main()
