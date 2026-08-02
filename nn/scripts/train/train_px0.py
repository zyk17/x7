#!/usr/bin/env python3
"""PX0 主线训练入口：正式 heads + 仅训练期辅助 heads。"""

from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

NN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NN_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nn.dataset_px0 import Px0ChunkDataset, Px0DatasetConfig
from nn.model import (
    CNN_TRUNK_KIND,
    KnowledgeResNet,
    KnowledgeTransformer,
    build_model,
    moves_left_loss,
    soften_policy_targets,
    soft_policy_cross_entropy,
    value_q_mse_from_wdl,
    value_wdl_cross_entropy,
)
from nn.px0_kaggle import load_prepared_px0_training_data
from nn.train_config import load_train_config
from train_checkpoint import (
    learning_rate_at_step,
    save_checkpoint,
    set_optimizer_learning_rate,
)
from train_common import TRAIN_SEED, default_num_workers


OPTIMIZER_KIND = "adamw"
KnowledgeModel = KnowledgeResNet | KnowledgeTransformer


def build_optimizer(model: torch.nn.Module, *, learning_rate: float, weight_decay: float) -> AdamW:
    """Apply decoupled decay only to convolutional and linear weights."""
    decay = [param for param in model.parameters() if param.ndim >= 2]
    no_decay = [param for param in model.parameters() if param.ndim < 2]
    return AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train policy/value model on PX0 chunks")
    ap.add_argument("--config", type=Path, required=True, help="唯一训练配置 YAML")
    return ap


def validate_args(args: argparse.Namespace) -> None:
    model_kind = getattr(args, "model_kind", CNN_TRUNK_KIND)
    if model_kind == CNN_TRUNK_KIND:
        if int(args.width) < 4 or int(args.width) % 2 != 0 or int(args.blocks) < 3 or int(args.bottleneck_channels) < 1:
            raise SystemExit("CNN model.width/blocks/bottleneck_channels 非法")
    elif (
        int(args.width) < 4
        or int(args.blocks) < 1
        or int(getattr(args, "heads", 16)) < 1
        or int(getattr(args, "ffn_channels", int(args.width) * 3 // 2)) < int(args.width)
        or int(args.width) % int(getattr(args, "heads", 16)) != 0
    ):
        raise SystemExit("Transformer model.width/blocks/heads/ffn_channels 非法")
    if not str(args.px0_version).strip() or not (0.0 < float(args.px0_val_ratio) < 1.0):
        raise SystemExit("dataset.px0_version 或 dataset.val_ratio 非法")
    if int(args.shuffle_size) < 0 or int(args.full_validation_every) < 1:
        raise SystemExit("training shuffle/validation 配置非法")
    if (
        float(args.final_value_loss_weight) < 0.0
        or float(args.root_wdl_loss_weight) < 0.0
        or float(args.moves_left_loss_weight) < 0.0
    ):
        raise SystemExit("training loss weight 须非负")
    if float(args.soft_policy_weight) < 0.0 or float(args.soft_policy_temperature) <= 0.0:
        raise SystemExit("training soft policy 配置非法")
    if int(args.steps) < 1 or int(args.warmup_steps) < 0 or int(args.warmup_steps) > int(args.steps):
        raise SystemExit("training.steps/warmup_steps 非法")
    if float(args.lr) <= 0.0 or not 0.0 <= float(args.min_lr_scale) <= 1.0 or float(args.weight_decay) < 0.0:
        raise SystemExit("training AdamW/cosine 配置非法")
    if args.init_from is not None and not Path(args.init_from).is_file():
        raise SystemExit(f"--init-from 文件不存在: {args.init_from}")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("training.device=cuda，但 CUDA 不可用")
    return device


def take_training_outputs(
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(output, tuple) or len(output) != 5:
        raise TypeError("train_px0 requires formal heads plus two auxiliary heads")
    return output  # type: ignore[return-value]


def forward_training(
    model: KnowledgeModel, boards: torch.Tensor, *, amp_enabled: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use FP16 for the spatial trunk and explicit FP32 heads/losses.

    This follows PyTorch AMP's mixed-precision pattern while preserving stable
    policy/value reductions in FP32. KataGoMethods.md motivates the two
    training-only heads; their tensors never enter the exporter.
    """
    if amp_enabled:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            trunk = model.forward_trunk(boards)
        with torch.amp.autocast("cuda", enabled=False):
            return take_training_outputs(model.forward_heads(trunk.float()))
    return take_training_outputs(model(boards))


def compute_loss_terms(
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    raw_policy: torch.Tensor,
    winner_wdl: torch.Tensor,
    root_wdl: torch.Tensor,
    plies_left: torch.Tensor,
    final_value_loss_weight: float,
    moves_left_loss_weight: float,
    soft_policy_weight: float,
    soft_policy_temperature: float,
    root_wdl_loss_weight: float,
) -> dict[str, torch.Tensor]:
    policy_logits, final_value, pred_moves_left, soft_logits, root_value = output
    legal_mask = raw_policy >= 0
    target_policy = raw_policy.clamp_min(0.0)
    policy = soft_policy_cross_entropy(policy_logits, target_policy, legal_mask)
    soft_target = soften_policy_targets(target_policy, legal_mask, temperature=soft_policy_temperature)
    soft_policy_ce = soft_policy_cross_entropy(soft_logits, soft_target, legal_mask)
    soft_target_entropy = (
        -torch.where(
            soft_target > 0,
            soft_target * soft_target.log(),
            torch.zeros_like(soft_target),
        )
        .sum(dim=1)
        .mean()
    )
    final_ce = value_wdl_cross_entropy(final_value, winner_wdl)
    root_ce = value_wdl_cross_entropy(root_value, root_wdl)
    moves = moves_left_loss(pred_moves_left, plies_left)
    model_loss = policy + final_value_loss_weight * final_ce + moves_left_loss_weight * moves
    return {
        "total": model_loss + soft_policy_weight * soft_policy_ce + root_wdl_loss_weight * root_ce,
        "formal": model_loss,
        "policy": policy,
        "soft_policy_kl": (soft_policy_ce - soft_target_entropy).clamp_min(0.0),
        "final_value_ce": final_ce,
        "final_value_q_mse": value_q_mse_from_wdl(final_value, winner_wdl[:, 0] - winner_wdl[:, 2]),
        "root_value_ce": root_ce,
        "moves_left": moves,
    }


def validate_existing_output_checkpoint(
    ckpt: dict,
    *,
    width: int,
    blocks: int,
    bottleneck_channels: int,
    trunk_kind: str = CNN_TRUNK_KIND,
    heads: int = 16,
    ffn_channels: int = 0,
) -> None:
    if str(ckpt.get("trunk_kind")) != trunk_kind:
        raise SystemExit("--out 已存在，但模型架构与当前 YAML 不一致；请换新输出文件")
    if trunk_kind != CNN_TRUNK_KIND and ckpt.get("model_format") != "px0_attentionbody_v1":
        raise SystemExit("--out 是旧 v3 AttentionBody checkpoint；请换新输出文件")
    keys = (
        ("width", "blocks", "bottleneck_channels")
        if trunk_kind == CNN_TRUNK_KIND
        else (
            "width",
            "blocks",
            "heads",
            "ffn_channels",
        )
    )
    expected = (
        (width, blocks, bottleneck_channels) if trunk_kind == CNN_TRUNK_KIND else (width, blocks, heads, ffn_channels)
    )
    actual = tuple(int(ckpt.get(key, 0)) for key in keys)
    if actual != expected:
        if trunk_kind == CNN_TRUNK_KIND:
            raise SystemExit("checkpoint 的 width/blocks/bottleneck_channels 与当前 YAML 不一致")
        raise SystemExit("checkpoint 的 Transformer 模型尺寸与当前 YAML 不一致")
    if not bool(ckpt.get("moves_left_head")) or not bool(ckpt.get("auxiliary_heads")):
        raise SystemExit("--out 已存在，但不含当前训练辅助头；请换新输出文件")


def validate_existing_optimizer_checkpoint(ckpt: dict) -> None:
    if str(ckpt.get("optimizer_kind", "")) != OPTIMIZER_KIND:
        raise SystemExit("--out 已存在，但 optimizer 不是当前 AdamW")


def run_val(
    model: KnowledgeModel,
    loader: DataLoader,
    *,
    device: torch.device,
    final_value_loss_weight: float,
    moves_left_loss_weight: float,
    soft_policy_weight: float,
    soft_policy_temperature: float,
    root_wdl_loss_weight: float,
    amp_enabled: bool,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    sums = {
        name: 0.0
        for name in (
            "total",
            "formal",
            "policy",
            "soft_policy_kl",
            "final_value_ce",
            "final_value_q_mse",
            "root_value_ce",
            "moves_left",
        )
    }
    batches = 0
    with torch.no_grad():
        for batch in loader:
            boards = batch["board"].to(device=device, dtype=torch.float32)
            raw_policy = batch["policy"].to(device=device, dtype=torch.float32)
            winner_wdl = batch["winner_wdl"].to(device=device, dtype=torch.float32)
            root_wdl = batch["root_wdl"].to(device=device, dtype=torch.float32)
            plies_left = batch["plies_left"].to(device=device, dtype=torch.float32)
            terms = compute_loss_terms(
                forward_training(model, boards, amp_enabled=amp_enabled),
                raw_policy=raw_policy,
                winner_wdl=winner_wdl,
                root_wdl=root_wdl,
                plies_left=plies_left,
                final_value_loss_weight=final_value_loss_weight,
                moves_left_loss_weight=moves_left_loss_weight,
                soft_policy_weight=soft_policy_weight,
                soft_policy_temperature=soft_policy_temperature,
                root_wdl_loss_weight=root_wdl_loss_weight,
            )
            for name, value in terms.items():
                sums[name] += float(value.item())
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    return {name: total / batches if batches else float("nan") for name, total in sums.items()}


def build_dataset_configs(
    args: argparse.Namespace,
) -> tuple[Px0DatasetConfig, Px0DatasetConfig, Px0DatasetConfig, Path]:
    prepared, validation_manifest = load_prepared_px0_training_data(
        args.px0_version,
        root=args.px0_root,
        val_ratio=float(args.px0_val_ratio),
        seed=int(args.px0_seed),
    )
    return (
        Px0DatasetConfig(
            file_list_path=prepared.train_manifest,
            shuffle_files=True,
            shuffle_size=int(args.shuffle_size),
            sample_rate=32,
            verify_files=False,
        ),
        Px0DatasetConfig(
            file_list_path=validation_manifest,
            shuffle_files=True,
            shuffle_size=max(1, int(args.shuffle_size * args.px0_val_ratio)),
            sample_rate=32,
            verify_files=False,
        ),
        Px0DatasetConfig(file_list_path=validation_manifest, verify_files=False),
        validation_manifest,
    )


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
    amp_enabled = bool(args.amp) and device.type == "cuda"
    print(f"torch {torch.__version__} | device={device} | trunk_amp={amp_enabled}")

    try:
        train_cfg, val_cfg, full_val_cfg, validation_manifest = build_dataset_configs(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    train_ds = Px0ChunkDataset(train_cfg)
    val_ds = Px0ChunkDataset(val_cfg)
    full_val_ds = Px0ChunkDataset(full_val_cfg)
    workers = default_num_workers() if args.num_workers is None else int(args.num_workers)
    train_loader = DataLoader(
        train_ds, batch_size=int(args.batch_size), num_workers=workers, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(args.batch_size), num_workers=max(0, min(2, workers)), pin_memory=device.type == "cuda"
    )
    full_val_loader = DataLoader(
        full_val_ds,
        batch_size=int(args.batch_size),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    trunk_kind = str(args.model_kind)
    model = build_model(
        trunk_kind=trunk_kind,
        in_planes=int(args.in_planes),
        width=int(args.width),
        blocks=int(args.blocks),
        num_moves=int(args.num_moves),
        bottleneck_channels=int(args.bottleneck_channels),
        heads=int(args.heads),
        ffn_channels=int(args.ffn_channels),
        value_head=True,
        moves_left_head=True,
        auxiliary_heads=True,
    ).to(device)
    opt = build_optimizer(model, learning_rate=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_step, best_full_formal = 0, float("inf")
    lr_decay_steps = int(args.steps)
    if args.out.is_file() and args.init_from is not None:
        raise SystemExit("--out 已存在时不要再传 --init-from")
    if args.out.is_file() or args.init_from is not None:
        source = args.out if args.out.is_file() else args.init_from
        assert source is not None
        ckpt = torch.load(source, map_location=device)
        validate_existing_output_checkpoint(
            ckpt,
            trunk_kind=trunk_kind,
            width=int(args.width),
            blocks=int(args.blocks),
            bottleneck_channels=int(args.bottleneck_channels),
            heads=int(args.heads),
            ffn_channels=int(args.ffn_channels),
        )
        model.load_state_dict(ckpt["model"], strict=True)
        if source == args.out:
            validate_existing_optimizer_checkpoint(ckpt)
            opt.load_state_dict(ckpt["optimizer"])
            start_step = int(ckpt.get("completed_steps", 0))
            best_full_formal = float(ckpt.get("best_full_formal_loss", float("inf")))
            lr_decay_steps = int(ckpt.get("lr_decay_steps", int(args.steps)))
        print(f"{'resume' if source == args.out else 'init'} from {source}")
    if start_step >= int(args.steps):
        raise SystemExit("--steps 必须大于 checkpoint 的 completed_steps")

    regular_validation_batches = max(1, len(val_ds.files) * 10 // int(args.batch_size))
    print(
        f"px0: train_files={len(train_ds.files)} val_files={len(val_ds.files)} "
        f"batch={args.batch_size} steps={args.steps} {args.model_kind} b{args.blocks}c{args.width} "
        f"h{args.heads}ffn{args.ffn_channels} val_batches={regular_validation_batches} "
        f"loss_weights=(final={args.final_value_loss_weight}, root={args.root_wdl_loss_weight}, "
        f"moves={args.moves_left_loss_weight}, soft=T{args.soft_policy_temperature}/w{args.soft_policy_weight})"
    )
    train_iter = iter(train_loader)
    train_sums: dict[str, float] = {}
    train_batches = 0
    for step in range(start_step + 1, int(args.steps) + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter, batch = iter(train_loader), None
            batch = next(train_iter)
        model.train()
        boards = batch["board"].to(device=device, dtype=torch.float32)
        raw_policy = batch["policy"].to(device=device, dtype=torch.float32)
        winner_wdl = batch["winner_wdl"].to(device=device, dtype=torch.float32)
        root_wdl = batch["root_wdl"].to(device=device, dtype=torch.float32)
        plies_left = batch["plies_left"].to(device=device, dtype=torch.float32)
        set_optimizer_learning_rate(
            opt,
            learning_rate_at_step(
                step,
                total_steps=lr_decay_steps,
                lr=float(args.lr),
                warmup_steps=int(args.warmup_steps),
                min_lr_scale=float(args.min_lr_scale),
            ),
        )
        opt.zero_grad(set_to_none=True)
        terms = compute_loss_terms(
            forward_training(model, boards, amp_enabled=amp_enabled),
            raw_policy=raw_policy,
            winner_wdl=winner_wdl,
            root_wdl=root_wdl,
            plies_left=plies_left,
            final_value_loss_weight=float(args.final_value_loss_weight),
            moves_left_loss_weight=float(args.moves_left_loss_weight),
            soft_policy_weight=float(args.soft_policy_weight),
            soft_policy_temperature=float(args.soft_policy_temperature),
            root_wdl_loss_weight=float(args.root_wdl_loss_weight),
        )
        loss = terms["total"]
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        for name, value in terms.items():
            train_sums[name] = train_sums.get(name, 0.0) + float(value.item())
        train_batches += 1
        full_validation = step % int(args.full_validation_every) == 0 or step == int(args.steps)
        if step % max(1, int(args.eval_every)) == 0 or full_validation:
            val = run_val(
                model,
                full_val_loader if full_validation else val_loader,
                device=device,
                final_value_loss_weight=float(args.final_value_loss_weight),
                moves_left_loss_weight=float(args.moves_left_loss_weight),
                soft_policy_weight=float(args.soft_policy_weight),
                soft_policy_temperature=float(args.soft_policy_temperature),
                root_wdl_loss_weight=float(args.root_wdl_loss_weight),
                amp_enabled=amp_enabled,
                max_batches=None if full_validation else regular_validation_batches,
            )
            train = {name: total / train_batches for name, total in train_sums.items()}
            print(f"step {step:,}/{int(args.steps):,} | lr={opt.param_groups[0]['lr']:.2e}")
            print(
                f"  train[{train_batches:>4}] total={train['total']:.4f} formal={train['formal']:.4f} "
                f"policy={train['policy']:.4f} final_wdl={train['final_value_ce']:.4f} "
                f"root_wdl={train['root_value_ce']:.4f} moves={train['moves_left']:.4f} "
                f"soft_kl={train['soft_policy_kl']:.4f}"
            )
            print(
                f"  {'full_val' if full_validation else 'val':<7} total={val['total']:.4f} "
                f"formal={val['formal']:.4f} policy={val['policy']:.4f} "
                f"final_wdl={val['final_value_ce']:.4f} root_wdl={val['root_value_ce']:.4f} "
                f"moves={val['moves_left']:.4f} soft_kl={val['soft_policy_kl']:.4f}"
            )
            payload = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "optimizer_kind": OPTIMIZER_KIND,
                "width": int(args.width),
                "blocks": int(args.blocks),
                "bottleneck_channels": int(args.bottleneck_channels),
                "heads": int(args.heads),
                "ffn_channels": int(args.ffn_channels),
                "in_planes": int(args.in_planes),
                "n_moves": int(args.num_moves),
                "format": "px0_v7",
                "value_head": True,
                "moves_left_head": True,
                "auxiliary_heads": True,
                "trunk_kind": model.trunk_kind,
                "model_format": "px0_attentionbody_v1" if trunk_kind != CNN_TRUNK_KIND else "x7_v2",
                "value_head_format": "wdl",
                "value_target_kind": "final_wdl_plus_root_wdl",
                "soft_policy_temperature": float(args.soft_policy_temperature),
                "soft_policy_weight": float(args.soft_policy_weight),
                "final_value_loss_weight": float(args.final_value_loss_weight),
                "root_wdl_loss_weight": float(args.root_wdl_loss_weight),
                "amp": amp_enabled,
                "completed_steps": step,
                "best_full_formal_loss": min(best_full_formal, val["formal"]) if full_validation else best_full_formal,
                "lr_decay_steps": lr_decay_steps,
                "config_path": str(args.config_path),
                "run_name": str(args.name),
                "px0_version": str(args.px0_version),
                "px0_root": str(args.px0_root.resolve()),
                "validation_manifest": str(validation_manifest),
                "batch_size": int(args.batch_size),
                "validation_batches": regular_validation_batches,
                "full_validation_every": int(args.full_validation_every),
                "last_validation_is_full": full_validation,
                "warmup_steps": int(args.warmup_steps),
                "lr": float(args.lr),
                "min_lr_scale": float(args.min_lr_scale),
                "weight_decay": float(args.weight_decay),
                "last_val": val,
                "last_train": train,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            save_checkpoint(payload, args.out)
            if full_validation and math.isfinite(val["formal"]) and val["formal"] < best_full_formal:
                best_full_formal = val["formal"]
                save_checkpoint(payload, args.out.with_name(args.out.stem + ".best" + args.out.suffix))
            train_sums.clear()
            train_batches = 0


if __name__ == "__main__":
    main()
