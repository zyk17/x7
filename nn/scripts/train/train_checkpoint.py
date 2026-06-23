"""Checkpoint 读写与 resume。"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, LinearLR, SequentialLR

from train_common import MIN_LR, WARMUP_EPOCHS


def lr_scheduler(opt: AdamW, *, epochs: int):
    warmup = WARMUP_EPOCHS
    if warmup >= epochs:
        return CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=MIN_LR)
    warm = LinearLR(opt, start_factor=1e-2, end_factor=1.0, total_iters=warmup)
    cos = CosineAnnealingLR(opt, T_max=max(1, epochs - warmup), eta_min=MIN_LR)
    return SequentialLR(opt, [warm, cos], milestones=[warmup])


def scheduler_for_resume(opt: AdamW, *, total_epochs: int, completed_epochs: int):
    scheduler = lr_scheduler(opt, epochs=max(1, total_epochs))
    if completed_epochs > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(completed_epochs):
                scheduler.step()
    return scheduler


def set_optimizer_lr(opt: AdamW, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = float(lr)


def assert_ckpt_compatible(
    ckpt: dict,
    *,
    moves: list[str],
    n_moves: int,
    width: int,
    blocks: int,
    value_head: bool,
) -> None:
    if int(ckpt.get("n_moves", -1)) != n_moves:
        raise ValueError(
            f"checkpoint n_moves={ckpt.get('n_moves')} 与当前词表长度 {n_moves} 不一致"
        )
    if ckpt.get("moves") is not None and ckpt["moves"] != moves:
        raise ValueError("checkpoint 中的 moves 列表与 --vocab 不一致")
    if int(ckpt.get("width", -1)) != width:
        raise ValueError(
            f"checkpoint width={ckpt.get('width')} 与 --width={width} 不一致"
        )
    if int(ckpt.get("blocks", -1)) != blocks:
        raise ValueError(
            f"checkpoint blocks={ckpt.get('blocks')} 与 --blocks={blocks} 不一致"
        )
    if bool(ckpt.get("value_head", False)) != bool(value_head):
        raise ValueError("checkpoint value_head 配置与本次训练不一致")
    if value_head and ckpt.get("value_head_format") not in (None, "wdl"):
        raise ValueError("checkpoint value_head_format 与当前 WDL value 不一致")


def load_resume(
    out_path: Path,
    model: torch.nn.Module,
    opt: AdamW,
    *,
    moves: list[str],
    n_moves: int,
    width: int,
    blocks: int,
    value_head: bool,
    lr: float,
    lr_schedule_epochs: int,
    device: torch.device,
) -> tuple[int, float, int, LRScheduler, dict[str, Any]]:
    ckpt = torch.load(out_path, map_location=device)
    assert_ckpt_compatible(
        ckpt,
        moves=moves,
        n_moves=n_moves,
        width=width,
        blocks=blocks,
        value_head=value_head,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    start_epoch = int(ckpt.get("completed_epochs", 0))
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    best_epoch = int(ckpt.get("best_epoch", 0))
    if "optimizer" in ckpt:
        try:
            opt.load_state_dict(ckpt["optimizer"])
        except ValueError:
            print("提示: 忽略旧 optimizer 状态，按当前实验新建")
    set_optimizer_lr(opt, float(lr))
    scheduler = scheduler_for_resume(
        opt,
        total_epochs=lr_schedule_epochs,
        completed_epochs=start_epoch,
    )
    print(f"resume from {out_path} | completed_epochs={start_epoch}")
    return start_epoch, best_val_loss, best_epoch, scheduler, ckpt


def load_init_weights(
    init_path: Path,
    model: torch.nn.Module,
    *,
    moves: list[str],
    n_moves: int,
    width: int,
    blocks: int,
    device: torch.device,
) -> dict[str, Any]:
    ckpt = torch.load(init_path, map_location=device)
    if int(ckpt.get("n_moves", -1)) != n_moves:
        raise ValueError(
            f"init checkpoint n_moves={ckpt.get('n_moves')} 与当前词表长度 {n_moves} 不一致"
        )
    if ckpt.get("moves") is not None and ckpt["moves"] != moves:
        raise ValueError("init checkpoint 中的 moves 列表与 --vocab 不一致")
    if int(ckpt.get("width", -1)) != width:
        raise ValueError(
            f"init checkpoint width={ckpt.get('width')} 与 --width={width} 不一致"
        )
    if int(ckpt.get("blocks", -1)) != blocks:
        raise ValueError(
            f"init checkpoint blocks={ckpt.get('blocks')} 与 --blocks={blocks} 不一致"
        )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        raise ValueError(f"init checkpoint 含未知参数: {unexpected}")
    print(
        f"init from {init_path} | missing={len(missing)} "
        f"({' '.join(missing[:4]) + (' ...' if len(missing) > 4 else '')})"
    )
    return ckpt


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    opt: AdamW,
    scheduler,
    args,
    moves: list[str],
    n_moves: int,
    value_head: bool,
    epoch: int,
    lr_schedule_epochs: int,
    best_val_loss: float,
    best_epoch: int,
    val_mean: float,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "in_planes": int(getattr(model, "in_planes", 15)),
        "width": args.width,
        "blocks": args.blocks,
        "n_moves": n_moves,
        "moves": moves,
        "value_head": value_head,
        "value_head_format": "wdl" if value_head else "off",
        "value_head_hidden_dim": int(args.value_head_hidden_dim),
        "value_target_kind": "qmix_wdl" if value_head else "off",
        "value_loss_weight": float(args.value_loss_weight),
        "q_ratio": float(getattr(args, "q_ratio", 0.0)),
        "value_min_visits": int(args.value_min_visits),
        "train_mix": str(args.train_mix) if args.train_mix else None,
        "train_dir": str(args.train_dir) if args.train_dir else None,
        "val_dir": str(args.val_dir),
        "search_policy_weight": float(args.search_policy_weight),
        "freeze_trunk": bool(args.freeze_trunk),
        "freeze_policy_head": bool(args.freeze_policy_head),
        "freeze_value_head": bool(args.freeze_value_head),
        "completed_epochs": epoch + 1,
        "lr_schedule_epochs": lr_schedule_epochs,
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "last_val_loss": val_mean,
    }


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
