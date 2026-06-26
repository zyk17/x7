#!/usr/bin/env python3
"""PX0 主线训练入口。"""

from __future__ import annotations

import argparse
import math
import random
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

NN_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(NN_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nn.dataset_px0 import Px0ChunkDataset, Px0DatasetConfig
from nn.px0_kaggle import DEFAULT_PX0_ROOT, ensure_px0_version
from nn.model import (
    PolicyResNet,
    mix_wdl_targets,
    soft_policy_cross_entropy,
    value_q_mse,
    value_wdl_cross_entropy,
)

from train_checkpoint import lr_scheduler, save_checkpoint
from train_common import TRAIN_SEED, default_num_workers


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train small policy/value model on PX0 v6 chunks")
    ap.add_argument("--px0-version", default=None, help="Kaggle px0data version; if set, auto prepare local chunks")
    ap.add_argument("--px0-root", type=Path, default=DEFAULT_PX0_ROOT)
    ap.add_argument("--px0-val-ratio", type=float, default=0.1)
    ap.add_argument("--px0-seed", type=int, default=42)
    ap.add_argument("--px0-force-download", action="store_true")
    ap.add_argument("--train-glob", action="append", default=None, help="glob for training chunk files")
    ap.add_argument("--val-glob", action="append", default=None, help="glob for validation chunk files")
    ap.add_argument("--train-list", type=Path, default=None, help="JSON file list for training chunks")
    ap.add_argument("--val-list", type=Path, default=None, help="JSON file list for validation chunks")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--in-planes", type=int, default=124)
    ap.add_argument("--num-moves", type=int, default=2062)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--val-batches", type=int, default=32)
    ap.add_argument("--train-max-files", type=int, default=0)
    ap.add_argument("--val-max-files", type=int, default=0)
    ap.add_argument("--train-limit-samples", type=int, default=0)
    ap.add_argument("--val-limit-samples", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-loss-weight", type=float, default=1.0)
    ap.add_argument("--q-ratio", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=default_num_workers())
    ap.add_argument("--resume", action="store_true", help="resume from --out if it exists")
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.px0_version:
        if any([args.train_list, args.val_list, args.train_glob, args.val_glob]):
            raise SystemExit("使用 --px0-version 时，不要再传 --train-* / --val-*")
        if not (0.0 < float(args.px0_val_ratio) < 1.0):
            raise SystemExit("--px0-val-ratio 须在 (0,1) 内")
    else:
        train_sources = int(bool(args.train_list)) + int(bool(args.train_glob))
        val_sources = int(bool(args.val_list)) + int(bool(args.val_glob))
        if train_sources != 1:
            raise SystemExit("训练数据需要二选一：--train-glob 或 --train-list")
        if val_sources != 1:
            raise SystemExit("验证数据需要二选一：--val-glob 或 --val-list")
    if not (0.0 <= float(args.q_ratio) <= 1.0):
        raise SystemExit("--q-ratio 须在 [0,1] 内")


def take_logits_and_value(
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, tuple):
        raise TypeError("train_px0 requires value_head=True")
    return output


def validate_resume_checkpoint(
    ckpt: dict,
    *,
    args: argparse.Namespace,
    train_files: list[Path],
    val_files: list[Path],
) -> None:
    expected_q_ratio = float(args.q_ratio)
    got_q_ratio = ckpt.get("q_ratio")
    if got_q_ratio is not None and abs(float(got_q_ratio) - expected_q_ratio) > 1e-9:
        raise SystemExit(
            f"--resume 的 checkpoint q_ratio={float(got_q_ratio):.6f}，"
            f"但当前命令是 q_ratio={expected_q_ratio:.6f}；请换新输出文件或取消 --resume"
        )

    ckpt_train_files = ckpt.get("train_files")
    ckpt_val_files = ckpt.get("val_files")
    current_train_files = [str(p.resolve()) for p in train_files]
    current_val_files = [str(p.resolve()) for p in val_files]
    if ckpt_train_files is not None and list(ckpt_train_files) != current_train_files:
        raise SystemExit("--resume 的 checkpoint train_files 与当前数据集不一致；请换新输出文件或取消 --resume")
    if ckpt_val_files is not None and list(ckpt_val_files) != current_val_files:
        raise SystemExit("--resume 的 checkpoint val_files 与当前数据集不一致；请换新输出文件或取消 --resume")


def run_val(
    model: PolicyResNet,
    loader: DataLoader,
    *,
    device: torch.device,
    val_batches: int,
    value_loss_weight: float,
    q_ratio: float,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_policy = 0.0
    total_value_ce = 0.0
    total_value_q_mse = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            boards = batch["board"].to(device=device, dtype=torch.float32)
            raw_policy = batch["policy"].to(device=device, dtype=torch.float32)
            winner_wdl = batch["winner_wdl"].to(device=device, dtype=torch.float32)
            search_wdl = batch["search_wdl"].to(device=device, dtype=torch.float32)
            target_value = mix_wdl_targets(winner_wdl, search_wdl, q_ratio=q_ratio)
            legal_mask = raw_policy >= 0
            target_policy = raw_policy.clamp_min(0.0)
            output = model(boards)
            policy_logits, pred_value = take_logits_and_value(output)
            policy_loss = soft_policy_cross_entropy(policy_logits, target_policy, legal_mask)
            value_ce = value_wdl_cross_entropy(pred_value, target_value)
            value_q = value_q_mse(pred_value, target_value)
            loss = policy_loss + float(value_loss_weight) * value_ce
            total_loss += float(loss.item())
            total_policy += float(policy_loss.item())
            total_value_ce += float(value_ce.item())
            total_value_q_mse += float(value_q.item())
            batches += 1
            if batches >= val_batches:
                break
    if batches < 1:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        total_loss / batches,
        total_policy / batches,
        total_value_ce / batches,
        total_value_q_mse / batches,
    )


def build_dataset_configs(args: argparse.Namespace) -> tuple[Px0DatasetConfig, Px0DatasetConfig]:
    if args.px0_version:
        prepared = ensure_px0_version(
            args.px0_version,
            root=args.px0_root,
            val_ratio=float(args.px0_val_ratio),
            seed=int(args.px0_seed),
            force_download=bool(args.px0_force_download),
        )
        train_list = prepared.train_manifest
        val_list = prepared.val_manifest
    else:
        train_list = args.train_list.resolve() if args.train_list else None
        val_list = args.val_list.resolve() if args.val_list else None
    train_cfg = Px0DatasetConfig(
        patterns=tuple(args.train_glob or ()),
        file_list_path=train_list,
        shuffle_files=True,
        max_files=int(args.train_max_files),
        limit_samples=int(args.train_limit_samples),
    )
    val_cfg = Px0DatasetConfig(
        patterns=tuple(args.val_glob or ()),
        file_list_path=val_list,
        shuffle_files=False,
        max_files=int(args.val_max_files),
        limit_samples=int(args.val_limit_samples),
    )
    return train_cfg, val_cfg


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | cuda.is_available={torch.cuda.is_available()} | device={device}")

    train_cfg, val_cfg = build_dataset_configs(args)
    train_ds = Px0ChunkDataset(train_cfg)
    val_ds = Px0ChunkDataset(val_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        num_workers=max(0, min(2, int(args.num_workers))),
        pin_memory=device.type == "cuda",
    )

    model = PolicyResNet(
        in_planes=int(args.in_planes),
        width=int(args.width),
        num_blocks=int(args.blocks),
        num_moves=int(args.num_moves),
        value_head=True,
        value_head_hidden_dim=64,
    ).to(device)
    opt = AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = lr_scheduler(opt, epochs=max(1, int(args.steps)))
    start_step = 0
    best_val = float("inf")
    if args.resume and args.out.is_file():
        ckpt = torch.load(args.out, map_location=device)
        validate_resume_checkpoint(
            ckpt,
            args=args,
            train_files=train_ds.files,
            val_files=val_ds.files,
        )
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("completed_steps", 0))
        best_val = float(ckpt.get("best_val_loss", float("inf")))
        if start_step >= int(args.steps):
            raise SystemExit(
                f"--out 已完成到 step={start_step}，当前 --steps={int(args.steps)}；请增大 --steps 或换新文件名"
            )
        scheduler = lr_scheduler(opt, epochs=max(1, int(args.steps)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(start_step):
                scheduler.step()
        print(f"resume from {args.out} | completed_steps={start_step}")

    print(
        f"px0: train_files={len(train_ds.files)} val_files={len(val_ds.files)} "
        f"batch_size={int(args.batch_size)} steps={int(args.steps)} q_ratio={float(args.q_ratio):.3f}"
    )
    if args.px0_version:
        print(
            f"px0_kaggle: version={args.px0_version} root={args.px0_root.resolve()} "
            f"val_ratio={float(args.px0_val_ratio):.3f}"
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
        target_value = mix_wdl_targets(winner_wdl, search_wdl, q_ratio=float(args.q_ratio))
        legal_mask = raw_policy >= 0
        target_policy = raw_policy.clamp_min(0.0)

        opt.zero_grad(set_to_none=True)
        output = model(boards)
        policy_logits, pred_value = take_logits_and_value(output)
        policy_loss = soft_policy_cross_entropy(policy_logits, target_policy, legal_mask)
        value_ce = value_wdl_cross_entropy(pred_value, target_value)
        value_q = value_q_mse(pred_value, target_value)
        loss = policy_loss + float(args.value_loss_weight) * value_ce
        loss.backward()
        opt.step()
        scheduler.step()

        if step == 1 or step % max(1, int(args.eval_every)) == 0 or step == int(args.steps):
            val_loss, val_policy, val_value_ce, val_value_q_mse = run_val(
                model,
                val_loader,
                device=device,
                val_batches=int(args.val_batches),
                value_loss_weight=float(args.value_loss_weight),
                q_ratio=float(args.q_ratio),
            )
            print(
                f"step {step}/{int(args.steps)} "
                f"train_loss={loss.item():.4f} "
                f"train_policy={policy_loss.item():.4f} "
                f"train_value_ce={value_ce.item():.4f} "
                f"train_value_q_mse={value_q.item():.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_policy={val_policy:.4f} "
                f"val_value_ce={val_value_ce:.4f} "
                f"val_value_q_mse={val_value_q_mse:.4f} "
                f"lr={opt.param_groups[0]['lr']:.2e}"
            )
            payload = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "width": int(args.width),
                "blocks": int(args.blocks),
                "in_planes": int(args.in_planes),
                "n_moves": int(args.num_moves),
                "format": "px0_v6",
                "value_head": True,
                "value_head_format": "wdl",
                "value_head_hidden_dim": 64,
                "value_target_kind": "qmix_wdl",
                "q_ratio": float(args.q_ratio),
                "completed_steps": step,
                "best_val_loss": min(best_val, val_loss),
                "train_glob": list(args.train_glob or ()),
                "val_glob": list(args.val_glob or ()),
                "train_list": str(args.train_list.resolve()) if args.train_list else None,
                "val_list": str(args.val_list.resolve()) if args.val_list else None,
                "px0_version": str(args.px0_version) if args.px0_version else None,
                "px0_root": str(args.px0_root.resolve()) if args.px0_version else None,
                "px0_val_ratio": float(args.px0_val_ratio) if args.px0_version else None,
                "px0_seed": int(args.px0_seed) if args.px0_version else None,
                "train_files": [str(p) for p in train_ds.files],
                "val_files": [str(p) for p in val_ds.files],
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "value_loss_weight": float(args.value_loss_weight),
                "last_val_value_ce": float(val_value_ce),
                "last_val_value_q_mse": float(val_value_q_mse),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            save_checkpoint(payload, args.out)
            if math.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                best_out = args.out.with_name(args.out.stem + ".best" + args.out.suffix)
                save_checkpoint(payload, best_out)


if __name__ == "__main__":
    main()
