#!/usr/bin/env python3
"""训练 policy/value 主线（薄入口）。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.optim import AdamW

NN_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(NN_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nn.model import PolicyResNet

from train_checkpoint import (
    checkpoint_payload,
    load_init_weights,
    load_resume,
    lr_scheduler,
    save_checkpoint,
)
from train_common import (
    TRAIN_SEED,
    default_num_workers,
    default_val_num_workers,
    set_requires_grad,
)
from train_data import build_datasets, build_loaders, validate_train_args
from train_loop import format_val_log, run_train_epoch, run_val_epoch


def _resolve_repo_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train policy/value on XRSH v5")

    data = ap.add_argument_group("data")
    data.add_argument("--train-dir", type=Path, default=None)
    data.add_argument("--train-mix", type=Path, default=None, help="混合训练 manifest JSON")
    data.add_argument("--val-dir", type=Path, required=True)
    data.add_argument("--vocab", type=Path, required=True)
    data.add_argument("--train-dataset-mode", choices=("eager", "lazy"), default="eager")
    data.add_argument("--val-dataset-mode", choices=("eager", "lazy"), default="lazy")

    model = ap.add_argument_group("model")
    model.add_argument("--width", type=int, default=128)
    model.add_argument("--blocks", type=int, default=8)
    model.add_argument("--value-head", action="store_true")
    model.add_argument("--value-head-hidden-dim", type=int, default=0)
    model.add_argument("--freeze-trunk", action="store_true")
    model.add_argument("--freeze-policy-head", action="store_true")
    model.add_argument("--freeze-value-head", action="store_true")

    loss = ap.add_argument_group("loss")
    loss.add_argument("--value-loss-weight", type=float, default=0.5)
    loss.add_argument(
        "--q-ratio",
        type=float,
        default=1.0,
        help="value target = q_ratio*search_wdl + (1-q_ratio)*winner_wdl；XRSH 兼容路径只使用退化 search_wdl",
    )
    loss.add_argument(
        "--value-min-visits",
        type=int,
        default=1,
        help="仅 search_visits>=该值的样本参与 value loss",
    )
    loss.add_argument(
        "--search-policy-weight",
        type=float,
        default=0.0,
        help=">0 时额外拟合 MCTS 根 visit 分布",
    )

    runtime = ap.add_argument_group("runtime")
    runtime.add_argument("--batch-size", type=int, default=512)
    runtime.add_argument("--epochs", type=int, default=10)
    runtime.add_argument("--val-every", type=int, default=1)
    runtime.add_argument("--lr", type=float, default=1e-3)
    runtime.add_argument("--weight-decay", type=float, default=1e-4)
    runtime.add_argument("--device", default="cuda")
    runtime.add_argument("--amp", action="store_true")
    runtime.add_argument("--train-num-workers", type=int, default=default_num_workers())
    runtime.add_argument("--val-num-workers", type=int, default=default_val_num_workers())

    output = ap.add_argument_group("output")
    output.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "checkpoints" / "policy.pt",
    )
    output.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="仅在 --out 不存在时使用该 checkpoint 做非严格热启动",
    )
    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.train_dir = _resolve_repo_path(args.train_dir)
    args.train_mix = _resolve_repo_path(args.train_mix)
    args.val_dir = _resolve_repo_path(args.val_dir)
    args.vocab = _resolve_repo_path(args.vocab)
    args.out = _resolve_repo_path(args.out)
    args.init_from = _resolve_repo_path(args.init_from)
    validate_train_args(parser, args)

    random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)

    vocab_data = json.loads(args.vocab.read_text(encoding="utf-8"))
    moves: list[str] = vocab_data["moves"]
    move_to_idx = {m: i for i, m in enumerate(moves)}
    n_moves = len(moves)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"torch {torch.__version__} | cuda.is_available={torch.cuda.is_available()} | device={device}"
    )

    value_head = bool(args.value_head)
    search_policy_head = float(args.search_policy_weight) > 0.0

    train_ds, val_ds, train_note = build_datasets(
        args,
        move_to_idx,
        value_head=value_head,
        search_policy_head=search_policy_head,
    )
    print(
        f"dataset: train={len(train_ds)} val={len(val_ds)} | "
        f"train_src={train_note} | "
        f"value_target={'qmix_wdl' if value_head else 'off'} | "
        f"q_ratio={float(args.q_ratio):.3f} | "
        f"search_policy_weight={float(args.search_policy_weight):.3f}"
    )

    train_loader, val_loader, train_bs = build_loaders(
        args,
        train_ds,
        val_ds,
        device_type=device.type,
    )

    model = PolicyResNet(
        in_planes=15,
        width=args.width,
        num_blocks=args.blocks,
        num_moves=n_moves,
        value_head=value_head,
        value_head_hidden_dim=int(args.value_head_hidden_dim),
    ).to(device)
    if args.freeze_trunk:
        set_requires_grad(model.stem, False)
        set_requires_grad(model.blocks, False)
    if args.freeze_policy_head:
        set_requires_grad(model.policy_head, False)
        set_requires_grad(model.fc, False)
    if args.freeze_value_head and hasattr(model, "fc_value"):
        if hasattr(model, "value_head_module"):
            set_requires_grad(model.value_head_module, False)
        set_requires_grad(model.fc_value, False)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise SystemExit("当前设置下没有可训练参数")

    opt = AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    resume = args.out.is_file()
    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    lr_schedule_epochs = start_epoch + args.epochs
    scheduler = lr_scheduler(opt, epochs=max(1, lr_schedule_epochs))

    if resume:
        start_epoch, best_val_loss, best_epoch, scheduler, _ = load_resume(
            args.out,
            model,
            opt,
            moves=moves,
            n_moves=n_moves,
            width=args.width,
            blocks=args.blocks,
            value_head=value_head,
            lr=args.lr,
            lr_schedule_epochs=start_epoch + args.epochs,
            device=device,
        )
        lr_schedule_epochs = start_epoch + args.epochs
    elif args.init_from is not None:
        load_init_weights(
            args.init_from,
            model,
            moves=moves,
            n_moves=n_moves,
            width=args.width,
            blocks=args.blocks,
            device=device,
        )

    best_out = args.out.with_name(args.out.stem + ".best" + args.out.suffix)
    amp_enabled = bool(args.amp) and device.type == "cuda"
    print(
        f"runtime: train_workers={int(args.train_num_workers)} "
        f"val_workers={int(args.val_num_workers)} amp={amp_enabled}"
    )
    if device.type == "cuda" and int(args.train_num_workers) == 0:
        print("warning: train_num_workers=0 on CUDA, GPU 可能会等待数据")
    end_epoch = start_epoch + args.epochs

    for epoch in range(start_epoch, end_epoch):
        train_bs.set_epoch(epoch)
        train_loss = run_train_epoch(
            model,
            train_loader,
            opt,
            device=device,
            value_head=value_head,
            search_policy_head=search_policy_head,
            amp_enabled=amp_enabled,
            args=args,
            epoch=epoch,
            end_epoch=end_epoch,
        )
        print(f"train loss {train_loss:.4f} lr={opt.param_groups[0]['lr']:.2e}")

        should_validate = ((epoch + 1) % max(1, int(args.val_every)) == 0) or (epoch + 1 == end_epoch)
        val_mean = float("nan")
        improved = False
        if should_validate:
            val_result = run_val_epoch(
                model,
                val_loader,
                device=device,
                value_head=value_head,
                search_policy_head=search_policy_head,
                amp_enabled=amp_enabled,
                args=args,
                val_ds=val_ds,
            )
            val_mean = float(val_result["val_mean"])
            print(format_val_log(val_result, value_head=value_head, search_policy_head=search_policy_head))
            improved = val_mean < best_val_loss
            if improved:
                best_val_loss = val_mean
                best_epoch = epoch + 1
        else:
            print(f"skip val: epoch {epoch + 1}/{end_epoch}")

        scheduler.step()

        payload = checkpoint_payload(
            model=model,
            opt=opt,
            scheduler=scheduler,
            args=args,
            moves=moves,
            n_moves=n_moves,
            value_head=value_head,
            epoch=epoch,
            lr_schedule_epochs=lr_schedule_epochs,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            val_mean=val_mean,
        )
        save_checkpoint(payload, args.out)
        print(f"checkpoint -> {args.out}")
        if improved:
            save_checkpoint(payload, best_out)
            print(f"best checkpoint -> {best_out}")


if __name__ == "__main__":
    main()
