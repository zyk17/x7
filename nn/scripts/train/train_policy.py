#!/usr/bin/env python3
"""训练 policy/value 主线。"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nn.dataset_xrsh import PolicyXrshDataset
from nn.metrics import ValMetricsState, format_val_metrics_report
from nn.model import (
    PolicyResNet,
    policy_cross_entropy,
    soft_policy_cross_entropy,
    value_head_tanh_mse,
)
from nn.xrsh_io import xrsh_dir_is_complete

_TRAIN_SEED = 42
_LABEL_SMOOTHING = 0.08
_WARMUP_EPOCHS = 1
_MIN_LR = 1e-5


def _default_num_workers() -> int:
    if os.name == "nt":
        return 0
    return min(8, max(0, (os.cpu_count() or 8) - 2))


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(enabled)


def unpack_train_batch(
    batch: tuple[Any, ...],
    *,
    value_head: bool,
    search_policy_head: bool,
) -> dict[str, Any]:
    idx = 4
    out: dict[str, Any] = {
        "boards": batch[0],
        "masks": batch[1],
        "targets": batch[2],
        "weights": batch[3],
    }
    if value_head:
        out["t_val"] = batch[idx]
        idx += 1
    if search_policy_head:
        out["visit_target"] = batch[idx]
    return out


def unpack_val_batch(
    batch: tuple[Any, ...],
    *,
    value_head: bool,
    search_policy_head: bool,
) -> dict[str, Any]:
    idx = 4
    out: dict[str, Any] = {
        "boards": batch[0],
        "masks": batch[1],
        "targets": batch[2],
        "weights": batch[3],
    }
    if value_head:
        out["t_val"] = batch[idx]
        idx += 1
    if search_policy_head:
        out["visit_target"] = batch[idx]
        idx += 3
    out["plies"] = batch[idx]
    out["src_ids"] = batch[idx + 1]
    return out


class GameGroupedBatchSampler:
    """先随机打乱局顺序，再按局内行序串联后切块。"""

    def __init__(
        self,
        batch_size: int,
        *,
        row_group_ids: list[int],
        drop_last: bool = False,
        seed: int = _TRAIN_SEED,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        gid_to_idx: dict[int, list[int]] = defaultdict(list)
        for i, gid in enumerate(row_group_ids):
            gid_to_idx[int(gid)].append(i)
        self._groups = list(gid_to_idx.items())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups = [(gid, list(idxs)) for gid, idxs in self._groups]
        rng.shuffle(groups)
        stream: list[int] = []
        for _, idxs in groups:
            stream.extend(idxs)
        batch_size = self.batch_size
        for i in range(0, len(stream), batch_size):
            chunk = stream[i : i + batch_size]
            if len(chunk) < batch_size and self.drop_last:
                continue
            yield chunk

    def __len__(self) -> int:
        n = sum(len(idxs) for _, idxs in self._groups)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def _lr_scheduler(opt: AdamW, *, epochs: int):
    warmup = _WARMUP_EPOCHS
    if warmup >= epochs:
        return CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=_MIN_LR)
    warm = LinearLR(opt, start_factor=1e-2, end_factor=1.0, total_iters=warmup)
    cos = CosineAnnealingLR(opt, T_max=max(1, epochs - warmup), eta_min=_MIN_LR)
    return SequentialLR(opt, [warm, cos], milestones=[warmup])


def _scheduler_for_resume(
    opt: AdamW,
    *,
    total_epochs: int,
    completed_epochs: int,
):
    scheduler = _lr_scheduler(opt, epochs=max(1, total_epochs))
    if completed_epochs > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(completed_epochs):
                scheduler.step()
    return scheduler


def _set_optimizer_lr(opt: AdamW, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = float(lr)


def _assert_ckpt_compatible(
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train policy/value on XRSH")
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "checkpoints" / "policy.pt",
    )
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--value-head", action="store_true")
    ap.add_argument("--value-head-hidden-dim", type=int, default=0)
    ap.add_argument("--value-loss-weight", type=float, default=0.5)
    ap.add_argument("--value-target-weight-alpha", type=float, default=1.0)
    ap.add_argument(
        "--search-policy-weight",
        type=float,
        default=0.0,
        help=">0 时额外拟合 MCTS 根 visit 分布",
    )
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-dataset-mode", choices=("eager", "lazy"), default="eager")
    ap.add_argument("--val-dataset-mode", choices=("eager", "lazy"), default="lazy")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--train-num-workers", type=int, default=_default_num_workers())
    ap.add_argument("--val-num-workers", type=int, default=0)
    ap.add_argument("--freeze-trunk", action="store_true")
    ap.add_argument("--freeze-policy-head", action="store_true")
    ap.add_argument("--freeze-value-head", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    random.seed(_TRAIN_SEED)
    torch.manual_seed(_TRAIN_SEED)

    vocab_data = json.loads(args.vocab.read_text(encoding="utf-8"))
    moves: list[str] = vocab_data["moves"]
    move_to_idx = {m: i for i, m in enumerate(moves)}
    n_moves = len(moves)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"torch {torch.__version__} | cuda.is_available={torch.cuda.is_available()} | device={device}"
    )

    if not xrsh_dir_is_complete(args.train_dir):
        raise FileNotFoundError(f"--train-dir 不完整: {args.train_dir}")
    if not xrsh_dir_is_complete(args.val_dir):
        raise FileNotFoundError(f"--val-dir 不完整: {args.val_dir}")

    value_head = bool(args.value_head)
    search_policy_head = float(args.search_policy_weight) > 0.0

    train_ds = PolicyXrshDataset(
        args.train_dir,
        move_to_idx,
        for_training=True,
        with_value_labels=value_head,
        with_search_labels=search_policy_head,
        storage_mode=args.train_dataset_mode,
    )
    val_ds = PolicyXrshDataset(
        args.val_dir,
        move_to_idx,
        with_row_meta=True,
        with_value_labels=value_head,
        with_search_labels=search_policy_head,
        storage_mode=args.val_dataset_mode,
    )

    print(
        f"dataset: train={len(train_ds)} val={len(val_ds)} | "
        f"value_target={'search_q' if value_head else 'off'} | "
        f"search_policy_weight={float(args.search_policy_weight):.3f}"
    )

    def _loader_kw(nw: int) -> dict[str, Any]:
        out: dict[str, Any] = dict(
            num_workers=nw,
            pin_memory=device.type == "cuda",
            persistent_workers=nw > 0,
            prefetch_factor=(2 if nw > 0 else None),
        )
        if nw == 0:
            out.pop("prefetch_factor", None)
            out.pop("persistent_workers", None)
        return out

    train_bs = GameGroupedBatchSampler(
        batch_size=args.batch_size,
        row_group_ids=train_ds.row_group_ids,
        seed=_TRAIN_SEED,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_bs,
        **_loader_kw(int(args.train_num_workers)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **_loader_kw(int(args.val_num_workers)),
    )

    model = PolicyResNet(
        width=args.width,
        num_blocks=args.blocks,
        num_moves=n_moves,
        value_head=value_head,
        value_head_hidden_dim=int(args.value_head_hidden_dim),
    ).to(device)
    if args.freeze_trunk:
        _set_requires_grad(model.stem, False)
        _set_requires_grad(model.blocks, False)
    if args.freeze_policy_head:
        _set_requires_grad(model.fc, False)
    if args.freeze_value_head and hasattr(model, "fc_value"):
        _set_requires_grad(model.fc_value, False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    resume = args.out.is_file()
    start_epoch = 0
    ckpt: dict | None = None
    if resume:
        ckpt = torch.load(args.out, map_location=device)
        _assert_ckpt_compatible(
            ckpt,
            moves=moves,
            n_moves=n_moves,
            width=args.width,
            blocks=args.blocks,
            value_head=value_head,
        )
        model.load_state_dict(ckpt["model"], strict=True)
        start_epoch = int(ckpt.get("completed_epochs", 0))
        print(f"resume from {args.out} | completed_epochs={start_epoch}")

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise SystemExit("当前设置下没有可训练参数")

    opt = AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    lr_schedule_epochs = start_epoch + args.epochs
    scheduler = _lr_scheduler(opt, epochs=max(1, lr_schedule_epochs))
    if resume and ckpt is not None:
        if "optimizer" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer"])
            except ValueError:
                print("提示: 忽略旧 optimizer 状态，按当前实验新建")
        _set_optimizer_lr(opt, float(args.lr))
        scheduler = _scheduler_for_resume(
            opt,
            total_epochs=lr_schedule_epochs,
            completed_epochs=start_epoch,
        )

    best_val_loss = float("inf") if ckpt is None else float(ckpt.get("best_val_loss", float("inf")))
    best_epoch = 0 if ckpt is None else int(ckpt.get("best_epoch", 0))
    best_out = args.out.with_name(args.out.stem + ".best" + args.out.suffix)

    amp_enabled = bool(args.amp) and device.type == "cuda"
    end_epoch = start_epoch + args.epochs

    for epoch in range(start_epoch, end_epoch):
        train_bs.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        weight_sum = 0.0

        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{end_epoch} train"):
            bt = unpack_train_batch(
                batch,
                value_head=value_head,
                search_policy_head=search_policy_head,
            )
            boards = bt["boards"].to(device, non_blocking=device.type == "cuda")
            masks = bt["masks"].to(device, non_blocking=device.type == "cuda")
            targets = bt["targets"].to(device, non_blocking=device.type == "cuda")
            weights = bt["weights"].to(device, non_blocking=device.type == "cuda")

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                out = model(boards)
                if value_head:
                    logits, pred_value = out
                else:
                    logits = out
                    pred_value = None

                loss = policy_cross_entropy(
                    logits,
                    targets,
                    masks,
                    label_smoothing=_LABEL_SMOOTHING,
                    sample_weight=weights,
                )

                if value_head and pred_value is not None:
                    target_value = bt["t_val"].to(device, non_blocking=device.type == "cuda")
                    loss = loss + float(args.value_loss_weight) * value_head_tanh_mse(
                        pred_value,
                        target_value,
                        target_weight_alpha=float(args.value_target_weight_alpha),
                        sample_weight=weights,
                    )

                if search_policy_head:
                    visit_target = bt["visit_target"].to(device, non_blocking=device.type == "cuda")
                    loss = loss + float(args.search_policy_weight) * soft_policy_cross_entropy(
                        logits,
                        visit_target,
                        masks,
                        sample_weight=weights,
                    )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * weights.sum().item()
            weight_sum += weights.sum().item()

        print(f"train loss {total_loss / max(weight_sum, 1e-8):.4f} lr={opt.param_groups[0]['lr']:.2e}")

        should_validate = ((epoch + 1) % max(1, int(args.val_every)) == 0) or (epoch + 1 == end_epoch)
        val_mean = float("nan")
        improved = False
        if should_validate:
            model.eval()
            vloss = 0.0
            vcount = 0
            correct = 0
            value_mse = 0.0
            search_ce = 0.0
            val_metrics = ValMetricsState()
            with torch.no_grad():
                for batch in val_loader:
                    bv = unpack_val_batch(
                        batch,
                        value_head=value_head,
                        search_policy_head=search_policy_head,
                    )
                    boards = bv["boards"].to(device, non_blocking=device.type == "cuda")
                    masks = bv["masks"].to(device, non_blocking=device.type == "cuda")
                    targets = bv["targets"].to(device, non_blocking=device.type == "cuda")
                    weights = bv["weights"].to(device, non_blocking=device.type == "cuda")
                    plies = bv["plies"].to(device, non_blocking=device.type == "cuda")
                    src_ids = bv["src_ids"].to(device, non_blocking=device.type == "cuda")

                    autocast_ctx = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if amp_enabled
                        else nullcontext()
                    )
                    with autocast_ctx:
                        out = model(boards)
                        if value_head:
                            logits, pred_value = out
                        else:
                            logits = out
                            pred_value = None
                        loss = policy_cross_entropy(
                            logits,
                            targets,
                            masks,
                            label_smoothing=0.0,
                            sample_weight=weights,
                        )
                        if value_head and pred_value is not None:
                            target_value = bv["t_val"].to(device, non_blocking=device.type == "cuda")
                            value_mse += (
                                value_head_tanh_mse(
                                    pred_value,
                                    target_value,
                                    reduction="mean",
                                ).item()
                                * boards.size(0)
                            )
                        if search_policy_head:
                            visit_target = bv["visit_target"].to(device, non_blocking=device.type == "cuda")
                            search_ce += (
                                soft_policy_cross_entropy(
                                    logits,
                                    visit_target,
                                    masks,
                                    reduction="mean",
                                ).item()
                                * boards.size(0)
                            )
                    vloss += loss.item() * boards.size(0)
                    vcount += boards.size(0)
                    pred = logits.masked_fill(~masks, float("-inf")).argmax(dim=1)
                    correct += (pred == targets).sum().item()
                    val_metrics.update_batch(logits, targets, masks, plies, src_ids)

            val_mean = vloss / max(1, vcount)
            tail = ""
            if value_head and vcount > 0:
                tail += f" | val_value_mse {value_mse / vcount:.4f}"
            if search_policy_head and vcount > 0:
                tail += f" | val_search_ce {search_ce / vcount:.4f}"
            print(f"val loss {val_mean:.4f} acc {correct / max(1, vcount):.4f}{tail}")
            print(format_val_metrics_report(val_metrics, pgn_source_vocab=val_ds.pgn_source_vocab))
            improved = val_mean < best_val_loss
            if improved:
                best_val_loss = val_mean
                best_epoch = epoch + 1
        else:
            print(f"skip val: epoch {epoch + 1}/{end_epoch}")

        scheduler.step()

        payload = {
            "model": model.state_dict(),
            "width": args.width,
            "blocks": args.blocks,
            "n_moves": n_moves,
            "moves": moves,
            "value_head": value_head,
            "value_head_hidden_dim": int(args.value_head_hidden_dim),
            "value_target_kind": "search_q",
            "value_loss_weight": float(args.value_loss_weight),
            "value_target_weight_alpha": float(args.value_target_weight_alpha),
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
        torch.save(payload, args.out)
        print(f"checkpoint -> {args.out}")
        if improved:
            torch.save(payload, best_out)
            print(f"best checkpoint -> {best_out}")


if __name__ == "__main__":
    main()
