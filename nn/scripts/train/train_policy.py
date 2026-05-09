#!/usr/bin/env python3
"""训练人类策略网络（ResNet shared trunk + policy CE；可选 attack/danger/tactical 辅助头）。

数据源：**XRSH**（``--train-xrsh-dir`` / ``--val-xrsh-dir``），见 ``crates/xiangqi_dataset``。

辅助头标签由 ``nn.aux_pseudo_labels`` 结合 ``root_fen``/``uci_prefix`` 与合法 UCI 表在线生成（见 ARCHITECTURE）。

固定训练策略：按局采样 batch、水平镜像增强、fen 频数 1/sqrt(n) 降权、合法着上标签平滑、
warmup + 余弦学习率。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
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
    aux_heads_sigmoid_mse,
    policy_cross_entropy,
    value_head_tanh_mse,
)
from nn.xrsh_io import xrsh_dir_is_complete

_TRAIN_SEED = 42
_LABEL_SMOOTHING = 0.08
_WARMUP_EPOCHS = 1
_MIN_LR = 1e-5
_AUX_LOSS_WEIGHT = 0.1
_VALUE_LOSS_WEIGHT = 0.5
_OPTIONAL_HEAD_PREFIXES = (
    "fc_attack.",
    "fc_danger.",
    "fc_tactical.",
    "fc_value.",
)


def _missing_keys_are_optional_heads_only(missing_keys: list[str]) -> bool:
    return all(
        any(k.startswith(p) for p in _OPTIONAL_HEAD_PREFIXES) for k in missing_keys
    )


def _load_policy_state_dict(model: PolicyResNet, raw: dict[str, object]) -> None:
    """兼容单 policy checkpoint → 多头（辅助头键缺失时用初始化权重）。"""
    try:
        model.load_state_dict(raw, strict=True)
        return
    except RuntimeError:
        if not model.aux_heads:
            raise
        inc = model.load_state_dict(raw, strict=False)
        if inc.missing_keys and not _missing_keys_are_optional_heads_only(
            list(inc.missing_keys)
        ):
            raise RuntimeError(
                "checkpoint 与当前模型结构不兼容（缺失键不仅是可选头 fc_* / fc_value）："
                f" {list(inc.missing_keys)}"
            ) from None
        if inc.missing_keys:
            print(
                "checkpoint 非严格加载：以下参数保留初始化 "
                f"（通常为旧版仅 policy 权重→多头迁移）: {list(inc.missing_keys)}"
            )
        if inc.unexpected_keys:
            print(f"checkpoint 非严格加载：忽略未知键: {list(inc.unexpected_keys)}")


def unpack_train_batch(
    batch: tuple[Any, ...], *, aux_heads: bool, value_head: bool
) -> dict[str, Any]:
    if aux_heads and value_head:
        b, m, t, w, ta, td, tt, tv = batch
        return {
            "boards": b,
            "masks": m,
            "targets": t,
            "weights": w,
            "t_atk": ta,
            "t_dan": td,
            "t_tac": tt,
            "t_val": tv,
        }
    if aux_heads:
        b, m, t, w, ta, td, tt = batch
        return {
            "boards": b,
            "masks": m,
            "targets": t,
            "weights": w,
            "t_atk": ta,
            "t_dan": td,
            "t_tac": tt,
        }
    b, m, t, w = batch
    return {"boards": b, "masks": m, "targets": t, "weights": w}


def unpack_val_batch(
    batch: tuple[Any, ...], *, aux_heads: bool, value_head: bool
) -> dict[str, Any]:
    if aux_heads and value_head:
        b, m, t, w, ta, td, tt, tv, pl, sid = batch
        return {
            "boards": b,
            "masks": m,
            "targets": t,
            "weights": w,
            "t_atk": ta,
            "t_dan": td,
            "t_tac": tt,
            "t_val": tv,
            "plies": pl,
            "src_ids": sid,
        }
    if aux_heads:
        b, m, t, w, ta, td, tt, pl, sid = batch
        return {
            "boards": b,
            "masks": m,
            "targets": t,
            "weights": w,
            "t_atk": ta,
            "t_dan": td,
            "t_tac": tt,
            "plies": pl,
            "src_ids": sid,
        }
    b, m, t, w, pl, sid = batch
    return {
        "boards": b,
        "masks": m,
        "targets": t,
        "weights": w,
        "plies": pl,
        "src_ids": sid,
    }


class GameGroupedBatchSampler:
    """先随机打乱局顺序，再按局内行序串联后切块；每 epoch 调用 set_epoch 以换序。"""

    def __init__(
        self,
        batch_size: int,
        *,
        rows: list[dict],
        drop_last: bool = False,
        seed: int = _TRAIN_SEED,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.rows = rows
        gid_to_idx: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            gid = str(row.get("game_id", "")) or f"__row_{i}"
            gid_to_idx[gid].append(i)
        self._groups = list(gid_to_idx.items())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups = [(g, list(idxs)) for g, idxs in self._groups]
        rng.shuffle(groups)
        stream: list[int] = []
        for _, idxs in groups:
            stream.extend(idxs)
        bs = self.batch_size
        for i in range(0, len(stream), bs):
            chunk = stream[i : i + bs]
            if len(chunk) < bs and self.drop_last:
                continue
            yield chunk

    def __len__(self) -> int:
        n = sum(len(idxs) for _, idxs in self._groups)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def _lr_scheduler(opt: AdamW, *, epochs: int):
    w = _WARMUP_EPOCHS
    if w >= epochs:
        return CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=_MIN_LR)
    warm = LinearLR(opt, start_factor=1e-2, end_factor=1.0, total_iters=w)
    cos = CosineAnnealingLR(
        opt, T_max=max(1, epochs - w), eta_min=_MIN_LR
    )
    return SequentialLR(opt, [warm, cos], milestones=[w])


def _assert_ckpt_compatible(
    ckpt: dict,
    *,
    moves: list[str],
    n_moves: int,
    width: int,
    blocks: int,
) -> None:
    if int(ckpt.get("n_moves", -1)) != n_moves:
        raise ValueError(
            f"checkpoint n_moves={ckpt.get('n_moves')} 与当前词表长度 {n_moves} 不一致"
        )
    cm = ckpt.get("moves")
    if cm is not None and cm != moves:
        raise ValueError("checkpoint 中的 moves 列表与 --vocab 不一致，无法续训")
    if int(ckpt.get("width", -1)) != width:
        raise ValueError(
            f"checkpoint width={ckpt.get('width')} 与 --width={width} 不一致"
        )
    if int(ckpt.get("blocks", -1)) != blocks:
        raise ValueError(
            f"checkpoint blocks={ckpt.get('blocks')} 与 --blocks={blocks} 不一致"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PolicyResNet（XRSH 数据）")
    ap.add_argument(
        "--train-xrsh-dir",
        type=Path,
        required=True,
        help="训练 XRSH 目录（pack_meta.json + shard_*.xrsh）",
    )
    ap.add_argument(
        "--val-xrsh-dir",
        type=Path,
        required=True,
        help="验证 XRSH 目录",
    )
    ap.add_argument("--vocab", type=Path, required=True, help="build_vocab 生成的 JSON")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "checkpoints" / "policy.pt")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--no-aux-heads",
        action="store_true",
        help="不训练 attack/danger/tactical 辅助头（与仅含 fc 的旧 checkpoint 结构一致）",
    )
    ap.add_argument(
        "--aux-loss-weight",
        type=float,
        default=_AUX_LOSS_WEIGHT,
        help="辅助头 MSE 相对 policy CE 的权重",
    )
    ap.add_argument(
        "--value-head",
        action="store_true",
        help="训练 value 头（tanh 后 ∈[-1,1]，监督为 2*attack-1；需开启辅助头）",
    )
    ap.add_argument(
        "--value-loss-weight",
        type=float,
        default=_VALUE_LOSS_WEIGHT,
        help="value MSE 相对 policy CE 的权重",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--num-workers",
        type=int,
        default=min(8, max(0, (os.cpu_count() or 8) - 2)),
        help="DataLoader worker 数；遇 Windows 多进程问题可改为 0",
    )
    args = ap.parse_args()

    if args.value_head and args.no_aux_heads:
        raise SystemExit(
            "--value-head 与辅助头共用数据路径（attack 伪标），不能与 --no-aux-heads 同时使用"
        )

    random.seed(_TRAIN_SEED)
    torch.manual_seed(_TRAIN_SEED)

    vocab_data = json.loads(args.vocab.read_text(encoding="utf-8"))
    moves: list[str] = vocab_data["moves"]
    move_to_idx = {m: i for i, m in enumerate(moves)}
    n_moves = len(moves)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"torch {torch.__version__} | cuda.is_available={torch.cuda.is_available()} "
        f"| device={device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )
    if device.type == "cpu" and args.device == "cuda":
        print(
            "CUDA 不可用，已改用 CPU。若本机有 NVIDIA 显卡，多半是装了 CPU 版 PyTorch，"
            "请按 README「GPU / CUDA」一节重装带 CUDA 的轮子。"
        )

    if not xrsh_dir_is_complete(args.train_xrsh_dir):
        raise FileNotFoundError(f"--train-xrsh-dir 不完整: {args.train_xrsh_dir}")
    if not xrsh_dir_is_complete(args.val_xrsh_dir):
        raise FileNotFoundError(f"--val-xrsh-dir 不完整: {args.val_xrsh_dir}")

    aux_heads = not bool(args.no_aux_heads)
    value_head = bool(args.value_head)

    train_ds = PolicyXrshDataset(
        args.train_xrsh_dir,
        move_to_idx,
        for_training=True,
        with_aux_labels=aux_heads,
        with_value_labels=value_head,
    )
    val_ds = PolicyXrshDataset(
        args.val_xrsh_dir,
        move_to_idx,
        for_training=False,
        with_row_meta=True,
        with_aux_labels=aux_heads,
        with_value_labels=value_head,
    )

    print(f"train rows={len(train_ds)} val rows={len(val_ds)} vocab={n_moves}")
    print(
        "train data: XRSH v1（Rust .xrsh；合法着已物化为下标，训练步无 pyffish / json.loads）"
    )
    print("val data: XRSH v1")
    print(
        "train recipe: game-batch | mirror_p=0.5 | fen weight 1/sqrt(count) | "
        f"label_smooth={_LABEL_SMOOTHING} | lr warmup={_WARMUP_EPOCHS}ep + cosine→{_MIN_LR}"
    )
    if aux_heads:
        print(
            f"aux heads: attack/danger/tactical（pyffish 伪标签）| "
            f"aux_loss_weight={float(args.aux_loss_weight)}"
        )
    else:
        print("aux heads: 关闭（仅 policy）")
    if value_head:
        print(
            f"value head: tanh MSE vs 2*attack-1 | "
            f"value_loss_weight={float(args.value_loss_weight)}"
        )

    nw = args.num_workers
    pm = device.type == "cuda"
    loader_kw: dict = dict(
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=nw > 0,
        prefetch_factor=(2 if nw > 0 else None),
    )
    if nw == 0:
        loader_kw.pop("prefetch_factor", None)
        loader_kw.pop("persistent_workers", None)

    train_bs = GameGroupedBatchSampler(
        batch_size=args.batch_size,
        rows=train_ds.rows,
        drop_last=False,
        seed=_TRAIN_SEED,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_bs,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kw,
    )
    print(
        f"DataLoader train batches/epoch≈{len(train_loader)} "
        f"num_workers={nw} pin_memory={pm}"
    )

    model = PolicyResNet(
        width=args.width,
        num_blocks=args.blocks,
        num_moves=n_moves,
        aux_heads=aux_heads,
        value_head=value_head,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters={n_params:,} (~{n_params * 4 / 1e6:.2f} MiB fp32 权重)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    resume = args.out.is_file()
    start_epoch = 0
    ckpt: dict | None = None
    if resume:
        ckpt = torch.load(args.out, map_location=device)
        _assert_ckpt_compatible(
            ckpt, moves=moves, n_moves=n_moves, width=args.width, blocks=args.blocks
        )
        ck_vh = bool(ckpt.get("value_head", False))
        if ck_vh and not value_head:
            raise SystemExit(
                "checkpoint 含 value 头，续训请加上 --value-head（与保存结构一致）"
            )
        if not ck_vh and value_head:
            print("提示: checkpoint 无 value 头，fc_value 将随机初始化")
        _load_policy_state_dict(model, ckpt["model"])
        start_epoch = int(ckpt.get("completed_epochs", 0))
        print(
            f"续训: 已加载 {args.out} | 已完成 epoch 计数={start_epoch} | "
            f"本次将再训练 {args.epochs} 个 epoch（至 epoch {start_epoch + args.epochs}）"
        )
    lr_schedule_epochs = args.epochs
    if resume and ckpt is not None:
        lr_schedule_epochs = int(ckpt.get("lr_schedule_epochs", args.epochs))

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _lr_scheduler(opt, epochs=max(1, lr_schedule_epochs))
    if resume and ckpt is not None:
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        else:
            print(
                "提示: checkpoint 无 optimizer 状态，已新建 AdamW（学习率从命令行初值开始）"
            )
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            scheduler = _lr_scheduler(opt, epochs=max(1, args.epochs))
            print(
                "提示: checkpoint 无 scheduler 状态，已按本次 --epochs 重建余弦调度（非严格续接曲线）"
            )

    end_epoch = start_epoch + args.epochs

    best_val_loss = float("inf")
    best_epoch = 0
    if resume and ckpt is not None:
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_epoch = int(ckpt.get("best_epoch", 0))

    best_out = args.out.with_name(args.out.stem + ".best" + args.out.suffix)

    for epoch in range(start_epoch, end_epoch):
        train_bs.set_epoch(epoch)
        model.train()
        total = 0.0
        w_sum = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{end_epoch} train"):
            bt = unpack_train_batch(
                batch, aux_heads=aux_heads, value_head=value_head
            )
            boards = bt["boards"].to(device, non_blocking=pm)
            masks = bt["masks"].to(device, non_blocking=pm)
            targets = bt["targets"].to(device, non_blocking=pm)
            weights = bt["weights"].to(device, non_blocking=pm)
            if aux_heads:
                t_atk = bt["t_atk"].to(device, non_blocking=pm)
                t_dan = bt["t_dan"].to(device, non_blocking=pm)
                t_tac = bt["t_tac"].to(device, non_blocking=pm)
            if value_head:
                t_val = bt["t_val"].to(device, non_blocking=pm)
            out = model(boards)
            if aux_heads and value_head:
                logits, p_atk, p_dan, p_tac, p_val = out
            elif aux_heads:
                logits, p_atk, p_dan, p_tac = out
            elif value_head:
                logits, p_val = out
            else:
                logits = out
            loss_p = policy_cross_entropy(
                logits,
                targets,
                masks,
                label_smoothing=_LABEL_SMOOTHING,
                sample_weight=weights,
            )
            loss = loss_p
            if aux_heads:
                loss_a = aux_heads_sigmoid_mse(
                    p_atk,
                    p_dan,
                    p_tac,
                    t_atk,
                    t_dan,
                    t_tac,
                    sample_weight=weights,
                )
                loss = loss + float(args.aux_loss_weight) * loss_a
            if value_head:
                loss_v = value_head_tanh_mse(
                    p_val, t_val, sample_weight=weights
                )
                loss = loss + float(args.value_loss_weight) * loss_v
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * weights.sum().item()
            w_sum += weights.sum().item()
        print(
            f"train loss {total / max(1e-8, w_sum):.4f} lr={opt.param_groups[0]['lr']:.2e}"
        )

        model.eval()
        vloss = 0.0
        vcount = 0
        correct = 0
        v_aux = 0.0
        v_val = 0.0
        val_metrics = ValMetricsState()
        with torch.no_grad():
            for batch in val_loader:
                bv = unpack_val_batch(
                    batch, aux_heads=aux_heads, value_head=value_head
                )
                boards = bv["boards"].to(device, non_blocking=pm)
                masks = bv["masks"].to(device, non_blocking=pm)
                targets = bv["targets"].to(device, non_blocking=pm)
                wvb = bv["weights"].to(device, non_blocking=pm)
                plies = bv["plies"].to(device, non_blocking=pm)
                src_ids = bv["src_ids"].to(device, non_blocking=pm)
                if aux_heads:
                    t_atk = bv["t_atk"].to(device, non_blocking=pm)
                    t_dan = bv["t_dan"].to(device, non_blocking=pm)
                    t_tac = bv["t_tac"].to(device, non_blocking=pm)
                if value_head:
                    t_val = bv["t_val"].to(device, non_blocking=pm)
                out = model(boards)
                if aux_heads and value_head:
                    logits, p_atk, p_dan, p_tac, p_val = out
                elif aux_heads:
                    logits, p_atk, p_dan, p_tac = out
                elif value_head:
                    logits, p_val = out
                else:
                    logits = out
                if aux_heads:
                    v_aux += (
                        aux_heads_sigmoid_mse(
                            p_atk,
                            p_dan,
                            p_tac,
                            t_atk,
                            t_dan,
                            t_tac,
                            sample_weight=wvb,
                        ).item()
                        * boards.size(0)
                    )
                if value_head:
                    v_val += (
                        value_head_tanh_mse(
                            p_val, t_val, sample_weight=wvb
                        ).item()
                        * boards.size(0)
                    )
                loss = policy_cross_entropy(
                    logits,
                    targets,
                    masks,
                    label_smoothing=0.0,
                    sample_weight=wvb,
                )
                vloss += loss.item() * boards.size(0)
                vcount += boards.size(0)
                pred = logits.masked_fill(~masks, float("-inf")).argmax(dim=1)
                correct += (pred == targets).sum().item()
                val_metrics.update_batch(logits, targets, masks, plies, src_ids)
        val_mean = vloss / max(1, vcount)
        aux_tail = ""
        if aux_heads and vcount > 0:
            aux_tail += f" | val_aux_mse {v_aux / vcount:.4f}"
        if value_head and vcount > 0:
            aux_tail += f" | val_value_mse {v_val / vcount:.4f}"
        print(
            f"val loss {val_mean:.4f} acc {correct / max(1, vcount):.4f}{aux_tail}"
        )
        print(format_val_metrics_report(val_metrics, pgn_source_vocab=val_ds.pgn_source_vocab))

        improved = val_mean < best_val_loss
        if improved:
            best_val_loss = val_mean
            best_epoch = epoch + 1

        scheduler.step()

        payload = {
            "model": model.state_dict(),
            "width": args.width,
            "blocks": args.blocks,
            "n_moves": n_moves,
            "moves": moves,
            "aux_heads": aux_heads,
            "value_head": value_head,
            "aux_loss_weight": float(args.aux_loss_weight),
            "value_loss_weight": float(args.value_loss_weight),
            "completed_epochs": epoch + 1,
            "lr_schedule_epochs": lr_schedule_epochs,
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "last_lr": float(opt.param_groups[0]["lr"]),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "last_val_loss": val_mean,
        }
        torch.save(payload, args.out)
        print(f"checkpoint -> {args.out}")
        if improved:
            torch.save(payload, best_out)
            print(
                f"best checkpoint -> {best_out} "
                f"(val_loss={best_val_loss:.4f} epoch={best_epoch}/{end_epoch})"
            )


if __name__ == "__main__":
    main()
