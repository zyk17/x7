#!/usr/bin/env python3
"""训练人类策略网络（ResNet shared trunk + policy CE；可选 attack/danger/tactical；可选 value）。

数据源：**XRSH**（``--train-dir`` / ``--val-dir``；兼容旧名 ``--train-xrsh-dir`` / ``--val-xrsh-dir``），见 ``crates/xiangqi_dataset``。

辅助头标签来自分片内 Rust 预计算（XRSH v3 必需）。
**value 头默认开启**；监督为结局 × ``progress ** gamma``（``--value-progress-gamma``，默认 1.5）。辅助头默认 **BCEWithLogits**；``--aux-attack-scale`` 可压低 attack 权重（默认 0.25）。

固定训练策略：按局采样 batch、水平镜像增强、fen 频数 1/sqrt(n) 降权、合法着上标签平滑、
warmup + 余弦学习率。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nn.dataset_xrsh import PolicyXrshDataset
from nn.metrics import ValMetricsState, format_val_metrics_report
from nn.model import (
    PolicyResNet,
    aux_heads_sigmoid_bce,
    policy_cross_entropy,
    value_head_tanh_mse,
)
from nn.xrsh_io import xrsh_dir_is_complete

_TRAIN_SEED = 42
_LABEL_SMOOTHING = 0.08
_WARMUP_EPOCHS = 1
_MIN_LR = 1e-5
_AUX_LOSS_WEIGHT = 0.15
_VALUE_LOSS_WEIGHT = 0.4
_OPTIONAL_HEAD_PREFIXES = (
    "fc_attack.",
    "fc_danger.",
    "fc_tactical.",
    "fc_value.",
)


def _default_num_workers() -> int:
    # Windows DataLoader 使用 spawn；大 XRSH Dataset（数百万 Python dict）会被重复序列化到每个 worker，
    # 常见现象是首个 batch 长时间卡在 0/xxxx。这里默认保守设为 0，Linux/macOS 继续给并行默认值。
    if os.name == "nt":
        return 0
    return min(8, max(0, (os.cpu_count() or 8) - 2))


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
        inc = model.load_state_dict(raw, strict=False)
        if inc.missing_keys and not _missing_keys_are_optional_heads_only(
            list(inc.missing_keys)
        ):
            raise RuntimeError(
                "checkpoint 与当前模型结构不兼容（缺失键不仅是可选头 fc_* / fc_value）："
                f" {list(inc.missing_keys)}"
            ) from None
        if inc.unexpected_keys and not _missing_keys_are_optional_heads_only(
            list(inc.unexpected_keys)
        ):
            raise RuntimeError(
                "checkpoint 与当前模型结构不兼容（未知键不仅是可选头 fc_* / fc_value）："
                f" {list(inc.unexpected_keys)}"
            ) from None
        if inc.missing_keys:
            print(
                "checkpoint 非严格加载：以下参数保留初始化 "
                f"（通常为旧版仅 policy 权重→多头迁移）: {list(inc.missing_keys)}"
            )
        if inc.unexpected_keys:
            print(f"checkpoint 非严格加载：忽略未知键: {list(inc.unexpected_keys)}")


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(enabled)


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
    if value_head:
        b, m, t, w, tv = batch
        return {
            "boards": b,
            "masks": m,
            "targets": t,
            "weights": w,
            "t_val": tv,
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
    if value_head:
        b, m, t, w, tv, pl, sid = batch
        return {
            "boards": b,
            "masks": m,
            "targets": t,
            "weights": w,
            "t_val": tv,
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


def _set_optimizer_lr(opt: AdamW, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = float(lr)


def _scheduler_for_resume(
    opt: AdamW, *, total_epochs: int, completed_epochs: int
):
    scheduler = _lr_scheduler(opt, epochs=max(1, total_epochs))
    if completed_epochs > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(completed_epochs):
                scheduler.step()
    return scheduler


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


def _binary_logit_bce_mean(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    safe_tgt = tgt.clamp(eps, 1.0 - eps)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        pred, safe_tgt, reduction="mean"
    )


def _binary_logit_bce_weighted(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    *,
    pos_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    eps = 1e-6
    safe_tgt = tgt.clamp(eps, 1.0 - eps)
    pw = torch.as_tensor(
        float(max(pos_weight, 1e-6)),
        device=pred.device,
        dtype=pred.dtype,
    )
    err = torch.nn.functional.binary_cross_entropy_with_logits(
        pred,
        safe_tgt,
        reduction="none",
        pos_weight=pw,
    )
    if sample_weight is not None:
        if sample_weight.shape != pred.shape:
            raise ValueError("sample_weight 形状须与 pred 一致 [B]")
        err = err * sample_weight
        return err.sum() / sample_weight.sum().clamp(min=1e-8)
    return err.mean()


def _value_tanh_mse_mean(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return ((torch.tanh(pred) - tgt) ** 2).mean()


def _imbalance_pos_weight(mean: float, *, power: float, max_value: float) -> float:
    m = min(max(float(mean), 1e-6), 1.0 - 1e-6)
    raw = ((1.0 - m) / m) ** float(power)
    return min(float(max_value), max(1.0, raw))


def _head_target_stats(train_ds: PolicyXrshDataset) -> dict[str, float]:
    if train_ds.storage_mode != "eager":
        return {}
    assert train_ds.eager_aux is not None
    aux = train_ds.eager_aux.astype(np.float64)
    stats = {
        "attack_mean": float(aux[:, 0].mean()),
        "attack_std": float(aux[:, 0].std()),
        "danger_mean": float(aux[:, 1].mean()),
        "danger_std": float(aux[:, 1].std()),
        "tactical_mean": float(aux[:, 2].mean()),
        "tactical_std": float(aux[:, 2].std()),
    }
    if train_ds.with_value_labels:
        assert train_ds.eager_result_red is not None
        assert train_ds.eager_ply_total is not None
        assert train_ds.eager_plies is not None
        assert train_ds.eager_stms is not None
        gr = train_ds.eager_result_red.astype(np.float64)
        pt = train_ds.eager_ply_total.astype(np.float64)
        ply = train_ds.eager_plies.astype(np.float64)
        stm = train_ds.eager_stms.astype(np.float64)
        outcome_red = np.where(gr == 1.0, 1.0, np.where(gr == -1.0, -1.0, 0.0))
        base = np.where(stm == 1.0, outcome_red, -outcome_red)
        progress = np.where(
            pt <= 1.0,
            1.0,
            np.clip(ply / np.maximum(pt - 1.0, 1.0), 0.0, 1.0),
        )
        vals = base * np.power(progress, float(train_ds._value_progress_gamma))
        abs_vals = np.abs(vals)
        stats.update(
            {
                "value_mean": float(vals.mean()),
                "value_std": float(vals.std()),
                "value_abs_mean": float(abs_vals.mean()),
                "value_zero_mse": float((vals * vals).mean()),
            }
        )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PolicyResNet（XRSH 数据）")
    ap.add_argument(
        "--train-dir",
        "--train-xrsh-dir",
        dest="train_xrsh_dir",
        type=Path,
        required=True,
        help="训练 XRSH 目录（pack_meta.json + shard_*.xrsh）",
    )
    ap.add_argument(
        "--val-dir",
        "--val-xrsh-dir",
        dest="val_xrsh_dir",
        type=Path,
        required=True,
        help="验证 XRSH 目录",
    )
    ap.add_argument("--vocab", type=Path, required=True, help="canonical move_vocab.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "checkpoints" / "policy.pt")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument(
        "--aux-head-hidden-dim",
        type=int,
        default=0,
        help="attack/danger/tactical 头的隐藏层宽度；0=保持单层线性，>0=两层小 MLP",
    )
    ap.add_argument(
        "--value-head-hidden-dim",
        type=int,
        default=0,
        help="value 头的隐藏层宽度；0=保持单层线性，>0=两层小 MLP",
    )
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument(
        "--val-every",
        type=int,
        default=1,
        help="每 N 个 epoch 做一次完整验证；1 = 每轮都验",
    )
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
        default=0.2,
        help="辅助头总权重；若单头权重未显式指定，则 attack/danger/tactical 默认都取这个值",
    )
    ap.add_argument(
        "--attack-loss-weight",
        type=float,
        default=None,
        help="attack 单头权重；默认继承 --aux-loss-weight",
    )
    ap.add_argument(
        "--danger-loss-weight",
        type=float,
        default=None,
        help="danger 单头权重；默认继承 --aux-loss-weight",
    )
    ap.add_argument(
        "--tactical-loss-weight",
        type=float,
        default=None,
        help="tactical 单头权重；默认继承 --aux-loss-weight",
    )
    ap.add_argument(
        "--no-value-head",
        action="store_true",
        help="不训练 value 头（默认训练；仅 policy/辅助头实验时用）",
    )
    ap.add_argument(
        "--value-progress-gamma",
        type=float,
        default=1.5,
        help="value 标签 progress=ply/(ply_total-1) 的指数；越大早局越接近 0（见 temp.md）",
    )
    ap.add_argument(
        "--aux-attack-scale",
        type=float,
        default=0.25,
        help="辅助头 BCE 中 attack 项相对 danger/tactical 的权重倍率（<1 先弱训 attack）",
    )
    ap.add_argument(
        "--aux-pos-weight-power",
        type=float,
        default=0.5,
        help="按训练集均值自动推导 aux 正例重权重时的指数；0=关闭，0.5=平方根平衡",
    )
    ap.add_argument(
        "--aux-pos-weight-max",
        type=float,
        default=8.0,
        help="aux 正例重权重上限，避免 attack/tactical 因均值过低而过冲",
    )
    ap.add_argument(
        "--value-loss-weight",
        type=float,
        default=0.5,
        help="value MSE 相对 policy CE 的权重",
    )
    ap.add_argument(
        "--value-target-weight-alpha",
        type=float,
        default=1.5,
        help="value loss 中按 |target| 提升样本权重：1 + alpha * |target|，缓解大量近 0 目标吞噬监督",
    )
    ap.add_argument(
        "--freeze-trunk",
        action="store_true",
        help="冻结 stem/blocks，仅训练各 head；适合快速做复盘语义读出实验",
    )
    ap.add_argument(
        "--freeze-policy-head",
        action="store_true",
        help="冻结 policy fc；常与 --freeze-trunk 搭配，只训练辅助头 / value 头",
    )
    ap.add_argument(
        "--freeze-value-head",
        action="store_true",
        help="冻结 value 头；供后续在固定 policy+value 后单独训练其它语义头",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dataset-mode",
        choices=("eager", "lazy"),
        default=None,
        help="兼容旧参数：同时设置 train/val 的 XRSH 读取模式；建议改用 --train-dataset-mode / --val-dataset-mode",
    )
    ap.add_argument(
        "--train-dataset-mode",
        choices=("eager", "lazy"),
        default="eager",
        help="训练集 XRSH 读取模式；默认 eager",
    )
    ap.add_argument(
        "--val-dataset-mode",
        choices=("eager", "lazy"),
        default="lazy",
        help="验证集 XRSH 读取模式；默认 lazy",
    )
    ap.add_argument(
        "--amp",
        action="store_true",
        help="CUDA 上启用 AMP（推荐大 batch 训练时开启）",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=_default_num_workers(),
        help="兼容旧参数：同时设置 train/val DataLoader worker 数；建议改用 --train-num-workers / --val-num-workers",
    )
    ap.add_argument(
        "--train-num-workers",
        type=int,
        default=8,
        help="训练集 DataLoader worker 数；默认沿用 --num-workers",
    )
    ap.add_argument(
        "--val-num-workers",
        type=int,
        default=0,
        help="验证集 DataLoader worker 数；默认 0",
    )
    args = ap.parse_args()

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
    value_head = not bool(args.no_value_head)
    attack_loss_weight = float(
        args.attack_loss_weight
        if args.attack_loss_weight is not None
        else args.aux_loss_weight
    )
    danger_loss_weight = float(
        args.danger_loss_weight
        if args.danger_loss_weight is not None
        else args.aux_loss_weight
    )
    tactical_loss_weight = float(
        args.tactical_loss_weight
        if args.tactical_loss_weight is not None
        else args.aux_loss_weight
    )

    train_dataset_mode = (
        args.train_dataset_mode
        or args.dataset_mode
        or "eager"
    )
    val_dataset_mode = (
        args.val_dataset_mode
        or args.dataset_mode
        or "lazy"
    )
    train_num_workers = (
        int(args.train_num_workers)
        if args.train_num_workers is not None
        else int(args.num_workers)
    )
    val_num_workers = (
        int(args.val_num_workers)
        if args.val_num_workers is not None
        else (int(args.num_workers) if args.dataset_mode is not None else 0)
    )

    train_ds = PolicyXrshDataset(
        args.train_xrsh_dir,
        move_to_idx,
        for_training=True,
        with_aux_labels=aux_heads,
        with_value_labels=value_head,
        value_progress_gamma=float(args.value_progress_gamma),
        storage_mode=str(train_dataset_mode),
    )
    val_ds = PolicyXrshDataset(
        args.val_xrsh_dir,
        move_to_idx,
        for_training=False,
        with_row_meta=True,
        with_aux_labels=aux_heads,
        with_value_labels=value_head,
        value_progress_gamma=float(args.value_progress_gamma),
        storage_mode=str(val_dataset_mode),
    )

    print(f"train rows={len(train_ds)} val rows={len(val_ds)} vocab={n_moves}")
    if train_dataset_mode == "eager":
        print(
            "train eager cache: "
            + (
                "hit"
                if bool(getattr(train_ds, "cache_used", False))
                else ("rebuilt" if bool(getattr(train_ds, "cache_built", False)) else "n/a")
            )
        )
    if val_dataset_mode == "eager":
        print(
            "val eager cache: "
            + (
                "hit"
                if bool(getattr(val_ds, "cache_used", False))
                else ("rebuilt" if bool(getattr(val_ds, "cache_built", False)) else "n/a")
            )
        )
    if value_head:
        dropped_train = int(getattr(train_ds, "filtered_unknown_value_rows", 0))
        dropped_val = int(getattr(val_ds, "filtered_unknown_value_rows", 0))
        if dropped_train or dropped_val:
            print(
                "value filtering: 已自动跳过未知结局样本 "
                f"(train={dropped_train}, val={dropped_val})"
            )
    print(
        "train data: XRSH（Rust .xrsh；合法着已物化为下标；v3 含结局字段供 value）"
    )
    print(f"dataset mode: train={train_dataset_mode} | val={val_dataset_mode}")
    print("val data: XRSH（与 train 同 major 版本即可）")
    print(
        "train recipe: game-batch | mirror_p=0.5 | fen weight 1/sqrt(count) | "
        f"label_smooth={_LABEL_SMOOTHING} | lr warmup={_WARMUP_EPOCHS}ep + cosine→{_MIN_LR}"
    )
    head_stats = _head_target_stats(train_ds)
    aux_pos_weight_attack = 1.0
    aux_pos_weight_danger = 1.0
    aux_pos_weight_tactical = 1.0
    if aux_heads and head_stats:
        aux_pos_weight_attack = _imbalance_pos_weight(
            head_stats["attack_mean"],
            power=float(args.aux_pos_weight_power),
            max_value=float(args.aux_pos_weight_max),
        )
        aux_pos_weight_danger = _imbalance_pos_weight(
            head_stats["danger_mean"],
            power=float(args.aux_pos_weight_power),
            max_value=float(args.aux_pos_weight_max),
        )
        aux_pos_weight_tactical = _imbalance_pos_weight(
            head_stats["tactical_mean"],
            power=float(args.aux_pos_weight_power),
            max_value=float(args.aux_pos_weight_max),
        )
    if aux_heads:
        print(
            f"aux heads: BCE | attack_scale={float(args.aux_attack_scale)} | "
            f"head_loss_weights(atk/dan/tac)=({attack_loss_weight:.3f}/"
            f"{danger_loss_weight:.3f}/{tactical_loss_weight:.3f}) | "
            f"aux_head_hidden_dim={int(args.aux_head_hidden_dim)}"
        )
        if head_stats:
            combo_const = (
                float(args.aux_attack_scale)
                * (
                    -head_stats["attack_mean"] * math.log(max(head_stats["attack_mean"], 1e-12))
                    - (1.0 - head_stats["attack_mean"])
                    * math.log(max(1.0 - head_stats["attack_mean"], 1e-12))
                )
                + (
                    -head_stats["danger_mean"] * math.log(max(head_stats["danger_mean"], 1e-12))
                    - (1.0 - head_stats["danger_mean"])
                    * math.log(max(1.0 - head_stats["danger_mean"], 1e-12))
                )
                + (
                    -head_stats["tactical_mean"] * math.log(max(head_stats["tactical_mean"], 1e-12))
                    - (1.0 - head_stats["tactical_mean"])
                    * math.log(max(1.0 - head_stats["tactical_mean"], 1e-12))
                )
            ) / (2.0 + float(args.aux_attack_scale))
            print(
                "aux target stats: "
                f"attack μ={head_stats['attack_mean']:.4f} σ={head_stats['attack_std']:.4f} "
                f"| danger μ={head_stats['danger_mean']:.4f} σ={head_stats['danger_std']:.4f} "
                f"| tactical μ={head_stats['tactical_mean']:.4f} σ={head_stats['tactical_std']:.4f}"
            )
            print(
                "aux loss shaping: "
                f"pos_weight(atk/dan/tac)=({aux_pos_weight_attack:.2f}/"
                f"{aux_pos_weight_danger:.2f}/{aux_pos_weight_tactical:.2f}) "
                f"| const-baseline≈{combo_const:.4f}"
            )
    else:
        print("aux heads: 关闭（仅 policy）")
    if value_head:
        print(
            f"value head: 默认开启 | tanh MSE vs 结局×progress^{float(args.value_progress_gamma):.2f} | "
            f"value_loss_weight={float(args.value_loss_weight)} | "
            f"target_weight_alpha={float(args.value_target_weight_alpha):.2f} | "
            f"value_head_hidden_dim={int(args.value_head_hidden_dim)}"
        )
        if head_stats and "value_zero_mse" in head_stats:
            print(
                "value target stats: "
                f"μ={head_stats['value_mean']:.4f} σ={head_stats['value_std']:.4f} "
                f"| mean|v|={head_stats['value_abs_mean']:.4f} "
                f"| zero-baseline-mse≈{head_stats['value_zero_mse']:.4f}"
            )
    else:
        print("value head: 已用 --no-value-head 关闭")

    if os.name == "nt" and train_num_workers > 0:
        print(
            "警告: Windows 上 train_num_workers>0 会对大 XRSH Dataset 走 spawn 多进程复制；"
            "若首个 batch 长时间卡在 0/xxxx，建议先降 train worker 或改 lazy"
        )
    pm = device.type == "cuda"

    def _loader_kw(nw: int) -> dict[str, Any]:
        out: dict[str, Any] = dict(
            num_workers=nw,
            pin_memory=pm,
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
        drop_last=False,
        seed=_TRAIN_SEED,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_bs,
        **_loader_kw(train_num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **_loader_kw(val_num_workers),
    )
    print(
        f"DataLoader train batches/epoch≈{len(train_loader)} "
        f"train_num_workers={train_num_workers} val_num_workers={val_num_workers} pin_memory={pm}"
    )
    amp_enabled = bool(args.amp) and device.type == "cuda"
    print(f"amp: {'on' if amp_enabled else 'off'} | val_every={int(args.val_every)}")

    model = PolicyResNet(
        width=args.width,
        num_blocks=args.blocks,
        num_moves=n_moves,
        aux_heads=aux_heads,
        value_head=value_head,
        aux_head_hidden_dim=int(args.aux_head_hidden_dim),
        value_head_hidden_dim=int(args.value_head_hidden_dim),
    ).to(device)
    if args.freeze_trunk:
        _set_requires_grad(model.stem, False)
        _set_requires_grad(model.blocks, False)
    if args.freeze_policy_head:
        _set_requires_grad(model.fc, False)
    if args.freeze_value_head and hasattr(model, "fc_value"):
        _set_requires_grad(model.fc_value, False)
    n_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"parameters={n_params:,} (~{n_params * 4 / 1e6:.2f} MiB fp32 权重)"
        f" | trainable={trainable_params:,}"
    )

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
                "checkpoint 含 value 头，续训请勿使用 --no-value-head（与保存结构一致）"
            )
        if not ck_vh and value_head:
            print("提示: checkpoint 无 value 头，fc_value 将随机初始化")
        _load_policy_state_dict(model, ckpt["model"])
        start_epoch = int(ckpt.get("completed_epochs", 0))
        print(
            f"续训: 已加载 {args.out} | 已完成 epoch 计数={start_epoch} | "
            f"本次将再训练 {args.epochs} 个 epoch（至 epoch {start_epoch + args.epochs}）"
        )
    end_epoch = start_epoch + args.epochs
    lr_schedule_epochs = end_epoch
    if resume and ckpt is not None:
        lr_schedule_epochs = max(
            end_epoch, int(ckpt.get("lr_schedule_epochs", end_epoch))
        )

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise SystemExit("当前设置下没有可训练参数；请检查 freeze 选项")
    opt = AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _lr_scheduler(opt, epochs=max(1, lr_schedule_epochs))
    if resume and ckpt is not None:
        optimizer_state_loaded = False
        if "optimizer" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer"])
                optimizer_state_loaded = True
            except ValueError:
                print(
                    "提示: checkpoint 的 optimizer 参数组与当前实验不一致 "
                    "（常见于 freeze / head 结构变化）；已忽略旧 optimizer，改用当前实验新建状态"
                )
        else:
            print(
                "提示: checkpoint 无 optimizer 状态，已新建 AdamW（学习率从命令行初值开始）"
            )
        _set_optimizer_lr(opt, float(args.lr))
        scheduler = _scheduler_for_resume(
            opt,
            total_epochs=lr_schedule_epochs,
            completed_epochs=start_epoch,
        )
        if optimizer_state_loaded:
            print(
                "提示: 续训已按总 epoch 重建 scheduler，避免原余弦周期在到达最小 lr 后回升"
            )
        else:
            print(
                "提示: 已基于当前实验参数重建 optimizer/scheduler；模型权重保留，优化器状态从头开始"
            )

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
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
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
                    loss_atk = _binary_logit_bce_weighted(
                        p_atk,
                        t_atk,
                        pos_weight=aux_pos_weight_attack,
                        sample_weight=weights,
                    )
                    loss_dan = _binary_logit_bce_weighted(
                        p_dan,
                        t_dan,
                        pos_weight=aux_pos_weight_danger,
                        sample_weight=weights,
                    )
                    loss_tac = _binary_logit_bce_weighted(
                        p_tac,
                        t_tac,
                        pos_weight=aux_pos_weight_tactical,
                        sample_weight=weights,
                    )
                    loss = (
                        loss
                        + attack_loss_weight * float(args.aux_attack_scale) * loss_atk
                        + danger_loss_weight * loss_dan
                        + tactical_loss_weight * loss_tac
                    )
                if value_head:
                    loss_v = value_head_tanh_mse(
                        p_val,
                        t_val,
                        target_weight_alpha=float(args.value_target_weight_alpha),
                        sample_weight=weights,
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

        should_validate = ((epoch + 1) % max(1, int(args.val_every)) == 0) or (epoch + 1 == end_epoch)
        val_mean = float("nan")
        if should_validate:
            model.eval()
            vloss = 0.0
            vcount = 0
            correct = 0
            v_aux = 0.0
            v_atk = 0.0
            v_dan = 0.0
            v_tac = 0.0
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
                    autocast_ctx = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if amp_enabled
                        else nullcontext()
                    )
                    with autocast_ctx:
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
                            vatk = _binary_logit_bce_mean(p_atk, t_atk).item()
                            vdan = _binary_logit_bce_mean(p_dan, t_dan).item()
                            vtac = _binary_logit_bce_mean(p_tac, t_tac).item()
                            v_atk += vatk * boards.size(0)
                            v_dan += vdan * boards.size(0)
                            v_tac += vtac * boards.size(0)
                            v_aux += (
                                attack_loss_weight * float(args.aux_attack_scale) * vatk
                                + danger_loss_weight * vdan
                                + tactical_loss_weight * vtac
                            ) * boards.size(0)
                        if value_head:
                            v_val += (
                                _value_tanh_mse_mean(p_val, t_val).item()
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
                aux_tail += (
                    f" | val_aux_bce {v_aux / vcount:.4f}"
                    f" | atk {v_atk / vcount:.4f}"
                    f" | dan {v_dan / vcount:.4f}"
                    f" | tac {v_tac / vcount:.4f}"
                )
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
        else:
            print(f"skip val: epoch {epoch+1}/{end_epoch}（--val-every={int(args.val_every)}）")
            improved = False

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
            "attack_loss_weight": attack_loss_weight,
            "danger_loss_weight": danger_loss_weight,
            "tactical_loss_weight": tactical_loss_weight,
            "aux_attack_scale": float(args.aux_attack_scale),
            "aux_pos_weight_power": float(args.aux_pos_weight_power),
            "aux_pos_weight_max": float(args.aux_pos_weight_max),
            "value_loss_weight": float(args.value_loss_weight),
            "value_progress_gamma": float(args.value_progress_gamma),
            "value_target_weight_alpha": float(args.value_target_weight_alpha),
            "aux_head_hidden_dim": int(args.aux_head_hidden_dim),
            "value_head_hidden_dim": int(args.value_head_hidden_dim),
            "freeze_trunk": bool(args.freeze_trunk),
            "freeze_policy_head": bool(args.freeze_policy_head),
            "freeze_value_head": bool(args.freeze_value_head),
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
