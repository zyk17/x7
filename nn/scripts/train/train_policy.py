#!/usr/bin/env python3
"""训练人类策略网络（ResNet policy + masked CE）。

固定训练策略（无开关）：按局采样 batch、水平镜像增强、fen 频数 1/sqrt(n) 降权、
合法着上标签平滑、warmup + 余弦学习率。仅需改数据路径与模型/优化器体量参数。

默认：若 ``--out`` 已存在 PyTorch checkpoint，则加载权重（及优化器/调度器若存在）并续训。
需要从头训练时，删除该输出文件后再运行即可。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nn import PolicyJsonlDataset, PolicyJsonlMmapDataset
from nn.dataset_packed import PolicyPackedMmapDataset
from nn import (
    SAMPLER_ORDER,
    SAMPLER_SEG_PTR,
    index_dir_is_complete,
    index_sampler_is_complete,
)
from nn.policy_pack import pack_dir_is_complete
from nn.metrics import ValMetricsState, format_val_metrics_report
from nn import PolicyResNet, policy_cross_entropy

_TRAIN_SEED = 42
_LABEL_SMOOTHING = 0.08
_WARMUP_EPOCHS = 1
_MIN_LR = 1e-5


class GameGroupedBatchSampler:
    """先随机打乱局顺序，再按局内行序串联后切块；每 epoch 调用 set_epoch 以换序。

    ``index_dir`` 模式：使用索引内的 ``sampler_order`` / ``sampler_seg_ptr`` mmap，
    流式产出 batch，不把全量行下标展开成 Python ``list[int]``。
    """

    def __init__(
        self,
        batch_size: int,
        *,
        rows: list[dict] | None = None,
        index_dir: Path | None = None,
        drop_last: bool = False,
        seed: int = _TRAIN_SEED,
    ) -> None:
        n_modes = (rows is not None) + (index_dir is not None)
        if n_modes != 1:
            raise ValueError("GameGroupedBatchSampler 需且仅需指定 rows 或 index_dir 之一")
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.rows = rows
        self._index_dir = Path(index_dir) if index_dir else None
        self._sampler_mmaps: tuple[np.ndarray, np.ndarray] | None = None
        self._n_samples: int | None = None
        self._groups: list[tuple[str, list[int]]] | None = None
        if rows is not None:
            gid_to_idx: dict[str, list[int]] = defaultdict(list)
            for i, row in enumerate(rows):
                gid = str(row.get("game_id", "")) or f"__row_{i}"
                gid_to_idx[gid].append(i)
            self._groups = list(gid_to_idx.items())
        else:
            assert self._index_dir is not None
            ptr = np.load(self._index_dir / SAMPLER_SEG_PTR, mmap_mode="r")
            self._n_samples = int(ptr[-1])

    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        d["_sampler_mmaps"] = None
        return d

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._sampler_mmaps = None

    def _ensure_sampler_mmaps(self) -> None:
        if self._sampler_mmaps is not None or self._index_dir is None:
            return
        d = self._index_dir
        o = np.load(d / SAMPLER_ORDER, mmap_mode="r")
        p = np.load(d / SAMPLER_SEG_PTR, mmap_mode="r")
        self._sampler_mmaps = (o, p)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        if self._groups is not None:
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
            return

        self._ensure_sampler_mmaps()
        assert self._sampler_mmaps is not None and self._n_samples is not None
        order, ptr = self._sampler_mmaps
        n_seg = len(ptr) - 1
        seg_order = list(range(n_seg))
        rng.shuffle(seg_order)
        bs = self.batch_size
        batch: list[int] = []
        for si in seg_order:
            lo = int(ptr[si])
            hi = int(ptr[si + 1])
            for j in range(lo, hi):
                batch.append(int(order[j]))
                if len(batch) >= bs:
                    yield batch
                    batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self._groups is not None:
            n = sum(len(idxs) for _, idxs in self._groups)
        else:
            n = int(self._n_samples or 0)
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
    ap = argparse.ArgumentParser(description="Train PolicyResNet（固定训练策略）")
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument(
        "--train-index-dir",
        type=Path,
        default=None,
        help="训练集行索引目录（build_jsonl_index）；指定则 mmap 读 JSONL，不把整表载入内存",
    )
    ap.add_argument(
        "--val-index-dir",
        type=Path,
        default=None,
        help="验证集行索引目录；指定则 mmap 读 JSONL",
    )
    ap.add_argument(
        "--train-pack-dir",
        type=Path,
        default=None,
        help="materialize_policy_pack 输出目录；优先于 --train-index-dir（训练步不跑 pyffish）",
    )
    ap.add_argument(
        "--val-pack-dir",
        type=Path,
        default=None,
        help="验证集离线包目录；优先于 --val-index-dir",
    )
    ap.add_argument("--vocab", type=Path, required=True, help="build_vocab 生成的 JSON")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "checkpoints" / "policy.pt")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--num-workers",
        type=int,
        default=min(8, max(0, (os.cpu_count() or 8) - 2)),
        help="DataLoader worker 数；遇 Windows 多进程问题可改为 0",
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

    if args.train_pack_dir is not None:
        if not pack_dir_is_complete(args.train_pack_dir):
            raise FileNotFoundError(f"--train-pack-dir 不完整: {args.train_pack_dir}")
        train_ds = PolicyPackedMmapDataset(
            args.train_pack_dir, move_to_idx, for_training=True
        )
    elif args.train_index_dir is not None:
        if not index_dir_is_complete(args.train_index_dir):
            raise FileNotFoundError(
                f"--train-index-dir 不完整（缺少 npy/json）: {args.train_index_dir}"
            )
        if not index_sampler_is_complete(args.train_index_dir):
            raise FileNotFoundError(
                f"--train-index-dir 缺少按局采样文件（sampler_order.npy / sampler_seg_ptr.npy），"
                f"请用当前仓库的 build_jsonl_index 重建: {args.train_index_dir}"
            )
        train_ds = PolicyJsonlMmapDataset(
            args.train_jsonl,
            args.train_index_dir,
            move_to_idx,
            for_training=True,
        )
    else:
        train_ds = PolicyJsonlDataset(
            args.train_jsonl, move_to_idx, for_training=True
        )

    if args.val_pack_dir is not None:
        if not pack_dir_is_complete(args.val_pack_dir):
            raise FileNotFoundError(f"--val-pack-dir 不完整: {args.val_pack_dir}")
        val_ds = PolicyPackedMmapDataset(
            args.val_pack_dir, move_to_idx, for_training=False, with_row_meta=True
        )
    elif args.val_index_dir is not None:
        if not index_dir_is_complete(args.val_index_dir):
            raise FileNotFoundError(
                f"--val-index-dir 不完整（缺少 npy/json）: {args.val_index_dir}"
            )
        val_ds = PolicyJsonlMmapDataset(
            args.val_jsonl,
            args.val_index_dir,
            move_to_idx,
            for_training=False,
            with_row_meta=True,
        )
    else:
        val_ds = PolicyJsonlDataset(
            args.val_jsonl, move_to_idx, for_training=False, with_row_meta=True
        )

    print(f"train rows={len(train_ds)} val rows={len(val_ds)} vocab={n_moves}")
    if isinstance(train_ds, PolicyPackedMmapDataset):
        print("train data: policy pack mmap（物化后训练步无 pyffish / json.loads）")
    elif isinstance(train_ds, PolicyJsonlMmapDataset):
        print(
            "train data: mmap + index（fen 1/sqrt 权重须在建索引时 --weight-by-fen 写入）"
        )
    if isinstance(val_ds, PolicyPackedMmapDataset):
        print("val data: policy pack mmap")
    elif isinstance(val_ds, PolicyJsonlMmapDataset):
        print("val data: mmap + index")
    print(
        "train recipe: game-batch | mirror_p=0.5 | fen weight 1/sqrt(count) | "
        f"label_smooth={_LABEL_SMOOTHING} | lr warmup={_WARMUP_EPOCHS}ep + cosine→{_MIN_LR}"
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

    if isinstance(train_ds, PolicyPackedMmapDataset):
        train_bs = GameGroupedBatchSampler(
            batch_size=args.batch_size,
            index_dir=args.train_pack_dir,
            drop_last=False,
            seed=_TRAIN_SEED,
        )
    elif isinstance(train_ds, PolicyJsonlMmapDataset):
        train_bs = GameGroupedBatchSampler(
            batch_size=args.batch_size,
            index_dir=args.train_index_dir,
            drop_last=False,
            seed=_TRAIN_SEED,
        )
    else:
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

    model = PolicyResNet(width=args.width, num_blocks=args.blocks, num_moves=n_moves).to(device)
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
        model.load_state_dict(ckpt["model"])
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

    for epoch in range(start_epoch, end_epoch):
        train_bs.set_epoch(epoch)
        model.train()
        total = 0.0
        w_sum = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{end_epoch} train"):
            boards, masks, targets, weights = batch
            boards = boards.to(device, non_blocking=pm)
            masks = masks.to(device, non_blocking=pm)
            targets = targets.to(device, non_blocking=pm)
            weights = weights.to(device, non_blocking=pm)
            logits = model(boards)
            loss = policy_cross_entropy(
                logits,
                targets,
                masks,
                label_smoothing=_LABEL_SMOOTHING,
                sample_weight=weights,
            )
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
        val_metrics = ValMetricsState()
        with torch.no_grad():
            for boards, masks, targets, wv, plies, src_ids in val_loader:
                boards = boards.to(device, non_blocking=pm)
                masks = masks.to(device, non_blocking=pm)
                targets = targets.to(device, non_blocking=pm)
                wvb = wv.to(device, non_blocking=pm)
                plies = plies.to(device, non_blocking=pm)
                src_ids = src_ids.to(device, non_blocking=pm)
                logits = model(boards)
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
        print(
            f"val loss {vloss / max(1, vcount):.4f} acc {correct / max(1, vcount):.4f}"
        )
        print(format_val_metrics_report(val_metrics, pgn_source_vocab=val_ds.pgn_source_vocab))

        scheduler.step()

        torch.save(
            {
                "model": model.state_dict(),
                "width": args.width,
                "blocks": args.blocks,
                "n_moves": n_moves,
                "moves": moves,
                "completed_epochs": epoch + 1,
                "lr_schedule_epochs": lr_schedule_epochs,
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            args.out,
        )
        print(f"checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
