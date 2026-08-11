#!/usr/bin/env python3
"""在同一份 px0 validation manifest 上对比多个 checkpoint 的 formal/policy/WDL loss。

用法示例：
  nn/.venv/Scripts/python.exe nn/scripts/eval/compare_checkpoints.py `
    --px0-version 677 --px0-root C:/work/px0data --val-ratio 0.05 --seed 42 `
    --checkpoints `
      data/checkpoints/x7_v3_b12c512.best.pt `
      data/checkpoints/x7_v3_b12c512_v711.best.pt `
      data/checkpoints/x7_v3_b12c512_v712.best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

NN_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = NN_ROOT.parent
sys.path.insert(0, str(NN_ROOT / "src"))
sys.path.insert(0, str(NN_ROOT / "scripts" / "train"))

from nn import build_model  # noqa: E402
from nn.dataset_px0 import Px0ChunkDataset, Px0DatasetConfig  # noqa: E402
from nn.px0_kaggle import try_load_prepared_px0_version  # noqa: E402
from train_px0 import compute_loss_terms, forward_training, resolve_device  # noqa: E402


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(
        trunk_kind=str(ckpt["trunk_kind"]),
        in_planes=int(ckpt.get("in_planes", 124)),
        width=int(ckpt["width"]),
        blocks=int(ckpt["blocks"]),
        num_moves=int(ckpt["n_moves"]),
        bottleneck_channels=int(ckpt.get("bottleneck_channels", int(ckpt["width"]) // 2)),
        heads=int(ckpt.get("heads", 16)),
        ffn_channels=int(ckpt.get("ffn_channels", int(ckpt["width"]) * 3 // 2)),
        value_head=True,
        moves_left_head=True,
        auxiliary_heads=True,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, ckpt


def run_val_progress(
    model,
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
    label: str,
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
    t0 = time.time()
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
            if batches == 1 or batches % 50 == 0:
                elapsed = time.time() - t0
                avg = sums["formal"] / batches
                limit = "" if max_batches is None else f"/{max_batches}"
                print(
                    f"  [{label}] batch={batches}{limit} formal_avg={avg:.4f} elapsed={elapsed:.0f}s",
                    flush=True,
                )
            if max_batches is not None and batches >= max_batches:
                break
    print(f"  [{label}] done batches={batches} elapsed={time.time() - t0:.0f}s", flush=True)
    return {name: total / batches if batches else float("nan") for name, total in sums.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare checkpoints on one px0 val split")
    ap.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    ap.add_argument("--px0-version", required=True)
    ap.add_argument("--px0-root", type=Path, default=Path(r"C:\work\px0data"))
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--max-batches", type=int, default=None, help="None = full val")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    prepared = try_load_prepared_px0_version(
        str(args.px0_version),
        root=args.px0_root,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    if prepared is None:
        raise SystemExit(
            f"未找到已准备的 px0 {args.px0_version} "
            f"(root={args.px0_root}, val_ratio={args.val_ratio}, seed={args.seed})。"
            f"先跑 prepare_px0.py。"
        )

    device = resolve_device(args.device)
    amp_enabled = bool(args.amp) and device.type == "cuda"
    ds = Px0ChunkDataset(Px0DatasetConfig(file_list_path=prepared.val_manifest, verify_files=False))
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    suggested = max(1, len(ds.files) * 10 // int(args.batch_size))
    print(
        f"val_manifest={prepared.val_manifest} files={len(ds.files)} "
        f"device={device} amp={amp_enabled} max_batches={args.max_batches} "
        f"suggested_regular_val_batches≈{suggested}",
        flush=True,
    )

    loss_kw = dict(
        final_value_loss_weight=0.6,
        moves_left_loss_weight=0.5,
        soft_policy_weight=8.0,
        soft_policy_temperature=4.0,
        root_wdl_loss_weight=0.6,
        amp_enabled=amp_enabled,
        max_batches=args.max_batches,
    )

    rows = []
    for path in args.checkpoints:
        path = path if path.is_absolute() else (REPO_ROOT / path)
        print(f"evaluating {path.name} ...", flush=True)
        model, ckpt = load_model(path, device)
        metrics = run_val_progress(model, loader, device=device, label=path.name, **loss_kw)
        rows.append((path.name, ckpt.get("px0_version"), ckpt.get("completed_steps"), metrics))
        print(
            f"  -> formal={metrics['formal']:.4f} policy={metrics['policy']:.4f} "
            f"wdl={metrics['final_value_ce']:.4f} mlh={metrics['moves_left']:.4f} total={metrics['total']:.4f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    hdr = f"{'ckpt':<36} {'train_v':>7} {'steps':>7} {'formal':>8} {'policy':>8} {'wdl':>8} {'mlh':>8} {'total':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for name, train_v, steps, m in rows:
        print(
            f"{name:<36} {str(train_v):>7} {int(steps):>7} "
            f"{m['formal']:8.4f} {m['policy']:8.4f} {m['final_value_ce']:8.4f} "
            f"{m['moves_left']:8.4f} {m['total']:8.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
