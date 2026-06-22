#!/usr/bin/env python3
"""在带 search_q 的 XRSH 上评估 value 头，输出 CSV。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nn.dataset_xrsh import PolicyXrshDataset
from nn.model import PolicyResNet
from nn.xrsh_io import xrsh_dir_is_complete


def main() -> None:
    ap = argparse.ArgumentParser(description="Value 头评估 CSV")
    ap.add_argument("--xrsh-dir", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    if not xrsh_dir_is_complete(args.xrsh_dir):
        raise FileNotFoundError(f"XRSH 不完整: {args.xrsh_dir}")

    vocab_data = json.loads(args.vocab.read_text(encoding="utf-8"))
    moves: list[str] = vocab_data["moves"]
    move_to_idx = {m: i for i, m in enumerate(moves)}
    idx_to_move = {i: m for i, m in enumerate(moves)}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if not ckpt.get("value_head"):
        raise ValueError("checkpoint 未启用 value_head")

    model = PolicyResNet(
        width=int(ckpt["width"]),
        num_blocks=int(ckpt["blocks"]),
        num_moves=len(moves),
        value_head=True,
        value_head_hidden_dim=int(ckpt.get("value_head_hidden_dim", 0)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    ds = PolicyXrshDataset(
        args.xrsh_dir,
        move_to_idx,
        with_value_labels=True,
        storage_mode="lazy",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = len(ds) if args.max_rows <= 0 else min(len(ds), args.max_rows)
    sqerr_sum = 0.0
    n_labeled = 0

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "row_index",
                "game_id_hint",
                "ply",
                "target_idx",
                "move_uci",
                "fen",
                "pred_value",
                "target_value",
                "search_visits",
                "sqerr",
            ]
        )
        with torch.no_grad():
            for i in range(n):
                board, mask, target, _weight, t_val, search_visits = ds[i]
                if int(search_visits.item()) <= 0:
                    continue
                board = board.unsqueeze(0).to(device)
                logits, pred_value = model(board)
                pred = float(torch.tanh(pred_value[0]).item())
                tgt = float(t_val.item())
                err = (pred - tgt) ** 2
                sqerr_sum += err
                n_labeled += 1
                ti = int(target.item())
                w.writerow(
                    [
                        i,
                        ds.row_group_ids[i] if hasattr(ds, "row_group_ids") else "",
                        int(ds.row_refs[i].ply) if ds.row_refs else "",
                        ti,
                        idx_to_move.get(ti, ""),
                        "",
                        pred,
                        tgt,
                        int(search_visits.item()),
                        err,
                    ]
                )

    mse = sqerr_sum / max(1, n_labeled)
    print(f"labeled_rows={n_labeled} value_mse={mse:.6f} -> {args.out}")


if __name__ == "__main__":
    main()
