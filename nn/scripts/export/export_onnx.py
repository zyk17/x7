#!/usr/bin/env python3
"""将 train_policy 保存的 checkpoint 导出为 ONNX（静态 batch=1，便于移动端）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nn import PolicyResNet


def main() -> None:
    ap = argparse.ArgumentParser(description="checkpoint → ONNX")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="例如 data/policy.onnx")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    width = int(ckpt["width"])
    blocks = int(ckpt["blocks"])
    n_moves = int(ckpt["n_moves"])
    model = PolicyResNet(width=width, num_blocks=blocks, num_moves=n_moves)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros(1, 15, 10, 9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False：使用 TorchScript 导出路径，无需 onnxscript（PyTorch 2.x 默认 dynamo=True）
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["board"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    print(f"exported -> {args.out} moves={n_moves} width={width} blocks={blocks}")


if __name__ == "__main__":
    main()
