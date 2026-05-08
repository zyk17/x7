#!/usr/bin/env python3
"""将 train_policy 保存的 checkpoint 导出为 ONNX（静态 batch=1，便于移动端）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nn import PolicyResNet


class PolicyOnnxExport(nn.Module):
    """导出用包装：辅助头在图中即 sigmoid(logit)，与训练标签 [0,1] 语义一致。"""

    def __init__(self, inner: PolicyResNet) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        out = self.inner(x)
        if not self.inner.aux_heads:
            return out
        logits, a, d, t = out
        return logits, torch.sigmoid(a), torch.sigmoid(d), torch.sigmoid(t)


def main() -> None:
    ap = argparse.ArgumentParser(description="checkpoint → ONNX")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="例如 data/policy.onnx")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    width = int(ckpt["width"])
    blocks = int(ckpt["blocks"])
    n_moves = int(ckpt["n_moves"])
    sd = ckpt["model"]
    aux_heads = bool(ckpt.get("aux_heads", False))
    if not aux_heads and "fc_attack.weight" in sd:
        aux_heads = True
    model = PolicyResNet(
        width=width, num_blocks=blocks, num_moves=n_moves, aux_heads=aux_heads
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    export_mod = PolicyOnnxExport(model)

    dummy = torch.zeros(1, 15, 10, 9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_names = (
        ["logits", "attack", "danger", "tactical"]
        if aux_heads
        else ["logits"]
    )
    # dynamo=False：使用 TorchScript 导出路径，无需 onnxscript（PyTorch 2.x 默认 dynamo=True）
    torch.onnx.export(
        export_mod,
        dummy,
        str(args.out),
        input_names=["board"],
        output_names=out_names,
        opset_version=17,
        dynamo=False,
    )
    tail = ""
    if aux_heads:
        tail = "（attack/danger/tactical 已为 sigmoid 概率）"
    print(
        f"exported -> {args.out} moves={n_moves} width={width} blocks={blocks} "
        f"outputs={out_names}{tail}"
    )


if __name__ == "__main__":
    main()
