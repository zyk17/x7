#!/usr/bin/env python3
"""将 checkpoint 导出为 ONNX（动态 batch）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nn import PolicyResNet


TRUNK_KIND = "x7_v2_bottleneck_gbroadcast"


class PolicyOnnxExport(nn.Module):
    """导出 logits、WDL 概率和 moves-left，与引擎模型契约一致。"""

    def __init__(self, inner: PolicyResNet) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value, moves_left = self.inner(x)
        return logits, torch.softmax(value, dim=1), moves_left


def main() -> None:
    ap = argparse.ArgumentParser(description="checkpoint → ONNX")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="例如 data/x7.onnx")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if str(ckpt.get("trunk_kind")) != TRUNK_KIND:
        raise SystemExit(f"checkpoint 不是当前 {TRUNK_KIND} 架构，不能导出")
    in_planes = int(ckpt.get("in_planes", 124))
    width = int(ckpt["width"])
    blocks = int(ckpt["blocks"])
    bottleneck_channels = int(ckpt.get("bottleneck_channels", width * 7 // 16))
    n_moves = int(ckpt["n_moves"])
    sd = ckpt["model"]
    value_head = bool(ckpt.get("value_head", False))
    moves_left_head = bool(ckpt.get("moves_left_head", False))
    if not value_head and "fc_value.weight" in sd:
        value_head = True
    if not value_head or not moves_left_head:
        raise SystemExit("当前 ONNX 契约要求 checkpoint 同时包含 WDL 与 moves-left head")
    model = PolicyResNet(
        in_planes=in_planes,
        width=width,
        num_blocks=blocks,
        num_moves=n_moves,
        bottleneck_channels=bottleneck_channels,
        value_head=value_head,
        moves_left_head=moves_left_head,
        trunk_kind=TRUNK_KIND,
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    export_mod = PolicyOnnxExport(model)

    dummy = torch.zeros(1, in_planes, 10, 9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_names = ["logits", "value", "moves_left"]
    # dynamo=False：使用 TorchScript 导出路径，无需 onnxscript（PyTorch 2.x 默认 dynamo=True）
    torch.onnx.export(
        export_mod,
        dummy,
        str(args.out),
        input_names=["board"],
        output_names=out_names,
        opset_version=17,
        dynamic_axes={
            "board": {0: "batch"},
            "logits": {0: "batch"},
            "value": {0: "batch"},
            "moves_left": {0: "batch"},
        },
        dynamo=False,
    )
    print(
        f"exported -> {args.out} moves={n_moves} width={width} blocks={blocks} bt={bottleneck_channels} "
        f"in_planes={in_planes} outputs={out_names}（value 已为 WDL 概率）"
    )


if __name__ == "__main__":
    main()
