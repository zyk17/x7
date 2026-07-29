#!/usr/bin/env python3
"""将 checkpoint 导出为正式 ONNX：FP16 trunk，FP32 input/heads/outputs。"""

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
    """Fixed `124x10x9 -> 2062 + WDL + moves-left` ONNX wrapper.

    Auxiliary policy/search-value heads are intentionally never called here,
    therefore they are absent from the ONNX graph and have no inference cost.
    """

    def __init__(self, inner: PolicyResNet, *, mixed_fp16: bool = False) -> None:
        super().__init__()
        self.inner = inner
        self.mixed_fp16 = mixed_fp16
        if mixed_fp16:
            for module in (
                inner.stem,
                inner.stage1,
                inner.broadcast4,
                inner.stage2,
                inner.broadcast8,
                inner.stage3,
                inner.trunk_bn,
            ):
                module.half()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trunk = self.inner.forward_trunk(x.half() if self.mixed_fp16 else x)
        logits, value, moves_left = self.inner.forward_formal_heads(trunk.float() if self.mixed_fp16 else trunk)
        return logits, torch.softmax(value, dim=1), moves_left


def main() -> None:
    ap = argparse.ArgumentParser(description="checkpoint → ONNX")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="例如 data/x7.onnx")
    ap.add_argument("--precision", choices=("fp32", "mixed-fp16"), default="mixed-fp16")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if str(ckpt.get("trunk_kind")) != TRUNK_KIND:
        raise SystemExit(f"checkpoint 不是当前 {TRUNK_KIND} 架构，不能导出")
    width = int(ckpt["width"])
    model = PolicyResNet(
        in_planes=int(ckpt.get("in_planes", 124)),
        width=width,
        num_blocks=int(ckpt["blocks"]),
        num_moves=int(ckpt["n_moves"]),
        bottleneck_channels=int(ckpt["bottleneck_channels"]),
        value_head=bool(ckpt.get("value_head")),
        moves_left_head=bool(ckpt.get("moves_left_head")),
        auxiliary_heads=bool(ckpt.get("auxiliary_heads")),
        trunk_kind=TRUNK_KIND,
    )
    if not model.value_head or not model.moves_left_head:
        raise SystemExit("当前 ONNX 契约要求 checkpoint 同时包含 WDL 与 moves-left head")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    export_mod = PolicyOnnxExport(model, mixed_fp16=args.precision == "mixed-fp16").eval()

    dummy = torch.zeros(1, model.in_planes, 10, 9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_names = ["logits", "value", "moves_left"]
    torch.onnx.export(
        export_mod,
        dummy,
        str(args.out),
        input_names=["board"],
        output_names=out_names,
        opset_version=17,
        dynamic_axes={"board": {0: "batch"}, "logits": {0: "batch"}, "value": {0: "batch"}, "moves_left": {0: "batch"}},
        dynamo=False,
    )
    print(
        f"exported -> {args.out} precision={args.precision} "
        f"b{model.num_blocks}c{width}bt{model.bottleneck_channels} outputs={out_names}"
    )


if __name__ == "__main__":
    main()
