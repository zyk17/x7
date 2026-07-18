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
    """导出用包装：value 导出为 WDL 概率，与引擎消费语义一致。"""

    def __init__(self, inner: PolicyResNet) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        out = self.inner(x)
        if isinstance(out, torch.Tensor):
            return out
        if len(out) == 2:
            logits, value = out
        elif len(out) == 3:
            logits, value, _moves_left = out
        else:
            raise TypeError(f"unexpected model output len={len(out)}")
        return logits, torch.softmax(value, dim=1)


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
    out_names = ["logits"]
    if value_head:
        out_names.append("value")
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
            **({"value": {0: "batch"}} if value_head else {}),
        },
        dynamo=False,
    )
    tail = "（value 已为 WDL 概率）" if value_head else ""
    print(
        f"exported -> {args.out} moves={n_moves} width={width} blocks={blocks} bt={bottleneck_channels} "
        f"in_planes={in_planes} outputs={out_names}{tail}"
    )


if __name__ == "__main__":
    main()
