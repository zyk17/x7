#!/usr/bin/env python3
"""将 checkpoint 导出为正式 ONNX：FP16 encoder，FP32 input/heads/outputs。

mixed-fp16 说明（ORT TensorRT）：
- 图内 encoder 权重 FP16，heads FP32，两端用 Cast 隔开；I/O 保持 FLOAT。
- ORT `trt_fp16_enable` 开的是弱类型 BuilderFlag::kFP16，不是「让已有 FP16 权重生效」。
- 不要把 LayerNorm 单独 .float()：ORT LayerNormalization 要求激活与权重同 dtype，
  否则加载失败。数值保护靠引擎侧 `trt_layer_norm_fp32_fallback`。
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nn import KnowledgeModel


class KnowledgeOnnxExport(nn.Module):
    """Fixed `124x10x9 -> 2062 + WDL + moves-left` ONNX wrapper.

    Auxiliary policy/search-value heads are intentionally never called here,
    therefore they are absent from the ONNX graph and have no inference cost.
    """

    def __init__(self, inner: KnowledgeModel, *, mixed_fp16: bool = False) -> None:
        super().__init__()
        # Export precision must not mutate a caller's in-memory training model.
        self.inner = copy.deepcopy(inner)
        self.mixed_fp16 = mixed_fp16
        if mixed_fp16:
            for module in (self.inner.input_embedding, self.inner.blocks):
                module.half()
            self.inner.smolgen_weight.data = self.inner.smolgen_weight.data.half()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        body = self.inner.forward_body(x.half() if self.mixed_fp16 else x)
        logits, value, moves_left = self.inner.forward_formal_heads(body.float() if self.mixed_fp16 else body)
        return logits, torch.softmax(value, dim=1), moves_left


def main() -> None:
    ap = argparse.ArgumentParser(description="checkpoint → ONNX")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="例如 data/x7.onnx")
    ap.add_argument("--precision", choices=("fp32", "mixed-fp16"), default="mixed-fp16")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    width = int(ckpt["width"])
    model = KnowledgeModel(
        width=width,
        num_blocks=int(ckpt["blocks"]),
        heads=int(ckpt["heads"]),
        ffn_channels=int(ckpt["ffn_channels"]),
        value_head=bool(ckpt.get("value_head")),
        moves_left_head=bool(ckpt.get("moves_left_head")),
        auxiliary_heads=bool(ckpt.get("auxiliary_heads")),
    )
    if not model.value_head or not model.moves_left_head:
        raise SystemExit("当前 ONNX 契约要求 checkpoint 同时包含 WDL 与 moves-left head")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    export_mod = KnowledgeOnnxExport(model, mixed_fp16=args.precision == "mixed-fp16").eval()

    dummy = torch.zeros(1, 124, 10, 9)
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
    print(f"exported -> {args.out} precision={args.precision} b{model.num_blocks}c{width} outputs={out_names}")


if __name__ == "__main__":
    main()
