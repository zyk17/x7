#!/usr/bin/env python3
"""将 checkpoint 导出为正式 ONNX：FP16 trunk，FP32 input/heads/outputs。

mixed-fp16 说明（ORT TensorRT）：
- 图内 trunk 权重 FP16，heads FP32，两端用 Cast 隔开；I/O 保持 FLOAT。
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nn import CNN_TRUNK_KIND, TRANSFORMER_TRUNK_KIND, KnowledgeResNet, KnowledgeTransformer, build_model


class KnowledgeOnnxExport(nn.Module):
    """Fixed `124x10x9 -> 2062 + WDL + moves-left` ONNX wrapper.

    Auxiliary policy/search-value heads are intentionally never called here,
    therefore they are absent from the ONNX graph and have no inference cost.
    """

    def __init__(self, inner: KnowledgeResNet | KnowledgeTransformer, *, mixed_fp16: bool = False) -> None:
        super().__init__()
        # Export precision must not mutate a caller's in-memory training model.
        self.inner = copy.deepcopy(inner)
        self.mixed_fp16 = mixed_fp16
        if mixed_fp16:
            modules = (
                (
                    self.inner.stem,
                    self.inner.stage1,
                    self.inner.broadcast4,
                    self.inner.stage2,
                    self.inner.broadcast8,
                    self.inner.stage3,
                    self.inner.trunk_bn,
                )
                if self.inner.trunk_kind == CNN_TRUNK_KIND
                else (self.inner.input_embedding, self.inner.blocks)
            )
            for module in modules:
                module.half()
            if self.inner.trunk_kind == TRANSFORMER_TRUNK_KIND:
                self.inner.smolgen_weight.data = self.inner.smolgen_weight.data.half()

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
    trunk_kind = str(ckpt.get("trunk_kind"))
    if trunk_kind not in (CNN_TRUNK_KIND, TRANSFORMER_TRUNK_KIND):
        raise SystemExit(f"checkpoint 的 trunk_kind 不受支持: {trunk_kind}")
    width = int(ckpt["width"])
    model = build_model(
        trunk_kind=trunk_kind,
        in_planes=int(ckpt.get("in_planes", 124)),
        width=width,
        blocks=int(ckpt["blocks"]),
        num_moves=int(ckpt["n_moves"]),
        bottleneck_channels=int(ckpt.get("bottleneck_channels", width // 2)),
        heads=int(ckpt.get("heads", 16)),
        ffn_channels=int(ckpt.get("ffn_channels", width * 3 // 2)),
        value_head=bool(ckpt.get("value_head")),
        moves_left_head=bool(ckpt.get("moves_left_head")),
        auxiliary_heads=bool(ckpt.get("auxiliary_heads")),
    )
    if not model.value_head or not model.moves_left_head:
        raise SystemExit("当前 ONNX 契约要求 checkpoint 同时包含 WDL 与 moves-left head")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    export_mod = KnowledgeOnnxExport(model, mixed_fp16=args.precision == "mixed-fp16").eval()

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
        f"{trunk_kind} b{model.num_blocks}c{width} outputs={out_names}"
    )


if __name__ == "__main__":
    main()
