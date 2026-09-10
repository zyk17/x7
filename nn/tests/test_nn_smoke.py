import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export import KnowledgeOnnxExport
from nn import KnowledgeModel, moves_left_loss, soft_policy_cross_entropy, value_q_mse_from_wdl, value_wdl_cross_entropy
from nn.model import _load_move_vocab, soften_policy_targets


def make_model(*, auxiliary_heads: bool = False) -> KnowledgeModel:
    return KnowledgeModel(
        width=32,
        num_blocks=2,
        heads=4,
        ffn_channels=96,
        value_head=True,
        moves_left_head=True,
        auxiliary_heads=auxiliary_heads,
    )


def test_policy_vocab_is_packaged_with_python_module() -> None:
    moves = _load_move_vocab()
    assert len(moves) == 2062
    assert moves[0] == "a0a1"


def test_model_keeps_formal_contract_and_training_auxiliaries() -> None:
    model = make_model(auxiliary_heads=True)
    outputs = model(torch.zeros((2, 124, 10, 9)))
    assert isinstance(outputs, tuple) and [output.shape for output in outputs] == [
        (2, 2062),
        (2, 3),
        (2, 1),
        (2, 2062),
        (2, 3),
    ]


def test_onnx_export_keeps_only_formal_heads(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    model = make_model(auxiliary_heads=True).eval()
    out = tmp_path / "x7.onnx"
    torch.onnx.export(
        KnowledgeOnnxExport(model, mixed_fp16=True).eval(),
        torch.zeros((1, 124, 10, 9)),
        str(out),
        input_names=["board"],
        output_names=["logits", "value", "moves_left"],
        opset_version=17,
        dynamo=False,
    )
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    names = {initializer.name for initializer in onnx.load(str(out)).graph.initializer}
    assert not any("soft_policy_head" in name or "root_value_head" in name for name in names)


def test_mixed_fp16_onnx_keeps_layernorm_homogeneous(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import helper

    out = tmp_path / "x7.onnx"
    torch.onnx.export(
        KnowledgeOnnxExport(make_model().eval(), mixed_fp16=True),
        torch.zeros((1, 124, 10, 9)),
        str(out),
        input_names=["board"],
        output_names=["logits", "value", "moves_left"],
        opset_version=17,
        dynamo=False,
    )
    norms = [tensor for tensor in onnx.load(str(out)).graph.initializer if "norm" in tensor.name.lower()]
    assert norms
    assert all(helper.tensor_dtype_to_string(tensor.data_type) == "TensorProto.FLOAT16" for tensor in norms)


def test_formal_losses_are_finite() -> None:
    logits = torch.zeros((1, 4))
    target = torch.tensor([[-1.0, 0.25, -1.0, 0.75]])
    legal = target >= 0
    assert torch.isfinite(soft_policy_cross_entropy(logits, target.clamp_min(0), legal))
    assert torch.allclose(soften_policy_targets(target.clamp_min(0), legal).sum(dim=1), torch.ones(1))
    value = torch.zeros((1, 3))
    assert torch.isfinite(value_wdl_cross_entropy(value, torch.tensor([[0.6, 0.1, 0.3]])))
    assert torch.isfinite(value_q_mse_from_wdl(value, torch.tensor([0.4])))
    assert torch.isfinite(moves_left_loss(torch.zeros((1, 1)), torch.tensor([[24.0]])))
