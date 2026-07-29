import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "export"))

from nn import (
    PolicyResNet,
    soft_policy_cross_entropy,
    moves_left_loss,
    value_q_mse_from_wdl,
    value_wdl_cross_entropy,
    wdl_logits_to_q,
)
from nn.model import GlobalBroadcast, PreActBottleneck, ValueAuxHead, _load_move_vocab, soften_policy_targets
from export_onnx import PolicyOnnxExport


def test_policy_vocab_is_packaged_with_python_module():
    moves = _load_move_vocab()
    assert len(moves) == 2062
    assert moves[0] == "a0a1"


def test_px0_contract_shape():
    x = torch.zeros((124, 10, 9), dtype=torch.float32)
    assert tuple(x.shape) == (124, 10, 9)


def test_policy_forward_shape():
    m = PolicyResNet(in_planes=124, width=32, num_blocks=12, num_moves=2062)
    x = torch.zeros((1, 124, 10, 9), dtype=torch.float32)
    logits = m(x)
    assert logits.shape == (1, 2062)


def test_model_defaults_to_b15c384bt192():
    model = PolicyResNet()
    assert (model.num_blocks, model.stem.out_channels, model.bottleneck_channels) == (15, 384, 192)


def test_value_wdl_forward_and_loss():
    m = PolicyResNet(
        in_planes=124,
        width=32,
        num_blocks=12,
        num_moves=2062,
        value_head=True,
        moves_left_head=True,
    )
    x = torch.zeros((1, 124, 10, 9), dtype=torch.float32)
    logits, value, moves_left = m(x)
    assert logits.shape == (1, 2062)
    assert value.shape == (1, 3)
    assert moves_left.shape == (1, 1)
    tgt = torch.tensor([[0.6, 0.1, 0.3]], dtype=torch.float32)
    loss = value_wdl_cross_entropy(value, tgt)
    assert loss.ndim == 0 and torch.isfinite(loss)
    q = wdl_logits_to_q(value)
    assert q.shape == (1,)
    ml_loss = moves_left_loss(moves_left, torch.tensor([[24.0]], dtype=torch.float32))
    assert ml_loss.ndim == 0 and torch.isfinite(ml_loss)


def test_onnx_export_wrapper_keeps_moves_left_output():
    model = PolicyResNet(width=32, num_blocks=12, value_head=True, moves_left_head=True)
    logits, value, moves_left = PolicyOnnxExport(model)(torch.zeros((1, 124, 10, 9)))
    assert logits.shape == (1, 2062)
    assert value.shape == (1, 3)
    assert moves_left.shape == (1, 1)
    assert torch.allclose(value.sum(dim=1), torch.ones(1))


def test_auxiliary_heads_are_separate_from_formal_heads():
    model = PolicyResNet(width=32, num_blocks=12, value_head=True, moves_left_head=True, auxiliary_heads=True)
    outputs = model(torch.zeros((1, 124, 10, 9)))
    assert isinstance(outputs, tuple) and len(outputs) == 5
    logits, value, moves_left = PolicyOnnxExport(model)(torch.zeros((1, 124, 10, 9)))
    assert logits.shape == (1, 2062) and value.shape == (1, 3) and moves_left.shape == (1, 1)


def test_auxiliary_heads_require_formal_value_heads():
    with pytest.raises(ValueError, match="auxiliary_heads"):
        PolicyResNet(width=32, num_blocks=12, auxiliary_heads=True)


def test_onnx_export_omits_auxiliary_head_weights(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    model = PolicyResNet(width=32, num_blocks=12, value_head=True, moves_left_head=True, auxiliary_heads=True).eval()
    out = tmp_path / "x7.onnx"
    torch.onnx.export(
        PolicyOnnxExport(model, mixed_fp16=True).eval(),
        torch.zeros((1, 124, 10, 9)),
        str(out),
        input_names=["board"],
        output_names=["logits", "value", "moves_left"],
        opset_version=17,
        dynamo=False,
    )
    names = {initializer.name for initializer in onnx.load(str(out)).graph.initializer}
    assert not any("soft_policy_head" in name or "root_value_out" in name for name in names)


def test_pure_cnn_policy_head_forward_shape():
    m = PolicyResNet(in_planes=124, width=32, num_blocks=12, num_moves=2062)
    x = torch.zeros((2, 124, 10, 9), dtype=torch.float32)
    logits = m(x)
    assert logits.shape == (2, 2062)
    assert torch.isfinite(logits).all()


def test_x7_v2_uses_three_stages_and_two_global_broadcasts():
    model = PolicyResNet(in_planes=124, width=32, num_blocks=12, num_moves=2062)
    assert len(model.stage1) == len(model.stage2) == len(model.stage3) == 4
    assert all(isinstance(block, PreActBottleneck) for block in (*model.stage1, *model.stage2, *model.stage3))
    assert isinstance(model.broadcast4, GlobalBroadcast)
    assert isinstance(model.broadcast8, GlobalBroadcast)
    assert model.stage1[0].conv1.kernel_size == (1, 1)
    assert model.stage1[0].conv2.kernel_size == (3, 3)
    assert model.stage1[0].conv3.kernel_size == (3, 3)
    assert model.stage1[0].conv4.kernel_size == (1, 1)
    assert model.stage1[0].conv1.out_channels == 16
    assert model.broadcast4.gpool_conv.kernel_size == (3, 3)
    assert model.broadcast4.gpool_to_bias.in_features == 32


def test_x7_v2_allows_non_baseline_depth_with_evenly_split_stages():
    model = PolicyResNet(in_planes=124, width=32, num_blocks=10, num_moves=2062)
    assert (len(model.stage1), len(model.stage2), len(model.stage3)) == (3, 3, 4)
    logits = model(torch.zeros((1, 124, 10, 9), dtype=torch.float32))
    assert logits.shape == (1, 2062)


def test_x7_v2_allows_explicit_bottleneck_width():
    model = PolicyResNet(in_planes=124, width=32, num_blocks=10, bottleneck_channels=20, num_moves=2062)
    assert model.bottleneck_channels == 20
    assert model.stage1[0].conv1.out_channels == 20


def test_policy_and_shared_value_aux_head_features():
    model = PolicyResNet(
        in_planes=124,
        width=32,
        num_blocks=12,
        num_moves=2062,
        value_head=True,
        moves_left_head=True,
    )
    assert model.policy_head.gpool_conv.kernel_size == (3, 3)
    assert model.policy_head.gpool_to_bias.in_features == 32
    assert isinstance(model.value_aux_head_module, ValueAuxHead)
    assert model.value_aux_head_module.fc.in_features == model.value_aux_head_module.conv.out_channels * 2
    assert model.value_aux_head_module.value_out.out_features == 3
    assert model.value_aux_head_module.moves_left_out.out_features == 1

    _logits, _value, moves_left = model(torch.randn((1, 124, 10, 9)))
    assert torch.all(moves_left >= 0)


def test_x7_v2_256x12bt128_parameter_count_is_stable():
    model = PolicyResNet(
        in_planes=124,
        width=256,
        num_blocks=12,
        num_moves=2062,
        value_head=True,
        moves_left_head=True,
    )
    assert sum(param.numel() for param in model.parameters()) == 6_619_704


def test_soft_policy_cross_entropy_masks_px0_illegal_minus_one_targets():
    logits = torch.zeros((1, 6), dtype=torch.float32)
    target = torch.tensor([[-1.0, 0.25, -1.0, 0.75, -1.0, -1.0]], dtype=torch.float32)
    legal_mask = target >= 0
    loss = soft_policy_cross_entropy(logits, target, legal_mask)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_soft_policy_target_stays_on_legal_moves():
    target = torch.tensor([[-1.0, 0.01, -1.0, 0.99]], dtype=torch.float32)
    legal = target >= 0
    softened = soften_policy_targets(target.clamp_min(0.0), legal, temperature=4.0)
    assert torch.equal(softened[~legal], torch.zeros(2))
    assert torch.allclose(softened.sum(dim=1), torch.ones(1))


def test_wdl_q_metric_is_finite():
    value_logits = torch.tensor([[0.2, -0.1, 0.0]], dtype=torch.float32)
    tgt_q = torch.tensor([[0.4]], dtype=torch.float32)
    loss = value_q_mse_from_wdl(value_logits, tgt_q)
    assert loss.ndim == 0 and torch.isfinite(loss)
