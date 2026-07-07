import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nn import (
    PolicyResNet,
    soft_policy_cross_entropy,
    mix_wdl_targets,
    policy_cross_entropy,
    value_wdl_cross_entropy,
    wdl_logits_to_q,
)


def test_px0_contract_shape():
    x = torch.zeros((124, 10, 9), dtype=torch.float32)
    assert tuple(x.shape) == (124, 10, 9)


def test_policy_forward_and_masked_loss():
    m = PolicyResNet(in_planes=124, width=32, num_blocks=2, num_moves=2062)
    x = torch.zeros((1, 124, 10, 9), dtype=torch.float32)
    logits = m(x)
    assert logits.shape == (1, 2062)
    mask = torch.zeros(1, 2062, dtype=torch.bool)
    mask[0, :8] = True
    tgt = torch.tensor([3])
    loss = policy_cross_entropy(logits, tgt, mask)
    assert loss.ndim == 0 and torch.isfinite(loss)

    loss_s = policy_cross_entropy(
        logits, tgt, mask, label_smoothing=0.1, reduction="mean"
    )
    assert loss_s.ndim == 0 and torch.isfinite(loss_s)

    w = torch.tensor([2.0])
    loss_w = policy_cross_entropy(
        logits, tgt, mask, sample_weight=w, reduction="mean"
    )
    assert loss_w.ndim == 0 and torch.isfinite(loss_w)


def test_value_wdl_forward_and_loss():
    m = PolicyResNet(
        in_planes=124,
        width=32,
        num_blocks=2,
        num_moves=2062,
        value_head=True,
    )
    x = torch.zeros((1, 124, 10, 9), dtype=torch.float32)
    logits, value = m(x)
    assert logits.shape == (1, 2062)
    assert value.shape == (1, 3)
    tgt = torch.tensor([[0.6, 0.1, 0.3]], dtype=torch.float32)
    loss = value_wdl_cross_entropy(value, tgt)
    assert loss.ndim == 0 and torch.isfinite(loss)
    q = wdl_logits_to_q(value)
    assert q.shape == (1,)


def test_attention_policy_head_forward_shape():
    m = PolicyResNet(in_planes=124, width=32, num_blocks=2, num_moves=2062)
    x = torch.zeros((2, 124, 10, 9), dtype=torch.float32)
    logits = m(x)
    assert logits.shape == (2, 2062)
    assert torch.isfinite(logits).all()


def test_soft_policy_cross_entropy_masks_px0_illegal_minus_one_targets():
    logits = torch.zeros((1, 6), dtype=torch.float32)
    target = torch.tensor([[-1.0, 0.25, -1.0, 0.75, -1.0, -1.0]], dtype=torch.float32)
    legal_mask = target >= 0
    loss = soft_policy_cross_entropy(logits, target, legal_mask)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_mix_wdl_targets_uses_lc0_q_ratio_semantics():
    winner = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    search = torch.tensor([[0.7, 0.1, 0.2]], dtype=torch.float32)
    assert torch.equal(mix_wdl_targets(winner, search, q_ratio=0.0), winner)
    assert torch.equal(mix_wdl_targets(winner, search, q_ratio=1.0), search)
