import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from constants import START_FEN
from nn.fen_tensor import fen_to_planes
from nn import (
    PolicyResNet,
    mix_wdl_targets,
    policy_cross_entropy,
    value_wdl_cross_entropy,
    wdl_logits_to_q,
)


def test_fen_planes_shape():
    t = fen_to_planes(START_FEN)
    assert tuple(t.shape) == (15, 10, 9)


def test_policy_forward_and_masked_loss():
    m = PolicyResNet(in_planes=15, width=32, num_blocks=2, num_moves=16)
    x = fen_to_planes(START_FEN).unsqueeze(0)
    logits = m(x)
    assert logits.shape == (1, 16)
    mask = torch.zeros(1, 16, dtype=torch.bool)
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
    m = PolicyResNet(in_planes=15, width=32, num_blocks=2, num_moves=16, value_head=True)
    x = fen_to_planes(START_FEN).unsqueeze(0)
    logits, value = m(x)
    assert logits.shape == (1, 16)
    assert value.shape == (1, 3)
    tgt = torch.tensor([[0.6, 0.1, 0.3]], dtype=torch.float32)
    loss = value_wdl_cross_entropy(value, tgt)
    assert loss.ndim == 0 and torch.isfinite(loss)
    q = wdl_logits_to_q(value)
    assert q.shape == (1,)


def test_mix_wdl_targets_uses_lc0_q_ratio_semantics():
    winner = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    search = torch.tensor([[0.7, 0.1, 0.2]], dtype=torch.float32)
    assert torch.equal(mix_wdl_targets(winner, search, q_ratio=0.0), winner)
    assert torch.equal(mix_wdl_targets(winner, search, q_ratio=1.0), search)
