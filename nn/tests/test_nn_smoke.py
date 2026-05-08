import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from constants import START_FEN
from nn.fen_tensor import fen_to_planes
from nn import PolicyResNet, policy_cross_entropy


def test_fen_planes_shape():
    t = fen_to_planes(START_FEN)
    assert tuple(t.shape) == (15, 10, 9)


def test_policy_forward_and_masked_loss():
    m = PolicyResNet(width=32, num_blocks=2, num_moves=16)
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
