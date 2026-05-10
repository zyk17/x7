import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from constants import START_FEN
from nn.aux_pseudo_labels import (
    pseudo_aux_labels_from_fen,
    pseudo_aux_labels_from_sample,
)
from nn.model import PolicyResNet, aux_heads_sigmoid_mse


def test_pseudo_aux_labels_range_and_start():
    a, d, t = pseudo_aux_labels_from_fen(START_FEN)
    assert 0.0 <= a <= 1.0 and 0.0 <= d <= 1.0 and 0.0 <= t <= 1.0
    # 开局无过河兵、对敌王无直接威胁 → attack 接近 0（与 Rust aux_labels 语义一致）
    assert a < 0.15


def test_pseudo_aux_with_prefix_matches_empty_start():
    a1, d1, t1 = pseudo_aux_labels_from_sample(
        START_FEN, root_fen=START_FEN, uci_prefix=[], legal_uci=None
    )
    a0, d0, t0 = pseudo_aux_labels_from_fen(START_FEN)
    assert abs(a1 - a0) < 1e-6 and abs(d1 - d0) < 1e-6 and abs(t1 - t0) < 1e-6


def test_policy_multi_head_forward_and_aux_loss():
    m = PolicyResNet(width=32, num_blocks=2, num_moves=16, aux_heads=True)
    x = torch.zeros(2, 15, 10, 9)
    logits, pa, pd, pt = m(x)
    assert logits.shape == (2, 16)
    assert pa.shape == (2,) and pd.shape == (2,) and pt.shape == (2,)
    tgt = torch.tensor([0.5, 0.3])
    loss = aux_heads_sigmoid_mse(pa, pd, pt, tgt, tgt, tgt)
    assert loss.ndim == 0 and torch.isfinite(loss)
