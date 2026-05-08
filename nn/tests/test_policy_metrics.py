import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nn.metrics import (
    ValMetricsState,
    metric_tensors,
    per_sample_entropy_bits,
    per_sample_nll,
    per_sample_topk_hit,
)


def test_entropy_uniform_two_legal_moves():
    # 两个合法着等 logit → 熵应为 ln(2)
    logits = torch.tensor([[0.0, 0.0, -1e9, -1e9]], dtype=torch.float32)
    mask = torch.tensor([[True, True, False, False]])
    ent = per_sample_entropy_bits(logits, mask)
    assert ent.shape == (1,)
    torch.testing.assert_close(ent[0], torch.tensor(math.log(2.0)), rtol=1e-4, atol=1e-4)


def test_nll_peaked_on_target():
    logits = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True]])
    targets = torch.tensor([0])
    nll = per_sample_nll(logits, targets, mask)
    assert nll[0] < 0.05


def test_topk_hit():
    logits = torch.tensor(
        [
            [5.0, 4.0, 3.0, 2.0],
            [1.0, 10.0, 2.0, 3.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
    targets = torch.tensor([0, 0])
    h3 = per_sample_topk_hit(logits, targets, mask, k=3)
    assert h3[0].item() == 1.0
    # 合法着排序：1,2,0；top2 为 {1,2}，不含 target 0
    h2 = per_sample_topk_hit(logits, targets, mask, k=2)
    assert h2[1].item() == 0.0


def test_metric_tensors_match_scalar_helpers():
    logits = torch.tensor(
        [
            [5.0, 4.0, 3.0, 2.0],
            [1.0, 10.0, 2.0, 3.0],
            [0.0, -1.0, -2.0, -3.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, False],
            [True, False, False, False],
        ]
    )
    targets = torch.tensor([0, 0, 0])

    nll, ent, top1, top3, top5 = metric_tensors(logits, targets, mask)
    torch.testing.assert_close(nll, per_sample_nll(logits, targets, mask))
    torch.testing.assert_close(ent, per_sample_entropy_bits(logits, mask))
    torch.testing.assert_close(top3, per_sample_topk_hit(logits, targets, mask, k=3))
    torch.testing.assert_close(top5, per_sample_topk_hit(logits, targets, mask, k=5))
    torch.testing.assert_close(top1, per_sample_topk_hit(logits, targets, mask, k=1))


def test_val_metrics_state_groups_match_expected_means():
    logits = torch.tensor(
        [
            [5.0, 4.0, 3.0, 2.0],
            [1.0, 10.0, 2.0, 3.0],
            [0.0, -1.0, -2.0, -3.0],
            [0.5, 0.2, 0.1, -1.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, False],
            [True, False, False, False],
            [True, True, True, False],
        ]
    )
    targets = torch.tensor([0, 0, 0, 2])
    plies = torch.tensor([5, 25, 45, 80])
    src_ids = torch.tensor([0, 1, 1, 0])

    state = ValMetricsState()
    state.update_batch(logits, targets, mask, plies, src_ids)

    nll, ent, top1, top3, top5 = metric_tensors(logits, targets, mask)
    assert state.overall.n == 4
    torch.testing.assert_close(torch.tensor(state.overall.sum_nll), nll.sum())
    torch.testing.assert_close(torch.tensor(state.overall.sum_entropy), ent.sum())
    torch.testing.assert_close(torch.tensor(state.overall.sum_top1), top1.sum())
    torch.testing.assert_close(torch.tensor(state.overall.sum_top3), top3.sum())
    torch.testing.assert_close(torch.tensor(state.overall.sum_top5), top5.sum())

    for pb in range(4):
        t = state.by_ply_bin[pb]
        assert t.n == 1

    assert state.by_source_id[0].n == 2
    assert state.by_source_id[1].n == 2
