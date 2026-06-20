import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from train_policy import _set_requires_grad, unpack_train_batch, unpack_val_batch
from nn.model import PolicyResNet


def test_unpack_train_batch_modes():
    b = torch.zeros(2, 15, 10, 9)
    m = torch.zeros(2, 8, dtype=torch.bool)
    t = torch.zeros(2, dtype=torch.long)
    w = torch.ones(2)
    tv = torch.zeros(2)
    vp = torch.zeros(2, 8)
    sq = torch.zeros(2)
    sv = torch.zeros(2, dtype=torch.long)

    out = unpack_train_batch((b, m, t, w), value_head=False, search_policy_head=False)
    assert set(out.keys()) == {"boards", "masks", "targets", "weights"}

    out = unpack_train_batch((b, m, t, w, tv), value_head=True, search_policy_head=False)
    assert set(out.keys()) == {"boards", "masks", "targets", "weights", "t_val"}

    out = unpack_train_batch((b, m, t, w, vp, sq, sv), value_head=False, search_policy_head=True)
    assert set(out.keys()) == {
        "boards",
        "masks",
        "targets",
        "weights",
        "visit_target",
    }

    out = unpack_train_batch((b, m, t, w, tv, vp, sq, sv), value_head=True, search_policy_head=True)
    assert set(out.keys()) == {
        "boards",
        "masks",
        "targets",
        "weights",
        "t_val",
        "visit_target",
    }


def test_unpack_val_batch_modes():
    b = torch.zeros(2, 15, 10, 9)
    m = torch.zeros(2, 8, dtype=torch.bool)
    t = torch.zeros(2, dtype=torch.long)
    w = torch.ones(2)
    tv = torch.zeros(2)
    vp = torch.zeros(2, 8)
    sq = torch.zeros(2)
    sv = torch.zeros(2, dtype=torch.long)
    pl = torch.zeros(2, dtype=torch.long)
    sid = torch.zeros(2, dtype=torch.long)

    out = unpack_val_batch((b, m, t, w, pl, sid), value_head=False, search_policy_head=False)
    assert set(out.keys()) == {"boards", "masks", "targets", "weights", "plies", "src_ids"}

    out = unpack_val_batch((b, m, t, w, tv, pl, sid), value_head=True, search_policy_head=False)
    assert set(out.keys()) == {
        "boards",
        "masks",
        "targets",
        "weights",
        "t_val",
        "plies",
        "src_ids",
    }

    out = unpack_val_batch((b, m, t, w, vp, sq, sv, pl, sid), value_head=False, search_policy_head=True)
    assert set(out.keys()) == {
        "boards",
        "masks",
        "targets",
        "weights",
        "visit_target",
        "plies",
        "src_ids",
    }

    out = unpack_val_batch((b, m, t, w, tv, vp, sq, sv, pl, sid), value_head=True, search_policy_head=True)
    assert set(out.keys()) == {
        "boards",
        "masks",
        "targets",
        "weights",
        "t_val",
        "visit_target",
        "plies",
        "src_ids",
    }


def test_set_requires_grad_can_freeze_value_head_only():
    model = PolicyResNet(
        width=32,
        num_blocks=2,
        num_moves=16,
        value_head=True,
        value_head_hidden_dim=16,
    )
    _set_requires_grad(model.fc_value, False)
    assert all(not p.requires_grad for p in model.fc_value.parameters())
    assert all(p.requires_grad for p in model.fc.parameters())
