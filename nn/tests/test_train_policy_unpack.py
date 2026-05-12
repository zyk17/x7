import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from train_policy import unpack_train_batch, unpack_val_batch
from train_policy import _set_requires_grad
from nn.model import PolicyResNet


def test_unpack_train_batch_all_modes():
    b = torch.zeros(2, 15, 10, 9)
    m = torch.zeros(2, 8, dtype=torch.bool)
    t = torch.zeros(2, dtype=torch.long)
    w = torch.ones(2)
    ta = torch.zeros(2)
    td = torch.zeros(2)
    tt = torch.zeros(2)
    tv = torch.zeros(2)

    out = unpack_train_batch((b, m, t, w), aux_heads=False, value_head=False)
    assert set(out.keys()) == {"boards", "masks", "targets", "weights"}

    out = unpack_train_batch((b, m, t, w, tv), aux_heads=False, value_head=True)
    assert set(out.keys()) == {"boards", "masks", "targets", "weights", "t_val"}

    out = unpack_train_batch(
        (b, m, t, w, ta, td, tt), aux_heads=True, value_head=False
    )
    assert set(out.keys()) == {
        "boards",
        "masks",
        "targets",
        "weights",
        "t_atk",
        "t_dan",
        "t_tac",
    }

    out = unpack_train_batch(
        (b, m, t, w, ta, td, tt, tv), aux_heads=True, value_head=True
    )
    assert set(out.keys()) == {
        "boards",
        "masks",
        "targets",
        "weights",
        "t_atk",
        "t_dan",
        "t_tac",
        "t_val",
    }


def test_unpack_val_batch_all_modes():
    b = torch.zeros(2, 15, 10, 9)
    m = torch.zeros(2, 8, dtype=torch.bool)
    t = torch.zeros(2, dtype=torch.long)
    w = torch.ones(2)
    ta = torch.zeros(2)
    td = torch.zeros(2)
    tt = torch.zeros(2)
    tv = torch.zeros(2)
    pl = torch.zeros(2, dtype=torch.long)
    sid = torch.zeros(2, dtype=torch.long)

    out = unpack_val_batch((b, m, t, w, pl, sid), aux_heads=False, value_head=False)
    assert set(out.keys()) == {
        "boards",
        "masks",
        "targets",
        "weights",
        "plies",
        "src_ids",
    }

    out = unpack_val_batch(
        (b, m, t, w, tv, pl, sid), aux_heads=False, value_head=True
    )
    assert set(out.keys()) == {
        "boards",
        "masks",
        "targets",
        "weights",
        "t_val",
        "plies",
        "src_ids",
    }

    out = unpack_val_batch(
        (b, m, t, w, ta, td, tt, pl, sid), aux_heads=True, value_head=False
    )
    assert set(out.keys()) == {
        "boards",
        "masks",
        "targets",
        "weights",
        "t_atk",
        "t_dan",
        "t_tac",
        "plies",
        "src_ids",
    }

    out = unpack_val_batch(
        (b, m, t, w, ta, td, tt, tv, pl, sid),
        aux_heads=True,
        value_head=True,
    )
    assert set(out.keys()) == {
        "boards",
        "masks",
        "targets",
        "weights",
        "t_atk",
        "t_dan",
        "t_tac",
        "t_val",
        "plies",
        "src_ids",
    }


def test_set_requires_grad_can_freeze_value_head_only():
    model = PolicyResNet(
        width=32,
        num_blocks=2,
        num_moves=16,
        aux_heads=False,
        value_head=True,
        value_head_hidden_dim=16,
    )
    _set_requires_grad(model.fc_value, False)
    assert all(not p.requires_grad for p in model.fc_value.parameters())
    assert all(p.requires_grad for p in model.fc.parameters())
