import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from train_common import set_requires_grad
from train_loss import compute_policy_value_loss
from nn.dataset_batch import (
    SAMPLE_BOARD,
    SAMPLE_BOARD90,
    SAMPLE_LEGAL_IDX,
    SAMPLE_MASK,
    SAMPLE_SEARCH_COUNTS,
    SAMPLE_SEARCH_VISITS,
    SAMPLE_STM,
    SAMPLE_T_VAL,
    SAMPLE_TARGET,
    SAMPLE_VISIT_TARGET,
    SAMPLE_VOCAB_SIZE,
    SAMPLE_WEIGHT,
    collate_xrsh_samples,
    move_batch_to_device,
)
from nn.model import PolicyResNet


def test_collate_xrsh_samples_policy_only():
    batch = [
        {
            SAMPLE_BOARD90: torch.zeros(90, dtype=torch.uint8),
            SAMPLE_STM: torch.tensor(1, dtype=torch.uint8),
            SAMPLE_LEGAL_IDX: torch.tensor([0, 2], dtype=torch.long),
            SAMPLE_VOCAB_SIZE: torch.tensor(4, dtype=torch.long),
            SAMPLE_TARGET: torch.tensor(0),
            SAMPLE_WEIGHT: torch.tensor(1.0),
        },
        {
            SAMPLE_BOARD90: torch.ones(90, dtype=torch.uint8),
            SAMPLE_STM: torch.tensor(0, dtype=torch.uint8),
            SAMPLE_LEGAL_IDX: torch.tensor([1, 3], dtype=torch.long),
            SAMPLE_VOCAB_SIZE: torch.tensor(4, dtype=torch.long),
            SAMPLE_TARGET: torch.tensor(1),
            SAMPLE_WEIGHT: torch.tensor(2.0),
        },
    ]
    out = collate_xrsh_samples(batch)
    assert out[SAMPLE_BOARD].shape == (2, 15, 10, 9)
    assert out[SAMPLE_MASK].shape == (2, 4)
    assert out[SAMPLE_TARGET].tolist() == [0, 1]
    assert out[SAMPLE_WEIGHT].tolist() == [1.0, 2.0]


def test_collate_xrsh_samples_value_and_search():
    batch = [
        {
            SAMPLE_BOARD90: torch.zeros(90, dtype=torch.uint8),
            SAMPLE_STM: torch.tensor(1, dtype=torch.uint8),
            SAMPLE_LEGAL_IDX: torch.tensor([0, 1], dtype=torch.long),
            SAMPLE_SEARCH_COUNTS: torch.tensor([1, 3], dtype=torch.long),
            SAMPLE_VOCAB_SIZE: torch.tensor(4, dtype=torch.long),
            SAMPLE_TARGET: torch.tensor(0),
            SAMPLE_WEIGHT: torch.tensor(1.0),
            SAMPLE_T_VAL: torch.tensor(0.5),
            SAMPLE_SEARCH_VISITS: torch.tensor(8),
        }
    ]
    out = collate_xrsh_samples(batch)
    assert set(out.keys()) == {
        SAMPLE_BOARD,
        SAMPLE_MASK,
        SAMPLE_TARGET,
        SAMPLE_WEIGHT,
        SAMPLE_T_VAL,
        SAMPLE_VISIT_TARGET,
        SAMPLE_SEARCH_VISITS,
    }


def test_move_batch_to_device():
    batch = collate_xrsh_samples(
        [
            {
                SAMPLE_BOARD90: torch.zeros(90, dtype=torch.uint8),
                SAMPLE_STM: torch.tensor(1, dtype=torch.uint8),
                SAMPLE_LEGAL_IDX: torch.tensor([0], dtype=torch.long),
                SAMPLE_VOCAB_SIZE: torch.tensor(4, dtype=torch.long),
                SAMPLE_TARGET: torch.tensor(0),
                SAMPLE_WEIGHT: torch.tensor(1.0),
            }
        ]
    )
    moved = move_batch_to_device(batch, torch.device("cpu"))
    assert moved[SAMPLE_BOARD].device.type == "cpu"


def test_set_requires_grad_can_freeze_value_head_only():
    model = PolicyResNet(
        width=32,
        num_blocks=2,
        num_moves=16,
        value_head=True,
        value_head_hidden_dim=16,
    )
    set_requires_grad(model.fc_value, False)
    assert all(not p.requires_grad for p in model.fc_value.parameters())
    assert all(p.requires_grad for p in model.fc.parameters())


def test_compute_policy_value_loss_includes_value_term() -> None:
    logits = torch.zeros(2, 4)
    pred_value = torch.tensor([0.1, -0.2])
    targets = torch.zeros(2, dtype=torch.long)
    masks = torch.ones(2, 4, dtype=torch.bool)
    weights = torch.ones(2)
    t_val = torch.tensor([0.5, -0.3])
    search_visits = torch.tensor([8, 0], dtype=torch.long)

    policy_only = compute_policy_value_loss(
        logits,
        pred_value,
        targets=targets,
        masks=masks,
        weights=weights,
        value_head=False,
        search_policy_head=False,
    )
    with_value = compute_policy_value_loss(
        logits,
        pred_value,
        targets=targets,
        masks=masks,
        weights=weights,
        value_head=True,
        search_policy_head=False,
        t_val=t_val,
        search_visits=search_visits,
        value_loss_weight=1.0,
        value_min_visits=1,
    )
    assert with_value.item() > policy_only.item()


def test_train_policy_cli_requires_train_source(tmp_path: Path) -> None:
    import subprocess

    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"moves": ["m0"]}', encoding="utf-8")
    val = tmp_path / "val"
    val.mkdir()
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train" / "train_policy.py"),
            "--val-dir",
            str(val),
            "--vocab",
            str(vocab),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    combined = (r.stderr + r.stdout).lower()
    assert "train-dir" in combined or "train-mix" in combined
    assert "nameerror" not in combined
