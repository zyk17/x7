from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from train_px0 import validate_existing_output_checkpoint


def _args() -> argparse.Namespace:
    return argparse.Namespace(q_ratio=0.5)


def test_validate_existing_output_checkpoint_accepts_matching_state(tmp_path: Path) -> None:
    train_file = (tmp_path / "train.gz").resolve()
    val_file = (tmp_path / "val.gz").resolve()
    ckpt = {
        "q_ratio": 0.5,
        "train_files": [str(train_file)],
        "val_files": [str(val_file)],
    }

    validate_existing_output_checkpoint(
        ckpt,
        args=_args(),
        train_files=[train_file],
        val_files=[val_file],
    )


def test_validate_existing_output_checkpoint_rejects_q_ratio_mismatch(tmp_path: Path) -> None:
    train_file = (tmp_path / "train.gz").resolve()
    val_file = (tmp_path / "val.gz").resolve()
    ckpt = {
        "q_ratio": 0.75,
        "train_files": [str(train_file)],
        "val_files": [str(val_file)],
    }

    with pytest.raises(SystemExit, match="q_ratio="):
        validate_existing_output_checkpoint(
            ckpt,
            args=_args(),
            train_files=[train_file],
            val_files=[val_file],
        )


def test_validate_existing_output_checkpoint_rejects_dataset_mismatch(tmp_path: Path) -> None:
    train_file = (tmp_path / "train.gz").resolve()
    other_train_file = (tmp_path / "other_train.gz").resolve()
    val_file = (tmp_path / "val.gz").resolve()
    ckpt = {
        "q_ratio": 0.5,
        "train_files": [str(train_file)],
        "val_files": [str(val_file)],
    }

    with pytest.raises(SystemExit, match="train_files"):
        validate_existing_output_checkpoint(
            ckpt,
            args=_args(),
            train_files=[other_train_file],
            val_files=[val_file],
        )
