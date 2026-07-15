from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from train_px0 import resolve_device, validate_existing_output_checkpoint


def test_validate_existing_output_checkpoint_accepts_current_architecture() -> None:
    validate_existing_output_checkpoint(
        {
            "trunk_kind": "katago_cnn_v1",
            "moves_left_head": True,
        }
    )


def test_resolve_device_accepts_explicit_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"
