#!/usr/bin/env python3
"""Prepare the PX0 download, file split, and fixed validation manifest once."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

NN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NN_ROOT / "src"))

from nn.px0_kaggle import prepare_px0_training_data
from nn.train_config import load_train_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PX0 data for later fast training starts")
    parser.add_argument("--config", type=Path, required=True, help="与训练相同的 YAML 配置")
    cli = parser.parse_args()
    try:
        args = load_train_config(cli.config)
        prepared, validation_manifest = prepare_px0_training_data(
            args.px0_version,
            root=args.px0_root,
            val_ratio=float(args.px0_val_ratio),
            seed=int(args.px0_seed),
            force_download=bool(args.px0_force_download),
            validation_samples=int(args.validation_samples),
            validation_source_files=int(args.validation_source_files),
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"prepared px0 version={prepared.version} chunks={len(prepared.chunk_files)} "
        f"train_manifest={prepared.train_manifest} validation_manifest={validation_manifest}"
    )


if __name__ == "__main__":
    main()
