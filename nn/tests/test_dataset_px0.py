from __future__ import annotations

import json
import random
from pathlib import Path

from nn.dataset_px0 import Px0DatasetConfig, load_px0_file_list, resolve_px0_files, shuffle_stream


def test_load_px0_file_list(tmp_path: Path) -> None:
    f0 = tmp_path / "a.gz"
    f1 = tmp_path / "b.gz"
    f0.write_bytes(b"x")
    f1.write_bytes(b"y")
    manifest = tmp_path / "files.json"
    manifest.write_text(
        json.dumps({"files": [str(f0), str(f1)]}),
        encoding="utf-8",
    )
    files = load_px0_file_list(manifest)
    assert files == [f0.resolve(), f1.resolve()]


def test_load_px0_file_list_can_skip_per_file_checks_for_prepared_training(tmp_path: Path) -> None:
    missing = tmp_path / "not-checked.gz"
    manifest = tmp_path / "files.json"
    manifest.write_text(json.dumps({"files": [str(missing)]}), encoding="utf-8")

    assert load_px0_file_list(manifest, verify_files=False) == [missing]


def test_resolve_px0_files_honors_max_files(tmp_path: Path) -> None:
    paths = []
    for idx in range(3):
        p = tmp_path / f"{idx}.gz"
        p.write_bytes(b"x")
        paths.append(p.resolve())
    cfg = Px0DatasetConfig(patterns=(str(tmp_path / "*.gz"),), max_files=2)
    files = resolve_px0_files(cfg)
    assert files == paths[:2]


def test_shuffle_stream_preserves_each_item_once() -> None:
    values = list(range(32))
    random.seed(7)
    shuffled = list(shuffle_stream(iter(values), size=8))
    assert sorted(shuffled) == values
    assert shuffled != values
