from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np

from nn.px0_kaggle import (
    PreparedPx0Version,
    ensure_px0_version,
    ensure_stratified_validation_manifest,
    kaggle_dataset_handle,
    load_prepared_px0_training_data,
    px0_version_dir,
    prepare_px0_training_data,
    write_px0_manifest,
)
from nn.px0_record import PX0_COLS, PX0_PLANES, PX0_ROWS, Px0Sample


def _sample_with_pieces(pieces: int) -> Px0Sample:
    planes = np.zeros((PX0_PLANES, PX0_ROWS, PX0_COLS), dtype=np.float32)
    planes[:14].flat[:pieces] = 1.0
    return Px0Sample(
        planes=planes,
        policy=np.zeros(2062, dtype=np.float32),
        winner_q=np.zeros(1, dtype=np.float32),
        winner_wdl=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        root_wdl=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        search_q=np.zeros(1, dtype=np.float32),
        search_wdl=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        search_visits=np.ones(1, dtype=np.float32),
        policy_kld=np.zeros(1, dtype=np.float32),
        plies_left=np.ones(1, dtype=np.float32),
    )


def _write_fake_archive(version_dir: Path) -> None:
    version_dir.mkdir(parents=True, exist_ok=True)
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as tf:
        for name in ("./run1/training.1.gz", "./run1/training.2.gz"):
            payload = name.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    tar_bytes.seek(0)

    archive_path = version_dir / "archive.zip"
    with zipfile.ZipFile(archive_path, mode="w") as zf:
        zf.writestr("data.bin", tar_bytes.read())


def test_kaggle_dataset_handle_supports_versioned_handle() -> None:
    assert kaggle_dataset_handle("latest") == "pikacat/px0data"
    assert kaggle_dataset_handle("7") == "pikacat/px0data/versions/7"


def test_ensure_px0_version_uses_existing_chunks_without_download(tmp_path: Path) -> None:
    version_dir = px0_version_dir("3", root=tmp_path)
    run_dir = version_dir / "run1"
    run_dir.mkdir(parents=True)
    chunk0 = run_dir / "training.123.gz"
    chunk1 = run_dir / "training.124.gz"
    chunk0.write_bytes(b"stub")
    chunk1.write_bytes(b"stub")

    prepared = ensure_px0_version("3", root=tmp_path, val_ratio=0.5, seed=7)
    assert isinstance(prepared, PreparedPx0Version)
    assert prepared.version_dir == version_dir
    assert prepared.chunk_files == [chunk0.resolve(), chunk1.resolve()]
    train_meta = json.loads(prepared.train_manifest.read_text(encoding="utf-8"))
    val_meta = json.loads(prepared.val_manifest.read_text(encoding="utf-8"))
    assert train_meta["version"] == "3"
    assert val_meta["version"] == "3"


def test_ensure_px0_version_extracts_archive_into_chunks(tmp_path: Path, monkeypatch) -> None:
    version_dir = px0_version_dir("12", root=tmp_path)
    _write_fake_archive(version_dir)

    called: list[tuple[str, Path, bool]] = []

    def _fake_download(version: str, target: Path, *, force_download: bool) -> None:
        called.append((version, target, force_download))

    monkeypatch.setattr("nn.px0_kaggle.download_px0_from_kaggle", _fake_download)
    prepared = ensure_px0_version("12", root=tmp_path, val_ratio=0.5, seed=11)

    assert called == []
    assert [p.name for p in prepared.chunk_files] == ["training.1.gz", "training.2.gz"]
    assert all(p.is_file() for p in prepared.chunk_files)


def test_ensure_px0_version_force_download_rebuilds_tree(tmp_path: Path, monkeypatch) -> None:
    version_dir = px0_version_dir("21", root=tmp_path)
    run_dir = version_dir / "run1"
    run_dir.mkdir(parents=True)
    stale_chunk = run_dir / "training.stale.gz"
    stale_chunk.write_bytes(b"stale")

    calls: list[tuple[str, Path, bool]] = []

    def _fake_download(version: str, target: Path, *, force_download: bool) -> None:
        calls.append((version, target, force_download))
        _write_fake_archive(target)

    monkeypatch.setattr("nn.px0_kaggle.download_px0_from_kaggle", _fake_download)
    prepared = ensure_px0_version("21", root=tmp_path, val_ratio=0.5, seed=5, force_download=True)

    assert calls == [("21", version_dir, True)]
    assert not stale_chunk.exists()
    assert [p.name for p in prepared.chunk_files] == ["training.1.gz", "training.2.gz"]


def test_ensure_px0_version_reuses_matching_manifests_without_rescan(tmp_path: Path, monkeypatch) -> None:
    version_dir = px0_version_dir("30", root=tmp_path)
    run_dir = version_dir / "run1"
    run_dir.mkdir(parents=True)
    chunk0 = (run_dir / "training.1.gz").resolve()
    chunk1 = (run_dir / "training.2.gz").resolve()
    chunk0.write_bytes(b"stub")
    chunk1.write_bytes(b"stub")
    manifest_dir = version_dir / "manifests"
    write_px0_manifest(manifest_dir / "train.json", files=[chunk0], version="30", seed=9, val_ratio=0.5)
    write_px0_manifest(manifest_dir / "val.json", files=[chunk1], version="30", seed=9, val_ratio=0.5)

    def _unexpected_discover(_version_dir):
        raise AssertionError("discover_px0_chunks should not run when manifests already match")

    monkeypatch.setattr("nn.px0_kaggle.discover_px0_chunks", _unexpected_discover)
    prepared = ensure_px0_version("30", root=tmp_path, val_ratio=0.5, seed=9)
    assert prepared.train_manifest == manifest_dir / "train.json"
    assert prepared.val_manifest == manifest_dir / "val.json"
    assert prepared.chunk_files == [chunk0, chunk1]


def test_stratified_validation_manifest_is_fixed_and_balanced(tmp_path: Path, monkeypatch) -> None:
    version_dir = px0_version_dir("31", root=tmp_path)
    chunk_paths = []
    for index in range(3):
        path = version_dir / f"training.{index}.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")
        chunk_paths.append(path.resolve())
    manifest_dir = version_dir / "manifests"
    write_px0_manifest(manifest_dir / "train.json", files=[chunk_paths[0]], version="31", seed=3, val_ratio=0.5)
    write_px0_manifest(manifest_dir / "val.json", files=chunk_paths[1:], version="31", seed=3, val_ratio=0.5)
    prepared = PreparedPx0Version(
        version="31",
        version_dir=version_dir,
        chunk_files=chunk_paths,
        train_manifest=manifest_dir / "train.json",
        val_manifest=manifest_dir / "val.json",
    )

    def _samples(_path: Path):
        return iter([_sample_with_pieces(30), _sample_with_pieces(20), _sample_with_pieces(10)] * 3)

    monkeypatch.setattr("nn.px0_kaggle.iter_px0_chunk_file", _samples)
    path = ensure_stratified_validation_manifest(prepared, samples=6, source_files=2, seed=3)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stage_counts"] == {"opening": 2, "middlegame": 2, "endgame": 2}
    assert len(payload["samples"]) == 6
    assert ensure_stratified_validation_manifest(prepared, samples=6, source_files=2, seed=3) == path


def test_load_prepared_training_data_does_not_rescan_chunks(tmp_path: Path, monkeypatch) -> None:
    version_dir = px0_version_dir("32", root=tmp_path)
    chunk_paths = []
    for index in range(3):
        path = version_dir / f"training.{index}.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")
        chunk_paths.append(path.resolve())
    manifest_dir = version_dir / "manifests"
    write_px0_manifest(manifest_dir / "train.json", files=[chunk_paths[0]], version="32", seed=3, val_ratio=0.5)
    write_px0_manifest(manifest_dir / "val.json", files=chunk_paths[1:], version="32", seed=3, val_ratio=0.5)
    prepared = PreparedPx0Version(
        version="32",
        version_dir=version_dir,
        chunk_files=chunk_paths,
        train_manifest=manifest_dir / "train.json",
        val_manifest=manifest_dir / "val.json",
    )

    def _samples(_path: Path):
        return iter([_sample_with_pieces(30), _sample_with_pieces(20), _sample_with_pieces(10)] * 3)

    monkeypatch.setattr("nn.px0_kaggle.iter_px0_chunk_file", _samples)
    _prepared, validation_manifest = prepare_px0_training_data(
        "32",
        root=tmp_path,
        val_ratio=0.5,
        seed=3,
        force_download=False,
        validation_samples=6,
        validation_source_files=2,
    )

    def _unexpected_scan(_version_dir):
        raise AssertionError("training startup must not scan chunk files")

    monkeypatch.setattr("nn.px0_kaggle.discover_px0_chunks", _unexpected_scan)
    loaded, loaded_manifest = load_prepared_px0_training_data(
        "32",
        root=tmp_path,
        val_ratio=0.5,
        seed=3,
        validation_samples=6,
        validation_source_files=2,
    )
    assert loaded.version == prepared.version
    assert loaded.train_manifest == prepared.train_manifest
    assert loaded.val_manifest == prepared.val_manifest
    assert loaded.chunk_files == []
    assert loaded_manifest == validation_manifest


def test_prepare_px0_training_data_builds_all_training_manifests(tmp_path: Path, monkeypatch) -> None:
    version_dir = px0_version_dir("33", root=tmp_path)
    run_dir = version_dir / "run"
    run_dir.mkdir(parents=True)
    for index in range(3):
        (run_dir / f"training.{index}.gz").write_bytes(b"stub")

    def _samples(_path: Path):
        return iter([_sample_with_pieces(30), _sample_with_pieces(20), _sample_with_pieces(10)] * 3)

    monkeypatch.setattr("nn.px0_kaggle.iter_px0_chunk_file", _samples)
    prepared, validation_manifest = prepare_px0_training_data(
        "33",
        root=tmp_path,
        val_ratio=0.34,
        seed=3,
        force_download=False,
        validation_samples=6,
        validation_source_files=0,
    )
    assert prepared.train_manifest.is_file()
    assert prepared.val_manifest.is_file()
    assert validation_manifest.is_file()
