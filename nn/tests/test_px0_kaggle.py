from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

from nn.px0_kaggle import (
    PreparedPx0Version,
    ensure_px0_version,
    kaggle_dataset_handle,
    px0_version_dir,
    write_px0_manifest,
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
