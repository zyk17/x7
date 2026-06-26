"""Kaggle px0data 本地准备逻辑。"""

from __future__ import annotations

import json
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from nn.px0_record import expand_chunk_globs

DEFAULT_PX0_ROOT = Path(r"C:\work\px0data")
DATASET_HANDLE = "pikacat/px0data"


@dataclass(frozen=True)
class PreparedPx0Version:
    version: str
    version_dir: Path
    chunk_files: list[Path]
    train_manifest: Path
    val_manifest: Path


def kaggle_dataset_handle(version: str) -> str:
    version = str(version).strip()
    if not version or version.lower() == "latest":
        return DATASET_HANDLE
    return f"{DATASET_HANDLE}/versions/{version}"


def px0_version_dir(version: str, *, root: Path | str = DEFAULT_PX0_ROOT) -> Path:
    return Path(root).resolve() / str(version).strip()


def discover_px0_chunks(version_dir: Path | str) -> list[Path]:
    return expand_chunk_globs([str(Path(version_dir) / "**" / "training.*.gz")], max_files=0)


def has_px0_download_artifacts(version_dir: Path | str) -> bool:
    version_dir = Path(version_dir)
    return any(version_dir.rglob("*.zip")) or any(version_dir.rglob("data.bin"))


def clear_prepared_px0_version(version_dir: Path | str) -> None:
    version_dir = Path(version_dir)
    for pattern in ("training.*.gz", "*.zip", "data.bin"):
        for path in sorted(version_dir.rglob(pattern)):
            if path.is_file():
                path.unlink()
    for name in ("extracted", "unpacked", "manifests"):
        for path in sorted(version_dir.rglob(name)):
            if path.is_dir():
                shutil.rmtree(path)


def write_px0_manifest(path: Path, *, files: list[Path], version: str, seed: int, val_ratio: float) -> None:
    payload = {
        "format": "px0_file_list_v2",
        "dataset": "pikacat/px0data",
        "version": str(version),
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "count": len(files),
        "files": [str(p) for p in files],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_px0_train_val(
    files: list[Path],
    *,
    seed: int,
    val_ratio: float,
) -> tuple[list[Path], list[Path]]:
    if len(files) < 2:
        raise ValueError("需要至少 2 个 chunk 文件才能切 train/val")
    import random

    ordered = list(sorted(files))
    random.Random(int(seed)).shuffle(ordered)
    val_count = max(1, int(round(len(ordered) * float(val_ratio))))
    val_files = sorted(ordered[:val_count])
    train_files = sorted(ordered[val_count:])
    if not train_files:
        raise ValueError("val_ratio 过大，train 为空")
    return train_files, val_files


def ensure_px0_version(
    version: str,
    *,
    root: Path | str = DEFAULT_PX0_ROOT,
    val_ratio: float = 0.1,
    seed: int = 42,
    force_download: bool = False,
) -> PreparedPx0Version:
    version = str(version).strip()
    if not version:
        raise ValueError("px0 version 不能为空")
    version_dir = px0_version_dir(version, root=root)
    version_dir.mkdir(parents=True, exist_ok=True)

    if force_download:
        clear_prepared_px0_version(version_dir)
        download_px0_from_kaggle(version, version_dir, force_download=True)
        prepare_px0_download_tree(version_dir)
        chunk_files = discover_px0_chunks(version_dir)
    else:
        chunk_files = discover_px0_chunks(version_dir)
        if not chunk_files:
            prepare_px0_download_tree(version_dir)
            chunk_files = discover_px0_chunks(version_dir)
        if not chunk_files and not has_px0_download_artifacts(version_dir):
            download_px0_from_kaggle(version, version_dir, force_download=False)
            prepare_px0_download_tree(version_dir)
            chunk_files = discover_px0_chunks(version_dir)
    if not chunk_files:
        raise FileNotFoundError(f"未在 {version_dir} 下找到 training.*.gz")

    manifest_dir = version_dir / "manifests"
    train_manifest = manifest_dir / "train.json"
    val_manifest = manifest_dir / "val.json"
    train_files, val_files = split_px0_train_val(chunk_files, seed=seed, val_ratio=val_ratio)
    write_px0_manifest(train_manifest, files=train_files, version=version, seed=seed, val_ratio=val_ratio)
    write_px0_manifest(val_manifest, files=val_files, version=version, seed=seed, val_ratio=val_ratio)

    return PreparedPx0Version(
        version=version,
        version_dir=version_dir,
        chunk_files=chunk_files,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
    )


def download_px0_from_kaggle(version: str, version_dir: Path, *, force_download: bool) -> None:
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "缺少 kagglehub；请先执行 "
            r"`C:\projects\77xiangqi_engine\nn\.venv\Scripts\python.exe -m pip install kagglehub`"
        ) from exc

    kagglehub.dataset_download(
        kaggle_dataset_handle(version),
        force_download=bool(force_download),
        output_dir=str(version_dir),
    )


def prepare_px0_download_tree(version_dir: Path | str) -> None:
    version_dir = Path(version_dir)
    if discover_px0_chunks(version_dir):
        return

    for archive_path in sorted(version_dir.rglob("*.zip")):
        extract_dir = archive_path.parent / "extracted"
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(extract_dir)

    if discover_px0_chunks(version_dir):
        return

    data_bins = sorted(version_dir.rglob("data.bin"))
    for data_bin in data_bins:
        if data_bin.parent.name == "extracted":
            unpack_dir = data_bin.parent.parent / "unpacked"
        else:
            unpack_dir = data_bin.parent / "unpacked"
        if discover_px0_chunks(unpack_dir):
            continue
        unpack_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(data_bin, mode="r:*") as tf:
            tf.extractall(unpack_dir)
