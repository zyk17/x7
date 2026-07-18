"""Kaggle px0data 本地准备逻辑。"""

from __future__ import annotations

import json
import random
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from nn.px0_record import Px0Sample, expand_chunk_globs, iter_px0_chunk_file

DEFAULT_PX0_ROOT = Path(r"C:\work\px0data")
DATASET_HANDLE = "pikacat/px0data"


@dataclass(frozen=True)
class PreparedPx0Version:
    version: str
    version_dir: Path
    chunk_files: list[Path]
    train_manifest: Path
    val_manifest: Path


VALIDATION_STAGES = ("opening", "middlegame", "endgame")


def read_px0_manifest(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _manifest_matches(payload: dict, *, version: str, seed: int, val_ratio: float) -> bool:
    return (
        str(payload.get("format")) == "px0_file_list_v2"
        and str(payload.get("dataset")) == "pikacat/px0data"
        and str(payload.get("version")) == str(version)
        and int(payload.get("seed", -1)) == int(seed)
        and abs(float(payload.get("val_ratio", -1.0)) - float(val_ratio)) <= 1e-12
    )


def try_load_prepared_px0_version(
    version: str,
    *,
    root: Path | str = DEFAULT_PX0_ROOT,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> PreparedPx0Version | None:
    version_dir = px0_version_dir(version, root=root)
    manifest_dir = version_dir / "manifests"
    train_manifest = manifest_dir / "train.json"
    val_manifest = manifest_dir / "val.json"
    if not train_manifest.is_file() or not val_manifest.is_file():
        return None

    train_payload = read_px0_manifest(train_manifest)
    val_payload = read_px0_manifest(val_manifest)
    if not _manifest_matches(train_payload, version=version, seed=seed, val_ratio=val_ratio):
        return None
    if not _manifest_matches(val_payload, version=version, seed=seed, val_ratio=val_ratio):
        return None

    train_files = [Path(str(item)).resolve() for item in train_payload.get("files", [])]
    val_files = [Path(str(item)).resolve() for item in val_payload.get("files", [])]
    chunk_files = sorted({*train_files, *val_files})
    if not train_files or not val_files or any(not path.is_file() for path in chunk_files):
        return None

    return PreparedPx0Version(
        version=str(version),
        version_dir=version_dir,
        chunk_files=chunk_files,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
    )


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


def _validation_stage(sample: Px0Sample) -> str:
    """Bucket a position by current material, not by unavailable game ply."""
    pieces = int(sample.planes[:14].sum())
    if pieces >= 28:
        return "opening"
    if pieces >= 16:
        return "middlegame"
    return "endgame"


def _validation_manifest_path(prepared: PreparedPx0Version, *, samples: int, source_files: int) -> Path:
    return prepared.version_dir / "manifests" / f"validation_{samples}_{source_files}.json"


def _training_ready_path(prepared: PreparedPx0Version, *, samples: int, source_files: int) -> Path:
    return prepared.version_dir / "manifests" / f"training_ready_{samples}_{source_files}.json"


def _validation_manifest_matches(
    payload: dict,
    *,
    prepared: PreparedPx0Version,
    samples: int,
    source_files: int,
    seed: int,
) -> bool:
    entries = payload.get("samples")
    return (
        payload.get("format") == "px0_stratified_validation_v1"
        and str(payload.get("version")) == prepared.version
        and int(payload.get("seed", -1)) == seed
        and int(payload.get("sample_count", -1)) == samples
        and int(payload.get("source_file_limit", -1)) == source_files
        and isinstance(entries, list)
        and len(entries) == samples
    )


def ensure_stratified_validation_manifest(
    prepared: PreparedPx0Version,
    *,
    samples: int,
    source_files: int,
    seed: int,
) -> Path:
    """Build a fixed, material-stratified validation sample manifest.

    Scanning every file in a 10% px0 split is impractical. We therefore select
    a deterministic random subset of validation chunks, reservoir-sample each
    material bucket, and persist the exact record references. Every later
    validation uses the same positions.
    """
    if samples < len(VALIDATION_STAGES):
        raise ValueError("validation_samples 必须至少为 3")
    if source_files < 0:
        raise ValueError("validation_source_files 须为非负整数")

    path = _validation_manifest_path(prepared, samples=samples, source_files=source_files)
    existing = try_load_stratified_validation_manifest(
        prepared,
        samples=samples,
        source_files=source_files,
        seed=seed,
    )
    if existing is not None:
        return existing

    val_payload = read_px0_manifest(prepared.val_manifest)
    val_files = [Path(str(item)).resolve() for item in val_payload["files"]]
    rng = random.Random(seed)
    rng.shuffle(val_files)
    selected_files = val_files if source_files == 0 else val_files[:source_files]
    quotas = {
        stage: samples // len(VALIDATION_STAGES) + (index < samples % len(VALIDATION_STAGES))
        for index, stage in enumerate(VALIDATION_STAGES)
    }
    reservoirs: dict[str, list[dict[str, int | str]]] = {stage: [] for stage in VALIDATION_STAGES}
    seen = {stage: 0 for stage in VALIDATION_STAGES}

    for chunk_path in selected_files:
        for record_index, sample in enumerate(iter_px0_chunk_file(chunk_path)):
            stage = _validation_stage(sample)
            seen[stage] += 1
            entry: dict[str, int | str] = {"file": str(chunk_path), "record_index": record_index, "stage": stage}
            reservoir = reservoirs[stage]
            if len(reservoir) < quotas[stage]:
                reservoir.append(entry)
                continue
            replace_at = rng.randrange(seen[stage])
            if replace_at < quotas[stage]:
                reservoir[replace_at] = entry
        if all(len(reservoirs[stage]) >= quotas[stage] for stage in VALIDATION_STAGES):
            break

    missing = [stage for stage in VALIDATION_STAGES if len(reservoirs[stage]) < quotas[stage]]
    if missing:
        raise ValueError(
            "固定验证样本无法覆盖分层: "
            + ", ".join(f"{stage}={len(reservoirs[stage])}/{quotas[stage]}" for stage in missing)
            + "; 增大 validation_source_files，或设为 0 自动扫描"
        )
    entries = [entry for stage in VALIDATION_STAGES for entry in reservoirs[stage]]
    entries.sort(key=lambda item: (str(item["file"]), int(item["record_index"])))
    payload = {
        "format": "px0_stratified_validation_v1",
        "version": prepared.version,
        "seed": seed,
        "sample_count": samples,
        "source_file_limit": source_files,
        "source_file_count": len({Path(str(entry["file"])) for entry in entries}),
        "source_files": sorted({str(entry["file"]) for entry in entries}),
        "stage_definition": {"opening": ">=28 pieces", "middlegame": "16-27 pieces", "endgame": "<=15 pieces"},
        "stage_counts": {stage: len(reservoirs[stage]) for stage in VALIDATION_STAGES},
        "samples": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def try_load_stratified_validation_manifest(
    prepared: PreparedPx0Version,
    *,
    samples: int,
    source_files: int,
    seed: int,
) -> Path | None:
    """Return an already prepared validation manifest without scanning chunks."""
    path = _validation_manifest_path(prepared, samples=samples, source_files=source_files)
    if not path.is_file():
        return None
    payload = read_px0_manifest(path)
    if not _validation_manifest_matches(
        payload,
        prepared=prepared,
        samples=samples,
        source_files=source_files,
        seed=seed,
    ):
        return None
    return path


def prepare_px0_training_data(
    version: str,
    *,
    root: Path | str,
    val_ratio: float,
    seed: int,
    force_download: bool,
    validation_samples: int,
    validation_source_files: int,
) -> tuple[PreparedPx0Version, Path]:
    """Perform the slow, one-time download, split, and validation preparation."""
    prepared = ensure_px0_version(
        version,
        root=root,
        val_ratio=val_ratio,
        seed=seed,
        force_download=force_download,
    )
    validation_manifest = ensure_stratified_validation_manifest(
        prepared,
        samples=validation_samples,
        source_files=validation_source_files,
        seed=seed,
    )
    ready_path = _training_ready_path(
        prepared,
        samples=validation_samples,
        source_files=validation_source_files,
    )
    ready_path.write_text(
        json.dumps(
            {
                "format": "px0_training_ready_v1",
                "version": prepared.version,
                "seed": seed,
                "val_ratio": val_ratio,
                "validation_samples": validation_samples,
                "validation_source_files": validation_source_files,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return prepared, validation_manifest


def load_prepared_px0_training_data(
    version: str,
    *,
    root: Path | str,
    val_ratio: float,
    seed: int,
    validation_samples: int,
    validation_source_files: int,
) -> tuple[PreparedPx0Version, Path]:
    """Load only prepared manifests; training must never download or rescan data."""
    version_dir = px0_version_dir(version, root=root)
    manifest_dir = version_dir / "manifests"
    train_manifest = manifest_dir / "train.json"
    val_manifest = manifest_dir / "val.json"
    prepared = PreparedPx0Version(
        version=str(version),
        version_dir=version_dir,
        chunk_files=[],
        train_manifest=train_manifest,
        val_manifest=val_manifest,
    )
    validation_manifest = _validation_manifest_path(
        prepared,
        samples=validation_samples,
        source_files=validation_source_files,
    )
    ready_path = _training_ready_path(
        prepared,
        samples=validation_samples,
        source_files=validation_source_files,
    )
    required_paths = (train_manifest, val_manifest, validation_manifest, ready_path)
    if any(not path.is_file() for path in required_paths):
        raise FileNotFoundError(
            f"PX0 {version} 尚未准备完成；先运行 scripts/data/prepare_px0.py --config <YAML>"
        )
    ready = read_px0_manifest(ready_path)
    if (
        ready.get("format") != "px0_training_ready_v1"
        or str(ready.get("version")) != str(version)
        or int(ready.get("seed", -1)) != int(seed)
        or abs(float(ready.get("val_ratio", -1.0)) - float(val_ratio)) > 1e-12
        or int(ready.get("validation_samples", -1)) != int(validation_samples)
        or int(ready.get("validation_source_files", -1)) != int(validation_source_files)
    ):
        raise FileNotFoundError(
            f"PX0 {version} 的准备参数与 YAML 不一致；先运行 scripts/data/prepare_px0.py --config <YAML>"
        )
    return prepared, validation_manifest


def split_px0_train_val(
    files: list[Path],
    *,
    seed: int,
    val_ratio: float,
) -> tuple[list[Path], list[Path]]:
    if len(files) < 2:
        raise ValueError("需要至少 2 个 chunk 文件才能切 train/val")
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
    if not force_download:
        prepared = try_load_prepared_px0_version(
            version,
            root=root,
            val_ratio=val_ratio,
            seed=seed,
        )
        if prepared is not None:
            return prepared
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
