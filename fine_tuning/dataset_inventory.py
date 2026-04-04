from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .constants import (
    CURRENT_DATASETS_DIR,
    FINAL_DATASETS_DIR,
    FOLD_DATASET_IDS,
    PREV_DATASETS_DIR,
)

DatasetCollection = Literal["prev", "current", "fold_a", "fold_b", "both"]
DatasetSource = Literal["prev", "current"]


@dataclass(frozen=True)
class DatasetAsset:
    dataset_id: int
    source: DatasetSource
    dataset_path: Path
    bots_path: Path


@dataclass(frozen=True)
class FinalDatasetAsset:
    dataset_id: int
    dataset_path: Path


def _dataset_id_from_path(path: Path) -> int:
    suffix = path.stem.split(".")[-1]
    return int(suffix)


def _scan_source_directory(
    *,
    source: DatasetSource,
    directory: Path,
) -> tuple[DatasetAsset, ...]:
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    assets: list[DatasetAsset] = []
    for dataset_path in sorted(directory.glob("dataset.posts&users.*.json")):
        dataset_id = _dataset_id_from_path(dataset_path)
        bots_path = directory / f"dataset.bots.{dataset_id}.txt"
        if not bots_path.exists():
            raise FileNotFoundError(
                f"Missing bot label file for dataset {dataset_id}: {bots_path}"
            )

        assets.append(
            DatasetAsset(
                dataset_id=dataset_id,
                source=source,
                dataset_path=dataset_path,
                bots_path=bots_path,
            )
        )

    if not assets:
        raise ValueError(f"No labeled datasets found under {directory}")
    return tuple(assets)


def discover_dataset_assets() -> tuple[DatasetAsset, ...]:
    prev_assets = _scan_source_directory(source="prev", directory=PREV_DATASETS_DIR)
    current_assets = _scan_source_directory(
        source="current",
        directory=CURRENT_DATASETS_DIR,
    )
    return prev_assets + current_assets


def discover_final_dataset_assets() -> tuple[FinalDatasetAsset, ...]:
    if not FINAL_DATASETS_DIR.exists():
        raise FileNotFoundError(f"Final dataset directory not found: {FINAL_DATASETS_DIR}")

    assets: list[FinalDatasetAsset] = []
    for dataset_path in sorted(FINAL_DATASETS_DIR.glob("dataset.posts&users.*.json")):
        assets.append(
            FinalDatasetAsset(
                dataset_id=_dataset_id_from_path(dataset_path),
                dataset_path=dataset_path,
            )
        )

    if not assets:
        raise ValueError(f"No final datasets found under {FINAL_DATASETS_DIR}")
    return tuple(assets)


def group_assets_by_collection(
    assets: tuple[DatasetAsset, ...],
) -> dict[DatasetCollection, tuple[DatasetAsset, ...]]:
    prev_assets = tuple(asset for asset in assets if asset.source == "prev")
    current_assets = tuple(asset for asset in assets if asset.source == "current")
    assets_by_id = {asset.dataset_id: asset for asset in assets}

    fold_assets: dict[str, tuple[DatasetAsset, ...]] = {}
    for fold_name, dataset_ids in FOLD_DATASET_IDS.items():
        missing_ids = [dataset_id for dataset_id in dataset_ids if dataset_id not in assets_by_id]
        if missing_ids:
            raise ValueError(
                f"Configured {fold_name} datasets are missing from disk: {missing_ids}"
            )
        fold_assets[fold_name] = tuple(assets_by_id[dataset_id] for dataset_id in dataset_ids)

    return {
        "prev": prev_assets,
        "current": current_assets,
        "fold_a": fold_assets["fold_a"],
        "fold_b": fold_assets["fold_b"],
        "both": prev_assets + current_assets,
    }
