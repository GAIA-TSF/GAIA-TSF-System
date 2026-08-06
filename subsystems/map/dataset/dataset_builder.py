"""Conversion of aligned feature stacks into reproducible temporal ML datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from subsystems.map.dataset.feature_loader import LoadedFeatures, RasterGrid


@dataclass(frozen=True)
class Dataset:
    """Flat ML samples plus enough index information to restore rasters."""

    features: np.ndarray
    targets: np.ndarray
    time_indices: np.ndarray
    pixel_indices: np.ndarray
    feature_names: tuple[str, ...]
    dates: tuple[str, ...]
    grid: RasterGrid
    mask: np.ndarray


@dataclass(frozen=True)
class DatasetSplits:
    """Chronological train, validation and test subsets."""

    train: Dataset
    validation: Dataset
    test: Dataset


class DatasetBuilder:
    """Build temporal samples from DAG features; feature engineering stays in DAG."""

    def build(
        self,
        loaded: LoadedFeatures,
        feature_names: list[str],
        target_feature: str,
        pixel_mask: np.ndarray | None = None,
    ) -> Dataset:
        """Build samples for all valid feature/target observations in ``pixel_mask``."""
        if target_feature not in loaded.features:
            raise KeyError(f'Target feature was not loaded: {target_feature}')
        if any(name not in loaded.features for name in feature_names):
            raise KeyError('One or more configured feature names were not loaded.')
        selected_mask = loaded.mask if pixel_mask is None else loaded.mask & pixel_mask
        target = loaded.features[target_feature]
        stacks = [loaded.features[name] for name in feature_names]
        valid = selected_mask[np.newaxis, :, :] & np.isfinite(target)
        for stack in stacks:
            valid &= np.isfinite(stack)
        time_indices, rows, columns = np.where(valid)
        pixel_indices = rows * loaded.grid.width + columns
        feature_matrix = np.column_stack([stack[valid] for stack in stacks])
        targets = target[valid]
        if targets.size == 0:
            raise ValueError(
                'No finite samples remain after feature and mask filtering.'
            )
        return Dataset(
            feature_matrix,
            targets,
            time_indices,
            pixel_indices,
            tuple(feature_names),
            loaded.dates,
            loaded.grid,
            selected_mask,
        )

    def split_temporal(
        self,
        dataset: Dataset,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
    ) -> DatasetSplits:
        """Split by acquisition time, preventing future observations entering training."""
        if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
            raise ValueError('Temporal split ratios must sum to 1.0.')
        time_count = len(dataset.dates)
        train_end = int(time_count * train_ratio)
        validation_end = train_end + int(time_count * validation_ratio)
        if train_end < 1 or validation_end <= train_end or validation_end >= time_count:
            raise ValueError(
                'Temporal split needs at least one acquisition per subset.'
            )
        return DatasetSplits(
            self._subset(dataset, dataset.time_indices < train_end),
            self._subset(
                dataset,
                (dataset.time_indices >= train_end)
                & (dataset.time_indices < validation_end),
            ),
            self._subset(dataset, dataset.time_indices >= validation_end),
        )

    @staticmethod
    def _subset(dataset: Dataset, include: np.ndarray) -> Dataset:
        if not np.any(include):
            raise ValueError('A temporal split contains no valid samples.')
        return Dataset(
            dataset.features[include],
            dataset.targets[include],
            dataset.time_indices[include],
            dataset.pixel_indices[include],
            dataset.feature_names,
            dataset.dates,
            dataset.grid,
            dataset.mask,
        )
