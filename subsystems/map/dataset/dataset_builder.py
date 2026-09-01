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

    def build_sequences(
        self,
        dataset: Dataset,
        look_back: int,
        horizon: int,
    ) -> Dataset:
        """Build causal, per-pixel sequences for a sequence-model plugin.

        Every sequence contains ``look_back`` consecutive feature vectors. The
        target occurs ``horizon`` acquisitions after the final input vector;
        consequently no target-time observation can enter the model input.

        Args:
            dataset: Flat temporal samples created from DAG feature rasters.
            look_back: Number of prior acquisitions in each input sequence.
            horizon: Number of acquisitions between sequence end and target.

        Returns:
            A dataset whose feature matrix has shape
            ``(samples, look_back, features)``.

        Raises:
            ValueError: If the sequence specification is invalid or produces
                no complete per-pixel samples.
        """
        if look_back < 1 or horizon < 1:
            raise ValueError('Sequence look_back and horizon must be positive.')
        if dataset.features.ndim != 2:
            raise ValueError('Sequence construction requires a tabular dataset.')

        rows_by_time = {
            time_index: np.flatnonzero(dataset.time_indices == time_index)
            for time_index in np.unique(dataset.time_indices)
        }
        sequence_features: list[np.ndarray] = []
        sequence_targets: list[np.ndarray] = []
        sequence_times: list[np.ndarray] = []
        sequence_pixels: list[np.ndarray] = []
        first_target_time = look_back + horizon - 1
        for target_time in range(first_target_time, len(dataset.dates)):
            input_times = range(
                target_time - horizon - look_back + 1,
                target_time - horizon + 1,
            )
            required_times = [*input_times, target_time]
            if any(time not in rows_by_time for time in required_times):
                continue
            common_pixels = dataset.pixel_indices[rows_by_time[required_times[0]]]
            for time_index in required_times[1:]:
                common_pixels = np.intersect1d(
                    common_pixels,
                    dataset.pixel_indices[rows_by_time[time_index]],
                    assume_unique=True,
                )
            if common_pixels.size == 0:
                continue

            feature_steps: list[np.ndarray] = []
            for time_index in input_times:
                rows = rows_by_time[time_index]
                pixel_values = dataset.pixel_indices[rows]
                matches = np.searchsorted(pixel_values, common_pixels)
                feature_steps.append(dataset.features[rows[matches]])
            target_rows = rows_by_time[target_time]
            target_pixels = dataset.pixel_indices[target_rows]
            target_matches = np.searchsorted(target_pixels, common_pixels)
            sequence_features.append(np.stack(feature_steps, axis=1))
            sequence_targets.append(dataset.targets[target_rows[target_matches]])
            sequence_times.append(
                np.full(common_pixels.size, target_time, dtype=np.int64),
            )
            sequence_pixels.append(common_pixels)

        if not sequence_features:
            raise ValueError(
                'No complete causal sequences remain for the configured '
                'look_back and horizon.',
            )
        return Dataset(
            np.concatenate(sequence_features, axis=0),
            np.concatenate(sequence_targets),
            np.concatenate(sequence_times),
            np.concatenate(sequence_pixels),
            dataset.feature_names,
            dataset.dates,
            dataset.grid,
            dataset.mask,
        )

    def split_temporal_window(
        self,
        dataset: Dataset,
        start_index: int,
        end_index: int,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
    ) -> DatasetSplits:
        """Chronologically split samples contained within an exclusive time range."""
        if not 0 <= start_index < end_index <= len(dataset.dates):
            raise ValueError('Temporal window indices are outside the dataset range.')
        if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
            raise ValueError('Temporal split ratios must sum to 1.0.')
        time_values = np.arange(start_index, end_index)
        train_end = int(time_values.size * train_ratio)
        validation_end = train_end + int(time_values.size * validation_ratio)
        if (
            train_end < 1
            or validation_end <= train_end
            or validation_end >= time_values.size
        ):
            raise ValueError(
                'Temporal window needs at least one acquisition per subset.'
            )
        return DatasetSplits(
            self._subset(
                dataset, np.isin(dataset.time_indices, time_values[:train_end])
            ),
            self._subset(
                dataset,
                np.isin(dataset.time_indices, time_values[train_end:validation_end]),
            ),
            self._subset(
                dataset, np.isin(dataset.time_indices, time_values[validation_end:])
            ),
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
