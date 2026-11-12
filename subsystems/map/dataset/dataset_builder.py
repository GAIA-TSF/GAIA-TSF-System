"""Causal forecast samples built from aligned DAG feature stacks.

One sample is one pixel (series) at one forecast origin ``t``. Its predictors
are the configured DAG stacks at acquisition ``t`` (or the ``look_back``
acquisitions ending at ``t`` for sequence models). DAG ``*_lagK`` stacks hold
values from ``t - K`` at row ``t``, so every predictor is known at the origin.
Targets are the configured target stack at acquisitions ``t + 1 .. t + H``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from subsystems.map.dataset.feature_loader import LoadedFeatures, RasterGrid


@dataclass(frozen=True)
class ForecastSamples:
    """Origin-aligned predictors, persistence anchors, and future targets."""

    features: np.ndarray
    anchors: np.ndarray
    anchor_indices: np.ndarray
    targets: np.ndarray
    origin_indices: np.ndarray
    pixel_indices: np.ndarray
    feature_names: tuple[str, ...]
    dates: tuple[str, ...]
    grid: RasterGrid

    @property
    def horizon(self) -> int:
        return int(self.targets.shape[1])

    @property
    def changes(self) -> np.ndarray:
        """Model targets: change from the last valid value at the origin."""
        return self.targets - self.anchors[:, np.newaxis]

    def subset(self, include: np.ndarray, target_end: int | None = None) -> 'ForecastSamples':
        """Select rows; mask targets at or after ``target_end`` (purging)."""
        targets = self.targets[include].copy()
        if target_end is not None:
            target_index = (
                self.origin_indices[include, np.newaxis]
                + np.arange(1, self.horizon + 1)[np.newaxis, :]
            )
            targets[target_index >= target_end] = np.nan
        return replace(
            self,
            features=self.features[include],
            anchors=self.anchors[include],
            anchor_indices=self.anchor_indices[include],
            targets=targets,
            origin_indices=self.origin_indices[include],
            pixel_indices=self.pixel_indices[include],
        )


@dataclass(frozen=True)
class SampleSplits:
    """Train, validation and test subsets."""

    train: ForecastSamples
    validation: ForecastSamples
    test: ForecastSamples

    def items(self) -> tuple[tuple[str, ForecastSamples], ...]:
        return (
            ('train', self.train),
            ('validation', self.validation),
            ('test', self.test),
        )


def build_forecast_samples(
    loaded: LoadedFeatures,
    feature_names: list[str],
    target_feature: str,
    horizon: int,
    look_back: int | None = None,
    origin_range: tuple[int, int] | None = None,
) -> ForecastSamples:
    """Build one sample per valid (origin, pixel) inside ``loaded.mask``.

    An origin is valid when every predictor is finite (over the whole
    ``look_back`` window for sequence models) and a target value has been
    observed at or before the origin. Future targets may be missing (NaN).
    """
    if horizon < 1:
        raise ValueError('Forecast horizon must be positive.')
    if target_feature not in loaded.features:
        raise KeyError(f'Target feature was not loaded: {target_feature}')
    missing = [name for name in feature_names if name not in loaded.features]
    if missing:
        raise KeyError(f'Predictors were not loaded: {", ".join(missing)}')
    rows, columns = np.nonzero(loaded.mask)
    target = loaded.features[target_feature][:, rows, columns]
    predictors = np.stack(
        [loaded.features[name][:, rows, columns] for name in feature_names], axis=-1
    )
    time_count, pixel_count = target.shape

    # Persistence anchor: last valid target at or before each origin.
    observed_index = np.where(np.isfinite(target), np.arange(time_count)[:, None], -1)
    anchor_index = np.maximum.accumulate(observed_index, axis=0)
    anchors = np.where(
        anchor_index >= 0,
        target[np.maximum(anchor_index, 0), np.arange(pixel_count)],
        np.nan,
    )

    finite = np.all(np.isfinite(predictors), axis=-1)
    window = 1 if look_back is None else int(look_back)
    if window < 1:
        raise ValueError('look_back must be positive.')
    invalid_count = np.cumsum(~finite, axis=0)
    shifted = np.zeros_like(invalid_count)
    shifted[window:] = invalid_count[:-window]
    valid = (invalid_count - shifted == 0) & (anchor_index >= 0)
    valid[: window - 1] = False
    if origin_range is not None:
        in_range = np.zeros(time_count, dtype=bool)
        in_range[origin_range[0] : origin_range[1]] = True
        valid &= in_range[:, None]

    origin, pixel = np.nonzero(valid)
    if origin.size == 0:
        raise ValueError(
            'No forecast origins have finite predictors and an observed target. '
            'Check predictor and target coverage within the selected pixels.'
        )
    if look_back is None:
        features = predictors[origin, pixel]
    else:
        steps = origin[:, None] - np.arange(window - 1, -1, -1)[None, :]
        features = predictors[steps, pixel[:, None]]

    leads = np.arange(1, horizon + 1)
    target_index = origin[:, None] + leads[None, :]
    future = np.full(target_index.shape, np.nan)
    inside = target_index < time_count
    future[inside] = target[target_index[inside], np.broadcast_to(pixel[:, None], target_index.shape)[inside]]
    return ForecastSamples(
        features=features,
        anchors=anchors[origin, pixel],
        anchor_indices=anchor_index[origin, pixel],
        targets=future,
        origin_indices=origin.astype(np.int64),
        pixel_indices=(rows[pixel] * loaded.grid.width + columns[pixel]).astype(np.int64),
        feature_names=tuple(feature_names),
        dates=loaded.dates,
        grid=loaded.grid,
    )


def _ratios(train: float, validation: float, test: float) -> None:
    if not np.isclose(train + validation + test, 1.0):
        raise ValueError('Split ratios must sum to 1.0.')
    if min(train, validation, test) <= 0.0:
        raise ValueError('Split ratios must all be positive.')


def split_temporal(
    samples: ForecastSamples,
    start: int,
    end: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> SampleSplits:
    """Chronological split of origins in ``[start, end)``.

    Targets are purged at each subset's end, so no training target is observed
    during validation or test and no target falls outside the window.
    """
    _ratios(train_ratio, validation_ratio, test_ratio)
    if not 0 <= start < end <= len(samples.dates):
        raise ValueError('Temporal window indices are outside the dataset range.')
    count = end - start
    train_end = start + int(count * train_ratio)
    validation_end = train_end + int(count * validation_ratio)
    if train_end <= start or validation_end <= train_end or validation_end >= end:
        raise ValueError('Temporal split needs at least one acquisition per subset.')
    origin = samples.origin_indices
    return SampleSplits(
        *(
            _non_empty(samples.subset((origin >= a) & (origin < b), target_end=b), name)
            for name, a, b in (
                ('training', start, train_end),
                ('validation', train_end, validation_end),
                ('test', validation_end, end),
            )
        )
    )


def split_spatial(
    samples: ForecastSamples,
    start: int,
    end: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> SampleSplits:
    """Assign whole pixel histories in ``[start, end)`` to one subset each."""
    _ratios(train_ratio, validation_ratio, test_ratio)
    if not 0 <= start < end <= len(samples.dates):
        raise ValueError('Spatial window indices are outside the dataset range.')
    in_window = (samples.origin_indices >= start) & (samples.origin_indices < end)
    pixels = np.random.default_rng(random_seed).permutation(
        np.unique(samples.pixel_indices[in_window])
    )
    train_end = int(pixels.size * train_ratio)
    validation_end = train_end + int(pixels.size * validation_ratio)
    if train_end < 1 or validation_end <= train_end or validation_end >= pixels.size:
        raise ValueError('Spatial split needs at least one pixel per subset.')
    expected = np.unique(samples.origin_indices[in_window])
    subsets = []
    for name, chosen in (
        ('training', pixels[:train_end]),
        ('validation', pixels[train_end:validation_end]),
        ('test', pixels[validation_end:]),
    ):
        subset = _non_empty(
            samples.subset(in_window & np.isin(samples.pixel_indices, chosen), end),
            name,
        )
        absent = np.setdiff1d(expected, subset.origin_indices)
        if absent.size:
            missing_dates = ', '.join(samples.dates[int(i)] for i in absent[:5])
            raise ValueError(
                f'Spatial {name} subset has no valid pixels on: {missing_dates}. '
                'Increase the subset ratio or improve feature coverage.'
            )
        subsets.append(subset)
    return SampleSplits(*subsets)


def rolling_origin_folds(
    samples: ForecastSamples, n_folds: int
) -> list[tuple[ForecastSamples, ForecastSamples]]:
    """Expanding-window chronological folds over the sample origins.

    The origin range is cut into ``n_folds + 1`` blocks. Fold ``k`` fits on
    blocks ``0..k`` and validates on origins in block ``k + 1``. Fitting
    targets are purged at the fit end. Validation targets may extend past
    their block, but never past the end of the origin range, so every lead is
    scored without touching data held out beyond ``samples``.
    """
    if n_folds < 1:
        raise ValueError('tuning.folds must be positive.')
    origins = samples.origin_indices
    start, end = int(origins.min()), int(origins.max()) + 1
    edges = np.linspace(start, end, n_folds + 2).round().astype(int)
    if np.any(np.diff(edges) < 1):
        raise ValueError(f'Too few forecast origins for {n_folds} tuning folds.')
    folds = []
    for fold in range(n_folds):
        fit_end, validation_end = int(edges[fold + 1]), int(edges[fold + 2])
        fit = samples.subset(origins < fit_end, target_end=fit_end)
        validation = samples.subset(
            (origins >= fit_end) & (origins < validation_end), target_end=end
        )
        folds.append(
            (
                _non_empty(fit, f'fold {fold + 1} fitting'),
                _non_empty(validation, f'fold {fold + 1} validation'),
            )
        )
    return folds


def _non_empty(samples: ForecastSamples, name: str) -> ForecastSamples:
    if not np.any(np.isfinite(samples.targets)):
        raise ValueError(f'The {name} subset contains no observed targets.')
    return samples
