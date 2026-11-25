"""Connected-region filtering for spatial anomaly products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpatialCoherenceResult:
    """Coherence-qualified anomaly flags and their summary statistics."""

    binary_stack: np.ndarray
    summary: dict[str, Any]
    regions: tuple[SpatialCoherenceRegion, ...]


@dataclass(frozen=True)
class SpatialCoherenceRegion:
    """A region qualified for regional monitoring at a causal activation date.

    The support is frozen when its spatial and temporal coherence is first
    established.  This permits its pre-existing calibration history to be
    evaluated without allowing later observations to redefine that history.
    """

    support: np.ndarray
    activation_index: int


class SpatialCoherenceDetector:
    """Retain only spatially connected anomaly regions persistent through time."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Validate configurable area, overlap, and persistence requirements."""
        self.enabled = bool(config.get('enabled', False))
        self.minimum_area_pixels = self._positive_integer(config, 'minimum_area_pixels')
        self.persistence = self._positive_integer(config, 'persistence')
        self.minimum_overlap = float(config.get('minimum_overlap', 0.3))
        if not 0.0 <= self.minimum_overlap <= 1.0:
            raise ValueError('spatial_coherence.minimum_overlap must be in [0, 1].')
        self.connectivity = int(config.get('connectivity', 8))
        if self.connectivity not in {4, 8}:
            raise ValueError('spatial_coherence.connectivity must be 4 or 8.')

    def detect(
        self, binary_stack: np.ndarray, mask: np.ndarray
    ) -> SpatialCoherenceResult:
        """Filter anomaly regions by area and inter-acquisition spatial overlap."""
        if binary_stack.ndim != 3 or binary_stack.shape[1:] != mask.shape:
            raise ValueError('Coherence input stack and TSF mask are incompatible.')
        if not self.enabled:
            return SpatialCoherenceResult(
                binary_stack & mask[np.newaxis, :, :],
                {'enabled': False},
                (),
            )
        output = np.zeros_like(binary_stack, dtype=bool)
        previous: list[tuple[np.ndarray, int, int | None]] = []
        regions: list[SpatialCoherenceRegion] = []
        retained_regions = 0
        for index in range(binary_stack.shape[0]):
            components = self._components(binary_stack[index] & mask)
            current: list[tuple[np.ndarray, int, int | None]] = []
            for component in components:
                if int(np.count_nonzero(component)) < self.minimum_area_pixels:
                    continue
                run = 1
                region_id: int | None = None
                if previous:
                    best_overlap, prior_run, region_id = max(
                        (
                            (self._iou(component, prior), prior_run, prior_region_id)
                            for prior, prior_run, prior_region_id in previous
                        ),
                        key=lambda candidate: (candidate[0], candidate[1]),
                    )
                    if best_overlap >= self.minimum_overlap:
                        run = prior_run + 1
                    else:
                        region_id = None
                if run >= self.persistence:
                    output[index] |= component
                    retained_regions += 1
                    if region_id is None:
                        region_id = len(regions)
                        regions.append(
                            SpatialCoherenceRegion(
                                support=component.copy(),
                                activation_index=index,
                            )
                        )
                current.append((component, run, region_id))
            previous = current
        return SpatialCoherenceResult(
            output,
            {
                'enabled': True,
                'minimum_area_pixels': self.minimum_area_pixels,
                'persistence': self.persistence,
                'minimum_overlap': self.minimum_overlap,
                'connectivity': self.connectivity,
                'coherent_regions_retained': retained_regions,
                'coherence_qualified_region_count': len(regions),
                'region_activation_indices': [
                    region.activation_index for region in regions
                ],
                'coherent_pixels_by_acquisition': [
                    int(np.count_nonzero(values)) for values in output
                ],
            },
            tuple(regions),
        )

    def _components(self, values: np.ndarray) -> list[np.ndarray]:
        """Return connected Boolean components without adding a GIS dependency."""
        remaining = values.copy()
        components: list[np.ndarray] = []
        offsets = (
            ((-1, 0), (1, 0), (0, -1), (0, 1))
            if self.connectivity == 4
            else tuple(
                (row_offset, column_offset)
                for row_offset in (-1, 0, 1)
                for column_offset in (-1, 0, 1)
                if row_offset or column_offset
            )
        )
        while np.any(remaining):
            row, column = np.argwhere(remaining)[0]
            component = np.zeros_like(remaining, dtype=bool)
            queue = [(int(row), int(column))]
            remaining[row, column] = False
            while queue:
                current_row, current_column = queue.pop()
                component[current_row, current_column] = True
                for row_offset, column_offset in offsets:
                    neighbour_row = current_row + row_offset
                    neighbour_column = current_column + column_offset
                    if (
                        0 <= neighbour_row < remaining.shape[0]
                        and 0 <= neighbour_column < remaining.shape[1]
                        and remaining[neighbour_row, neighbour_column]
                    ):
                        remaining[neighbour_row, neighbour_column] = False
                        queue.append((neighbour_row, neighbour_column))
            components.append(component)
        return components

    @staticmethod
    def _iou(first: np.ndarray, second: np.ndarray) -> float:
        """Return intersection-over-union of two connected regions."""
        union = np.count_nonzero(first | second)
        return 0.0 if union == 0 else float(np.count_nonzero(first & second) / union)

    @staticmethod
    def _positive_integer(config: dict[str, Any], key: str) -> int:
        """Read a strictly positive integer spatial-coherence setting."""
        value = int(config.get(key, 1))
        if value < 1:
            raise ValueError(f'spatial_coherence.{key} must be at least one.')
        return value


# --- Sector aggregation (shared by both engines) ---------------------------


@dataclass(frozen=True)
class SectorSummary:
    """Per-sector, per-acquisition aggregated severity and pixel counts.

    ``severity`` is ``(time, sector)``: the highest ordinal level 0-3 reached
    by a coherence-qualified group of pixels in the sector (see
    :func:`aggregate_sectors`), -1 when the sector has no valid pixel.
    ``pixel_counts`` is ``(time, sector, 5)``: raw pixels per severity
    NO_DATA/0/1/2/3. ``coherent_counts`` is ``(time, sector, 3)``: coherent
    pixels at severity >= 1, >= 2, >= 3.
    """

    sector_ids: tuple[str, ...]
    severity: np.ndarray
    valid_pixels: np.ndarray
    pixel_counts: np.ndarray
    coherent_counts: np.ndarray
    rule: dict[str, Any]


def whole_grid_sectors(mask: np.ndarray) -> dict[str, np.ndarray]:
    """One sector covering the TSF mask (used when no polygons are configured)."""
    return {'tsf': np.asarray(mask, dtype=bool)}


def read_sectors(
    path: str,
    id_field: str,
    shape: tuple[int, int],
    transform: tuple[float, ...],
    crs: str,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Rasterize sector polygons onto the grid (pixel centres), within ``mask``."""
    import geopandas
    from affine import Affine
    from rasterio.features import rasterize

    frame = geopandas.read_file(path)
    if frame.crs is not None:
        frame = frame.to_crs(crs)
    if id_field not in frame.columns:
        raise ValueError(f'Sector file {path} has no field {id_field!r}.')
    affine = Affine(*transform[:6])
    sectors = {}
    for identifier, geometry in zip(frame[id_field], frame.geometry):
        burned = rasterize(
            [(geometry, 1)], out_shape=shape, transform=affine, fill=0, dtype='uint8'
        )
        sectors[str(identifier)] = (burned > 0) & mask
    if not sectors:
        raise ValueError(f'Sector file {path} contains no polygons.')
    return sectors


def aggregate_sectors(
    severity: np.ndarray,
    sectors: dict[str, np.ndarray],
    coherence: dict[str, Any],
    min_pixels: int,
    min_fraction: float,
) -> SectorSummary:
    """Sector level = highest level held by enough coherent pixels.

    For each level L in 1..3, pixels with severity >= L pass through
    :class:`SpatialCoherenceDetector` (minimum area, temporal persistence,
    overlap) on the whole grid. A sector reaches L at an acquisition when its
    coherent pixels at >= L number at least ``min_pixels`` or at least
    ``min_fraction`` of its valid pixels. Level 0 when it has valid pixels
    but reaches no level; -1 (NO_DATA) without valid pixels.
    """
    if min_pixels < 1 or not 0.0 < min_fraction <= 1.0:
        raise ValueError('Need min_pixels >= 1 and 0 < min_fraction <= 1.')
    severity = np.asarray(severity)
    ids = tuple(sectors)
    union = np.zeros(severity.shape[1:], dtype=bool)
    for cells in sectors.values():
        union |= cells
    detector = SpatialCoherenceDetector({**coherence, 'enabled': True})
    time_count = severity.shape[0]
    coherent_counts = np.zeros((time_count, len(ids), 3), dtype=np.int64)
    for level in (1, 2, 3):
        coherent = detector.detect(severity >= level, union).binary_stack
        for position, sector in enumerate(ids):
            coherent_counts[:, position, level - 1] = np.count_nonzero(
                coherent[:, sectors[sector]], axis=1
            )
    pixel_counts = np.zeros((time_count, len(ids), 5), dtype=np.int64)
    for position, sector in enumerate(ids):
        values = severity[:, sectors[sector]]
        for code, value in enumerate((-1, 0, 1, 2, 3)):
            pixel_counts[:, position, code] = np.count_nonzero(values == value, axis=1)
    valid = pixel_counts[:, :, 1:].sum(axis=2)
    by_count = coherent_counts >= min_pixels
    by_fraction = (coherent_counts >= min_fraction * valid[:, :, None]) & (
        coherent_counts > 0
    )
    enough = by_count | by_fraction
    sector_severity = np.where(valid > 0, 0, -1).astype(np.int8)
    for level in (1, 2, 3):
        sector_severity[enough[:, :, level - 1] & (valid > 0)] = level
    return SectorSummary(
        sector_ids=ids,
        severity=sector_severity,
        valid_pixels=valid,
        pixel_counts=pixel_counts,
        coherent_counts=coherent_counts,
        rule={
            'method': 'SpatialCoherenceDetector per level, then min_pixels or min_fraction',
            'coherence': {**coherence, 'enabled': True},
            'min_pixels': int(min_pixels),
            'min_fraction': float(min_fraction),
        },
    )
