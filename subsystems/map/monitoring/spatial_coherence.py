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

    def detect(self, binary_stack: np.ndarray, mask: np.ndarray) -> SpatialCoherenceResult:
        """Filter anomaly regions by area and inter-acquisition spatial overlap."""
        if binary_stack.ndim != 3 or binary_stack.shape[1:] != mask.shape:
            raise ValueError('Coherence input stack and TSF mask are incompatible.')
        if not self.enabled:
            return SpatialCoherenceResult(
                binary_stack & mask[np.newaxis, :, :],
                {'enabled': False},
            )
        output = np.zeros_like(binary_stack, dtype=bool)
        previous: list[tuple[np.ndarray, int]] = []
        retained_regions = 0
        for index in range(binary_stack.shape[0]):
            components = self._components(binary_stack[index] & mask)
            current: list[tuple[np.ndarray, int]] = []
            for component in components:
                if int(np.count_nonzero(component)) < self.minimum_area_pixels:
                    continue
                run = 1
                if previous:
                    best_overlap, prior_run = max(
                        (self._iou(component, prior), prior_run)
                        for prior, prior_run in previous
                    )
                    if best_overlap >= self.minimum_overlap:
                        run = prior_run + 1
                current.append((component, run))
                if run >= self.persistence:
                    output[index] |= component
                    retained_regions += 1
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
                'coherent_pixels_by_acquisition': [
                    int(np.count_nonzero(values)) for values in output
                ],
            },
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
