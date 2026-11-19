"""MAP -> ALE hand-off contract: read, validate and harmonize MAP tables.

ALE reads only saved MAP outputs. The input is MAP's long-format forecast
residual table (``residuals.npz`` + ``residuals.json``, table schema 2.0, one
row per ``(series, forecast origin, lead)``). This module never imports MAP;
the column names and schema versions it accepts are listed here explicitly.

Harmonization (plan §5, decisions Q2/Q7). ALE works in mm and mm/day only:

* cumulative LOS: ``observed_mm``, ``predicted_mm``, ``persistence_mm``,
  ``residual_mm`` = MAP value x ``scale`` (source unit -> mm);
* rates over the forecast interval ``dt = target_date - last_observation_date``
  (days): ``observed_mm_per_day = (observed - persistence) / dt * scale``,
  ``predicted_mm_per_day = (predicted - persistence) / dt * scale``,
  ``residual_mm_per_day = residual / dt * scale``.

The operation order is fixed: the golden capture uses the same expressions,
so the stacks are bit-identical.

``split`` (spec field) combines MAP's two splits (Q3): the temporal ``period``
(calibration | monitoring | outside) and the model ``sample_role`` (train |
validation | test | none). Calibration-period rows that were neither trained
on nor used for validation/early stopping are ``calibration``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

CONTRACT_VERSION = '1.0'
TABLE_SCHEMA_VERSIONS = frozenset({'2.0'})
REQUIRED_COLUMNS = (
    'series_id',
    'row',
    'col',
    'forecast_origin_date',
    'lead',
    'target_date',
    'target_date_status',
    'last_observation_date',
    'persistence_value',
    'predicted_value',
    'period',
    'sample_role',
    'observed_value',
    'residual',
)
REQUIRED_METADATA = (
    'schema_version',
    'model',
    'model_version',
    'target_variable',
    'target_feature',
    'target_units',
    'grid',
    'monitoring_window',
)
UNIT_TO_MM = {'mm': 1.0, 'm': 1000.0}
PERIODS = frozenset({'calibration', 'monitoring', 'outside'})
SAMPLE_ROLES = frozenset({'train', 'validation', 'test', 'none'})
LOS_SIGN_CONVENTIONS = frozenset({'negative_away_from_satellite'})
# The table's target must be cumulative LOS for the *_mm fields to be
# cumulative displacement. Velocity targets need MAP's observed_series export.
CUMULATIVE_TARGETS = frozenset({'cumulative_los'})

# split codes (int8); index into SPLIT_NAMES.
NO_RECORD, TRAIN, VALIDATION, CALIBRATION, MONITORING, OUTSIDE = range(6)
SPLIT_NAMES = (
    'no_record',
    'train',
    'validation',
    'calibration',
    'monitoring',
    'outside',
)

CUMULATIVE_FIELDS = ('observed_mm', 'predicted_mm', 'persistence_mm', 'residual_mm')
RATE_FIELDS = ('observed_mm_per_day', 'predicted_mm_per_day', 'residual_mm_per_day')


class ContractError(ValueError):
    """The MAP output does not satisfy the hand-off contract."""


@dataclass(frozen=True)
class Grid:
    """Raster grid of the MAP table (pixel ``(row, col)`` addressing)."""

    crs: str
    transform: tuple[float, ...]
    height: int
    width: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


@dataclass(frozen=True)
class ForecastContract:
    """Harmonized, dense ``(time, row, col)`` view of one lead of a MAP table.

    ``dates`` are target (acquisition) dates. ``series_dates``/``observed_series_mm``
    hold the observed cumulative LOS on every acquisition the table knows,
    including forecast origins that are never a target.
    """

    attrs: dict[str, Any]
    grid: Grid
    lead: int
    dates: np.ndarray
    forecast_origin: np.ndarray
    delta_t_days: np.ndarray
    observed_mm: np.ndarray
    predicted_mm: np.ndarray
    persistence_mm: np.ndarray
    residual_mm: np.ndarray
    observed_mm_per_day: np.ndarray
    predicted_mm_per_day: np.ndarray
    residual_mm_per_day: np.ndarray
    split: np.ndarray
    mask: np.ndarray
    valid_mask: np.ndarray
    series_dates: np.ndarray
    observed_series_mm: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def monitoring_start_index(self) -> int:
        """First date index at or after the table's monitoring start."""
        return int(np.searchsorted(self.dates, self.attrs['monitoring_start']))

    def split_mask(self, *names: str) -> np.ndarray:
        """Boolean ``(time, row, col)`` mask of records in the named splits."""
        codes = [SPLIT_NAMES.index(name) for name in names]
        return np.isin(self.split, codes)


def read_mask(path: Path) -> np.ndarray:
    """Read a single-band mask GeoTIFF (non-zero = inside)."""
    import rasterio

    with rasterio.open(path) as raster:
        return raster.read(1) > 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _iso_days(values: np.ndarray, name: str) -> np.ndarray:
    """ISO date strings (date part only) -> datetime64[D]; clear error otherwise."""
    try:
        return np.asarray(values).astype('<U10').astype('datetime64[D]')
    except ValueError as error:
        raise ContractError(
            f'Column {name!r} holds a value that is not an ISO date.'
        ) from error


def _read_table(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(path)
    npz = path.with_suffix('.npz')
    sidecar_path = path.with_suffix('.json')
    if not npz.is_file():
        raise ContractError(f'MAP residual table not found: {npz}')
    if not sidecar_path.is_file():
        raise ContractError(f'MAP table sidecar not found: {sidecar_path}')
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise ContractError(
            f'MAP table sidecar is not valid JSON: {sidecar_path}'
        ) from error
    with np.load(npz, allow_pickle=False) as archive:
        table = {name: archive[name] for name in archive.files}
    return table, sidecar


def validate_table(table: dict[str, np.ndarray], sidecar: dict[str, Any]) -> None:
    """Raise :class:`ContractError` unless the table matches the contract."""
    missing_meta = [key for key in REQUIRED_METADATA if key not in sidecar]
    if missing_meta:
        raise ContractError(f'MAP table sidecar lacks required keys: {missing_meta}')
    version = str(sidecar['schema_version'])
    if version not in TABLE_SCHEMA_VERSIONS:
        raise ContractError(
            f'Unsupported MAP table schema_version {version!r}; '
            f'ALE accepts {sorted(TABLE_SCHEMA_VERSIONS)}.'
        )
    missing = [name for name in REQUIRED_COLUMNS if name not in table]
    if missing:
        raise ContractError(f'MAP table lacks required columns: {missing}')
    sizes = {table[name].shape for name in REQUIRED_COLUMNS}
    if len(sizes) != 1 or len(next(iter(sizes))) != 1:
        raise ContractError('MAP table columns must be 1-D and of equal length.')
    grid = sidecar['grid']
    for key in ('crs', 'transform', 'height', 'width'):
        if key not in grid:
            raise ContractError(f'MAP table sidecar grid lacks {key!r}.')
    height, width = int(grid['height']), int(grid['width'])
    row, col = table['row'], table['col']
    if row.size and (
        row.min() < 0 or row.max() >= height or col.min() < 0 or col.max() >= width
    ):
        raise ContractError('MAP table row/col lie outside the declared grid.')
    if not np.array_equal(table['series_id'], row * width + col):
        raise ContractError('MAP table series_id must equal row * width + col.')
    unknown_periods = set(np.unique(table['period']).tolist()) - PERIODS
    if unknown_periods:
        raise ContractError(f'Unknown MAP period values: {sorted(unknown_periods)}')
    unknown_roles = set(np.unique(table['sample_role']).tolist()) - SAMPLE_ROLES
    if unknown_roles:
        raise ContractError(f'Unknown MAP sample_role values: {sorted(unknown_roles)}')
    if table['lead'].size and table['lead'].min() < 1:
        raise ContractError('MAP lead must be a positive number of acquisitions.')


def _derive_split(period: np.ndarray, role: np.ndarray) -> np.ndarray:
    split = np.full(period.shape, CALIBRATION, dtype=np.int8)
    split[role == 'train'] = TRAIN
    split[role == 'validation'] = VALIDATION
    split[period == 'monitoring'] = MONITORING
    split[period == 'outside'] = OUTSIDE
    return split


def _dense(
    values: np.ndarray, time_index: np.ndarray, row: np.ndarray, col: np.ndarray, shape
) -> np.ndarray:
    stack = np.full(shape, np.nan, dtype=np.float64)
    stack[time_index, row, col] = values
    return stack


def _observed_series(
    target: np.ndarray,
    last: np.ndarray,
    observed: np.ndarray,
    persistence: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Observed cumulative LOS on every acquisition the rows reference.

    Each row gives the observation at its target date and, as the persistence
    value, the observation at its last-observation date. Both describe the
    same acquisition raster, so they must agree where they overlap.
    """
    dates = np.unique(np.concatenate([target, last]))
    shape = (dates.size, *grid_shape)
    series = np.full(shape, np.nan, dtype=np.float64)
    series[np.searchsorted(dates, last), row, col] = persistence
    at_target = np.searchsorted(dates, target)
    previous = series[at_target, row, col]
    both = np.isfinite(previous) & np.isfinite(observed)
    if np.any(both):
        tolerance = 1e-9 * max(1.0, float(np.nanmax(np.abs(observed[both]))))
        if np.max(np.abs(previous[both] - observed[both])) > tolerance:
            raise ContractError(
                'Observed values and persistence values disagree for the same '
                'acquisition; the table mixes different LOS inputs.'
            )
    series[at_target, row, col] = np.where(np.isfinite(observed), observed, previous)
    return dates.astype('<U10'), series


def load_contract(
    path: Path,
    *,
    lead: int = 1,
    model_id: str | None = None,
    source_units_override: str | None = None,
    mask: np.ndarray | None = None,
    site_id: str | None = None,
    los_sign_convention: str = 'negative_away_from_satellite',
    reference_date: str | None = None,
    reference_point: str | None = None,
) -> ForecastContract:
    """Read, validate and harmonize one lead of a MAP residual table.

    Args:
        path: ``residuals.npz`` (its ``.json`` sidecar must sit next to it).
        lead: Lead (acquisitions after the origin) to extract.
        model_id: Expected MAP model (``trf``/``lstm``); mismatch is an error.
        source_units_override: Unit of the table values when the sidecar's
            ``target_units`` is known to be wrong (synthetic sites write m).
        mask: Optional ``(height, width)`` TSF mask; default = pixels in the table.
        site_id: Site identifier; default = the experiment directory name.
        los_sign_convention: Recorded LOS sign convention.
        reference_date: Date of zero cumulative LOS; default = first acquisition.
        reference_point: Spatial reference description, recorded as given.
    """
    path = Path(path).with_suffix('.npz')
    table, sidecar = _read_table(path)
    validate_table(table, sidecar)
    if los_sign_convention not in LOS_SIGN_CONVENTIONS:
        raise ContractError(f'Unknown LOS sign convention {los_sign_convention!r}.')
    if model_id is not None and str(sidecar['model']) != model_id:
        raise ContractError(
            f'MAP table was produced by model {sidecar["model"]!r}; '
            f'the ALE configuration selects {model_id!r}.'
        )
    if str(sidecar['target_variable']) not in CUMULATIVE_TARGETS:
        raise ContractError(
            f'MAP target {sidecar["target_variable"]!r} is not cumulative LOS; '
            "this table needs MAP's observed_series export (not yet supported)."
        )
    declared = str(sidecar['target_units'])
    unit = source_units_override or declared
    if unit not in UNIT_TO_MM:
        raise ContractError(
            f'Cannot convert MAP unit {unit!r} to mm; known units {sorted(UNIT_TO_MM)}.'
        )
    if source_units_override and source_units_override != declared:
        LOGGER.warning(
            'MAP table declares %s but ALE uses the configured source unit %s.',
            declared,
            source_units_override,
        )
    scale = UNIT_TO_MM[unit]

    keep = (table['lead'] == lead) & (table['target_date_status'] == 'acquired')
    if not np.any(keep):
        raise ContractError(f'MAP table has no acquired rows for lead {lead}.')
    rows = {name: table[name][keep] for name in REQUIRED_COLUMNS}
    target_days = _iso_days(rows['target_date'], 'target_date')
    last_days = _iso_days(rows['last_observation_date'], 'last_observation_date')
    _iso_days(rows['forecast_origin_date'], 'forecast_origin_date')
    dt = (target_days - last_days).astype(np.float64)
    if np.any(dt <= 0):
        raise ContractError('Every row needs target_date after last_observation_date.')

    grid_meta = sidecar['grid']
    grid = Grid(
        crs=str(grid_meta['crs']),
        transform=tuple(float(value) for value in grid_meta['transform']),
        height=int(grid_meta['height']),
        width=int(grid_meta['width']),
    )
    target = rows['target_date'].astype('<U10')
    dates = np.unique(target)
    time_index = np.searchsorted(dates, target)
    row, col = rows['row'], rows['col']
    flat = (time_index * grid.height + row) * grid.width + col
    if np.unique(flat).size != flat.size:
        raise ContractError(
            f'MAP table has duplicate (target_date, pixel) rows for lead {lead}.'
        )
    shape = (dates.size, *grid.shape)

    origin = rows['forecast_origin_date'].astype('<U10')
    forecast_origin = np.full(dates.size, '', dtype='<U10')
    forecast_origin[time_index] = origin
    if np.any(forecast_origin[time_index] != origin):
        raise ContractError(
            'Rows sharing a target date must share their forecast origin.'
        )

    persistence = rows['persistence_value']
    observed = rows['observed_value']
    predicted = rows['predicted_value']
    residual = rows['residual']
    # Fixed operation order; the golden capture uses identical expressions.
    fields = {
        'observed_mm': observed * scale,
        'predicted_mm': predicted * scale,
        'persistence_mm': persistence * scale,
        'residual_mm': residual * scale,
        'observed_mm_per_day': (observed - persistence) / dt * scale,
        'predicted_mm_per_day': (predicted - persistence) / dt * scale,
        'residual_mm_per_day': residual / dt * scale,
    }
    stacks = {
        name: _dense(values, time_index, row, col, shape)
        for name, values in fields.items()
    }
    delta_t = _dense(dt, time_index, row, col, shape)
    split = np.full(shape, NO_RECORD, dtype=np.int8)
    split[time_index, row, col] = _derive_split(rows['period'], rows['sample_role'])

    if mask is None:
        mask = np.zeros(grid.shape, dtype=bool)
        mask[row, col] = True
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != grid.shape:
            raise ContractError(
                f'Mask shape {mask.shape} does not match the grid {grid.shape}.'
            )
    valid = (
        mask[np.newaxis]
        & np.isfinite(stacks['observed_mm'])
        & np.isfinite(stacks['predicted_mm'])
        & np.isfinite(delta_t)
    )

    series_dates, series = _observed_series(
        target,
        rows['last_observation_date'].astype('<U10'),
        fields['observed_mm'],
        fields['persistence_mm'],
        row,
        col,
        grid.shape,
    )
    table_sha = _sha256(path)
    tsf_version = sidecar.get('tsf_version')
    attrs = {
        'contract_version': CONTRACT_VERSION,
        'table_schema_version': str(sidecar['schema_version']),
        'table_path': str(path),
        'table_sha256': table_sha,
        'site_id': site_id or sidecar.get('site_id') or _site_from_path(path),
        'model_id': str(sidecar['model']),
        'model_version': str(sidecar['model_version']),
        'tsf_version': tsf_version or f'unversioned-{table_sha[:12]}',
        'tsf_version_provisional': tsf_version is None,
        'target_variable': str(sidecar['target_variable']),
        'target_feature': str(sidecar['target_feature']),
        'declared_units': declared,
        'source_units': unit,
        'units': {'cumulative': 'mm', 'rate': 'mm/day', 'delta_t': 'day'},
        'los_sign_convention': los_sign_convention,
        'reference_date': reference_date or str(series_dates[0]),
        'reference_point': reference_point,
        'calibration_window': list(sidecar.get('calibration_window') or []),
        'monitoring_window': list(sidecar['monitoring_window']),
        'monitoring_start': str(sidecar['monitoring_window'][0])[:10],
        'lead': int(lead),
    }
    return ForecastContract(
        attrs=attrs,
        grid=grid,
        lead=int(lead),
        dates=dates,
        forecast_origin=forecast_origin,
        delta_t_days=delta_t,
        split=split,
        mask=mask,
        valid_mask=valid,
        series_dates=series_dates,
        observed_series_mm=series,
        **stacks,
    )


def _site_from_path(path: Path) -> str:
    """``<site>/results/forecasts/<experiment>/residuals.npz`` -> ``<site>``."""
    parts = path.resolve().parts
    if len(parts) >= 5 and parts[-3] == 'forecasts' and parts[-4] == 'results':
        return parts[-5]
    return 'unknown'


def summary(contract: ForecastContract) -> dict[str, Any]:
    """JSON-ready description of a loaded contract (for ``validate``)."""
    counts = {
        name: int(np.sum(contract.split == code))
        for code, name in enumerate(SPLIT_NAMES)
    }
    calibration_dates = np.any(contract.split == CALIBRATION, axis=(1, 2))
    return {
        **contract.attrs,
        'grid': {
            'crs': contract.grid.crs,
            'transform': list(contract.grid.transform),
            'height': contract.grid.height,
            'width': contract.grid.width,
        },
        'dates': {
            'count': int(contract.dates.size),
            'first': str(contract.dates[0]),
            'last': str(contract.dates[-1]),
        },
        'mask_pixels': int(contract.mask.sum()),
        'records_by_split': counts,
        'calibration_split_dates': int(calibration_dates.sum()),
        'valid_records': int(contract.valid_mask.sum()),
    }
