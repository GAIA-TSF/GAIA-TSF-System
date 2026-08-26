"""Resolution of configured MAP temporal date windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class TemporalWindow:
    """Inclusive calendar window represented as zero-based time-index bounds."""

    start_index: int
    end_index: int
    start_date: str
    end_date: str


def resolve_temporal_window(
    dates: tuple[str, ...],
    config: dict[str, Any],
    name: str,
    *,
    end_inclusive: bool = True,
) -> TemporalWindow:
    """Resolve a named ISO-date window into an exclusive index range.

    Args:
        dates: Chronological acquisition date labels in ISO ``YYYY-MM-DD`` format.
        config: Dataset configuration containing ``temporal_windows``.
        name: Window name, normally ``calibration`` or ``monitoring``.

    Returns:
        Resolved temporal window with ``end_index`` as an exclusive bound.
    """
    windows = config.get('temporal_windows', {})
    if not isinstance(windows, dict) or not isinstance(windows.get(name), dict):
        raise KeyError(f'Missing datasets.*.temporal_windows.{name} configuration.')
    window = windows[name]
    start_value = window.get('start_date')
    end_value = window.get('end_date')
    if not isinstance(start_value, str) or not isinstance(end_value, str):
        raise ValueError(f'Temporal window {name} requires start_date and end_date.')
    try:
        start_date = date.fromisoformat(start_value)
        end_date = date.fromisoformat(end_value)
        acquisition_dates = [date.fromisoformat(value) for value in dates]
    except ValueError as exc:
        raise ValueError('MAP temporal windows require ISO YYYY-MM-DD acquisition dates.') from exc
    if end_date < start_date:
        raise ValueError(f'Temporal window {name} ends before it starts.')
    selected = [
        index
        for index, acquisition_date in enumerate(acquisition_dates)
        if start_date <= acquisition_date
        and (
            acquisition_date <= end_date
            if end_inclusive
            else acquisition_date < end_date
        )
    ]
    if not selected:
        raise ValueError(f'Temporal window {name} contains no acquisition dates.')
    return TemporalWindow(
        start_index=selected[0],
        end_index=selected[-1] + 1,
        start_date=start_value,
        end_date=end_value,
    )
