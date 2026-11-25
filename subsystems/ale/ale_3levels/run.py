"""``run``: score MAP output with one or both engines and write the common schema.

Only saved MAP files are read (through the contract). Changing any ALE
threshold requires only re-running this command; MAP outputs are never
written. The engines execute no action.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ale_3levels import __version__
from ale_3levels.config import config_hash, load_config
from ale_3levels.contract import ForecastContract, load_contract, read_mask
from ale_3levels.engines.fixed_gates import FixedGatesEngine
from ale_3levels.engines.three_level.calibration import UNKNOWN
from ale_3levels.engines.three_level.engine import ThreeLevelEngine
from ale_3levels.outputs import input_identity, write_engine_output, write_json
from ale_3levels.spatial import aggregate_sectors, read_sectors, whole_grid_sectors

LOGGER = logging.getLogger(__name__)
ENGINES = ('fixed_gates', 'three_level')


def load_input(
    config: dict[str, Any], input_path: Path | None = None
) -> ForecastContract:
    """Load the MAP residual table named by the CLI or the configuration."""
    source = config.get('input') or {}
    site = config.get('site') or {}
    path = input_path or source.get('residuals')
    if path is None:
        raise ValueError('No MAP table given (--input or input.residuals).')
    mask_file = source.get('mask_file')
    return load_contract(
        Path(path),
        lead=int(source.get('lead', 1)),
        model_id=source.get('model_id'),
        source_units_override=source.get('source_units_override'),
        mask=read_mask(Path(mask_file)) if mask_file else None,
        site_id=site.get('site_id'),
        los_sign_convention=site.get(
            'los_sign_convention', 'negative_away_from_satellite'
        ),
        reference_date=site.get('reference_date'),
        reference_point=site.get('reference_point'),
    )


def sectors_for(config: dict[str, Any], contract: ForecastContract):
    spatial = config.get('spatial') or {}
    if spatial.get('sectors_file'):
        return read_sectors(
            spatial['sectors_file'],
            str(spatial.get('sector_id_field', 'sector_id')),
            contract.grid.shape,
            contract.grid.transform,
            contract.grid.crs,
            contract.mask,
        )
    return whole_grid_sectors(contract.mask)


def make_engine(name: str, config: dict[str, Any], engine_dir: Path):
    engine_config = config['engine_configs'][name]
    if name == 'fixed_gates':
        return FixedGatesEngine(engine_config)
    run_mode = (config.get('input') or {}).get('run_mode') or UNKNOWN
    return ThreeLevelEngine(
        engine_config, engine_dir / 'engine_state', run_mode=run_mode
    )


def run(
    engine: str,
    out_dir: Path,
    *,
    config_path: Path,
    input_path: Path | None = None,
) -> dict[str, dict[str, Path]]:
    """Score with ``engine`` (``fixed_gates``, ``three_level`` or ``both``)."""
    names = ENGINES if engine == 'both' else (engine,)
    if any(name not in ENGINES for name in names):
        raise ValueError(f'Unknown engine {engine!r}.')
    config = load_config(config_path)
    out_dir = Path(out_dir)
    for name in names:
        target = out_dir / name
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(
                f'{target} is not empty; ALE outputs are written once. '
                'Use a new --out directory.'
            )
    contract = load_input(config, input_path)
    sectors = sectors_for(config, contract)
    spatial = config.get('spatial') or {}
    written: dict[str, dict[str, Path]] = {}
    for name in names:
        engine_dir = out_dir / name
        scorer = make_engine(name, config, engine_dir)
        scorer.fit(contract)
        result = scorer.score(contract, None)
        summary = aggregate_sectors(
            result.severity,
            sectors,
            dict(spatial.get('coherence') or {}),
            int(spatial.get('min_pixels', 25)),
            float(spatial.get('min_fraction', 0.10)),
        )
        written[name] = write_engine_output(
            result,
            contract,
            summary,
            engine_dir,
            extra={
                'ale_version': __version__,
                'ale_config_path': config['_config_path'],
            },
        )
        LOGGER.info('Engine %s written to %s', name, engine_dir)
    write_json(
        out_dir / 'run.json',
        {
            'ale_version': __version__,
            'engines': list(names),
            'config_path': config['_config_path'],
            'config_hash': config_hash(
                {key: value for key, value in config.items() if key != '_config_path'}
            ),
            'input': input_identity(contract),
        },
    )
    return written
