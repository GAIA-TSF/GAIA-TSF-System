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
from ale_3levels.state import load_state, save_state

LOGGER = logging.getLogger(__name__)
ENGINES = ('fixed_gates', 'three_level')


def select_model(config: dict[str, Any], model_id: str | None = None) -> str | None:
    """The MAP model that drives the alerts (B2.5): CLI, else input.model_id."""
    return model_id or (config.get('input') or {}).get('model_id')


def table_for(
    config: dict[str, Any], model_id: str | None, input_path: Path | None = None
) -> Path:
    """The residual table of the selected model.

    ``input.residuals`` is either one path or a mapping ``{model_id: path}``
    (one MAP table per model, each with its own engine-B calibration).
    """
    if input_path is not None:
        return Path(input_path)
    residuals = (config.get('input') or {}).get('residuals')
    if residuals is None:
        raise ValueError('No MAP table given (--input or input.residuals).')
    if isinstance(residuals, dict):
        if model_id not in residuals:
            raise ValueError(
                f'input.residuals has no table for model {model_id!r}; '
                f'configured: {sorted(residuals)}.'
            )
        return Path(residuals[model_id])
    return Path(residuals)


def load_input(
    config: dict[str, Any],
    input_path: Path | None = None,
    model_id: str | None = None,
) -> ForecastContract:
    """Load the MAP residual table of the selected model."""
    source = config.get('input') or {}
    site = config.get('site') or {}
    model = select_model(config, model_id)
    path = table_for(config, model, input_path)
    mask_file = source.get('mask_file')
    return load_contract(
        path,
        lead=int(source.get('lead', 1)),
        model_id=model,
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


def make_engine(
    name: str,
    config: dict[str, Any],
    engine_dir: Path,
    calibration_dir: Path | None = None,
):
    engine_config = config['engine_configs'][name]
    if name == 'fixed_gates':
        return FixedGatesEngine(engine_config)
    run_mode = (config.get('input') or {}).get('run_mode') or UNKNOWN
    return ThreeLevelEngine(
        engine_config,
        engine_dir / 'engine_state',
        run_mode=run_mode,
        calibration_dir=calibration_dir,
    )


def run(
    engine: str,
    out_dir: Path,
    *,
    config_path: Path,
    input_path: Path | None = None,
    calibration_dir: Path | None = None,
    model_id: str | None = None,
    state_dir: Path | None = None,
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
    contract = load_input(config, input_path, model_id)
    sectors = sectors_for(config, contract)
    spatial = config.get('spatial') or {}
    written: dict[str, dict[str, Path]] = {}
    for name in names:
        engine_dir = out_dir / name
        scorer = make_engine(name, config, engine_dir, calibration_dir)
        scorer.fit(contract)
        # B2.6: engine B continues from the previous run's state (resets,
        # identity, accumulators); its new state is written next to the output.
        state = load_state(state_dir) if (state_dir and name == 'three_level') else None
        result = scorer.score(contract, state)
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
        if getattr(scorer, 'last_state', None) is not None:
            written[name]['state'] = save_state(scorer.last_state, engine_dir / 'state')
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


def calibrate(
    out_dir: Path,
    *,
    config_path: Path,
    input_path: Path | None = None,
    model_id: str | None = None,
) -> Path:
    """Freeze engine-B calibration for one model and lead (B2.3, spec §6.3).

    Fits L1, L2 and L3 baselines from the pre-monitoring records of the
    selected MAP table (``calibration.population`` decides which) and writes
    them to ``<out>/calibration/<model_id>/lead<k>/`` with a manifest. Use the
    returned ``<out>/calibration`` directory with ``run --calibration``.
    """
    import tempfile

    from ale_3levels.engines.three_level.pipeline import (
        ThreeLevelPipeline,
        inputs_from_contract,
    )

    config = load_config(config_path)
    out_dir = Path(out_dir)
    root = out_dir / 'calibration'
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'{root} is not empty; calibrations are frozen once.')
    contract = load_input(config, input_path, model_id)
    run_mode = (config.get('input') or {}).get('run_mode') or UNKNOWN
    with tempfile.TemporaryDirectory() as scratch:
        pipeline = ThreeLevelPipeline(
            config['engine_configs']['three_level'], Path(scratch), write_legacy=False
        )
        result = pipeline.run(inputs_from_contract(contract, run_mode=run_mode))
        identity = result.metadata['calibration_identity']
        source = Path(result.metadata['l1_calibration_path']).parent
        target = root / identity['model_id'] / f'lead{identity["lead"]}'
        target.mkdir(parents=True)
        for path in sorted(source.glob('*.json')):
            (target / path.name).write_bytes(path.read_bytes())
    write_json(
        out_dir / 'calibration_manifest.json',
        {
            'ale_version': __version__,
            'identity': identity,
            'calibration_population': result.metadata['calibration_population'],
            'zscore_formula': result.metadata['zscore_formula'],
            'n_calibration': result.metadata['n_calibration'],
            'files': sorted(path.name for path in target.iterdir()),
            'input': input_identity(contract),
        },
    )
    return root
