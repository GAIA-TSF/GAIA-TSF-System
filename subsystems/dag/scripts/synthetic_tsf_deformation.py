"""
synthetic_tsf_deformation.py

Synthetic 3-year TSF InSAR benchmark generator.

Outputs:
- LOS observations GeoTIFFs
- True LOS GeoTIFFs
- Hotspot probability GeoTIFFs
- Binary anomaly labels GeoTIFFs
- Failure stage GeoTIFFs
- Atmosphere GeoTIFFs
- Rainfall trigger GeoTIFFs
- Coherence mask GeoTIFF
- AOI and deformation-ROI mask GeoTIFFs
- Static TSF DEM GeoTIFF
- Raw daily meteodata CSV (precipitation and temperature observations)
- Metadata CSV
- Failure timeline CSV

Usage:
python3 subsystems/dag/scripts/synthetic_tsf_deformation.py

Or override the configured scenario:

python3 subsystems/dag/scripts/synthetic_tsf_deformation.py \
  --scenario spatial_propagation

And possibly re-direct the outputs:
python3 subsystems/dag/scripts/synthetic_tsf_deformation.py \
  --scenario external_trigger \
  --output-root /path/to/experiments
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import geopandas as gpd
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import Affine, from_origin
from rasterio.warp import transform_geom
from scipy.ndimage import gaussian_filter

### CONSTANTS ###

CONFIG_FILE = Path(__file__).with_name('synthetic_tsf_config.yaml')

FAILURE_START_DAY = 730
A_MAX = -0.10

# Static TSF terrain model (metres).
DEM_BASE_ELEVATION = 450.0
DEM_EAST_SLOPE = 0.012
DEM_SOUTH_SLOPE = 0.006
DEM_BASIN_DEPTH = 8.0
DEM_EMBANKMENT_HEIGHT = 15.0
DEM_EMBANKMENT_RADIUS = 39.0
DEM_EMBANKMENT_WIDTH = 3.0
DEM_ROUGHNESS_STD = 0.75

SIGMA_BOWL = 18

# Rainfall trigger events that contribute to instability. Weather is exported
# daily so the DAG can derive rolling features over complete lookback periods.
TRIGGER_EVENTS = {790: 45, 860: 62, 950: 78, 1030: 55}

INPUT_SUBDIRS = [
    'los',
    'true_los',
    'hotspot_prob',
    'labels',
    'failure_stage',
    'velocity',
    'acceleration',
    'trigger_response',
    'atmosphere',
    'rainfall',
    'aux',
]

STAGE_NAMES = {
    0: 'stable',
    1: 'latent',
    2: 'developing',
    3: 'accelerating',
    4: 'failure',
}


@dataclass
class SimulationContext:
    """Configuration and derived grid state shared by all simulation steps."""

    config: dict
    scenario_name: str
    scenario: dict
    outdir: Path
    inputs_dir: Path
    static_dir: Path
    crs: object
    pixel_size: float
    polygon_mode: bool
    nx: int
    ny: int
    spatial_bounds: tuple
    transform: Affine
    aoi_geometries: list | None
    aoi_mask: np.ndarray
    roi_mask: np.ndarray
    roi_weight: np.ndarray
    roi_center: np.ndarray
    x: np.ndarray
    y: np.ndarray
    bowl: np.ndarray
    years: float
    revisit_days: int
    start_date: datetime
    incidence_deg: float
    hydro_amp: float
    atm_sigma: float
    meas_sigma: float
    random_seed: int


### SETUP ###
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG_FILE)
    parser.add_argument('--scenario')
    parser.add_argument('--output-root', type=Path)
    return parser.parse_args(argv)


def _resolve_input_path(value, config_path):
    path = Path(value).expanduser()
    return (
        path if path.is_absolute() else (config_path.resolve().parent / path).resolve()
    )


def _read_geometries(path, target_crs=None):
    if not path.exists():
        raise FileNotFoundError(f'Polygon input not found: {path}')
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f'Polygon input has no CRS: {path}')
    frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty]
    if frame.empty:
        raise ValueError(f'Polygon input has no geometries: {path}')
    if not frame.geom_type.isin(['Polygon', 'MultiPolygon']).all():
        raise ValueError(f'Polygon input contains non-polygon geometry: {path}')
    if target_crs is not None:
        frame = frame.to_crs(target_crs)
    return frame.geometry.tolist(), frame.crs, tuple(frame.total_bounds)


def _resolve_output_dir(config, scenario_name, scenario, config_path, output_root):
    root = output_root or Path(config['simulation']['output_root'])
    if not root.is_absolute():
        root = (config_path.resolve().parent / root).resolve()
    output_directory = scenario['output_directory']
    if config['simulation'].get('append_scenario_name', True):
        scenario_suffix = f'_{scenario_name}'
        if not output_directory.endswith(scenario_suffix):
            output_directory += scenario_suffix
    return root / output_directory


def _build_grid(grid_config, config_path):
    """Return CRS, grid size, bounds and AOI geometries for the configured mode."""
    pixel_size = grid_config['pixel_size_m']
    polygon_mode = grid_config.get('mode', 'rectangular') == 'polygon'
    if polygon_mode:
        aoi_file = _resolve_input_path(grid_config['aoi_file'], config_path)
        aoi_geometries, crs, source_bounds = _read_geometries(aoi_file)
        left, bottom, right, top = source_bounds
        xmin = np.floor(left / pixel_size) * pixel_size
        ymax = np.ceil(top / pixel_size) * pixel_size
        xmax = np.ceil(right / pixel_size) * pixel_size
        ymin = np.floor(bottom / pixel_size) * pixel_size
        nx = int(round((xmax - xmin) / pixel_size))
        ny = int(round((ymax - ymin) / pixel_size))
        spatial_bounds = source_bounds
    else:
        crs = grid_config['crs']
        nx, ny = grid_config['width_px'], grid_config['height_px']
        xmin, ymax = grid_config['xmin'], grid_config['ymax']
        spatial_bounds = (
            xmin,
            ymax - ny * pixel_size,
            xmin + nx * pixel_size,
            ymax,
        )
        aoi_geometries = None
    transform = from_origin(xmin, ymax, pixel_size, pixel_size)
    return (
        crs,
        pixel_size,
        polygon_mode,
        nx,
        ny,
        spatial_bounds,
        transform,
        aoi_geometries,
    )


def _build_masks(grid_config, config_path, crs, nx, ny, transform, aoi_geometries):
    if aoi_geometries is None:
        aoi_mask = np.ones((ny, nx), dtype=bool)
        return aoi_mask, aoi_mask.copy()
    all_touched = grid_config.get('all_touched', False)
    aoi_mask = geometry_mask(
        aoi_geometries,
        (ny, nx),
        transform,
        invert=True,
        all_touched=all_touched,
    )
    roi_path = _resolve_input_path(grid_config['deformation_roi_file'], config_path)
    roi_geometries, _, _ = _read_geometries(roi_path, crs)
    roi_mask = (
        geometry_mask(
            roi_geometries,
            (ny, nx),
            transform,
            invert=True,
            all_touched=all_touched,
        )
        & aoi_mask
    )
    if not roi_mask.any():
        raise ValueError('The deformation ROI has no pixels inside the AOI grid')
    return aoi_mask, roi_mask


def _build_roi_weight(grid_config, roi_mask, aoi_mask):
    roi_edge_sigma = float(grid_config.get('deformation_roi_edge_sigma_px', 0))
    if roi_edge_sigma < 0:
        raise ValueError('grid.deformation_roi_edge_sigma_px cannot be negative')
    roi_weight = (
        gaussian_filter(roi_mask.astype(float), sigma=roi_edge_sigma)
        if roi_edge_sigma > 0
        else roi_mask.astype(float)
    )
    # Permit a low, smoothly decaying influence outside the failure polygon. Only
    # the outer AOI remains a hard boundary because no synthetic data exist there.
    roi_weight *= aoi_mask
    roi_weight /= roi_weight.max()
    return roi_weight


def build_context(config_path, scenario_name=None, output_root=None):
    """Load configuration and derive the simulation grid, masks and paths."""
    with config_path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    scenario_name = scenario_name or config['active_scenario']
    if scenario_name not in config['scenarios']:
        raise ValueError(
            f'Unknown scenario {scenario_name!r}; choose one of {sorted(config["scenarios"])}'
        )
    scenario = config['scenarios'][scenario_name]

    outdir = _resolve_output_dir(
        config, scenario_name, scenario, config_path, output_root
    )

    grid_config = config['grid']
    (
        crs,
        pixel_size,
        polygon_mode,
        nx,
        ny,
        spatial_bounds,
        transform,
        aoi_geometries,
    ) = _build_grid(grid_config, config_path)
    aoi_mask, roi_mask = _build_masks(
        grid_config, config_path, crs, nx, ny, transform, aoi_geometries
    )

    y, x = np.mgrid[0:ny, 0:nx]

    roi_rows, roi_cols = np.where(roi_mask)
    roi_center = np.array([roi_cols.mean(), roi_rows.mean()])
    roi_weight = _build_roi_weight(grid_config, roi_mask, aoi_mask)

    aoi_rows, aoi_cols = np.where(aoi_mask)
    aoi_center = [aoi_cols.mean(), aoi_rows.mean()]
    bowl = np.exp(
        -((x - aoi_center[0]) ** 2 + (y - aoi_center[1]) ** 2) / (2 * SIGMA_BOWL**2)
    )

    simulation = config['simulation']
    observation = config['observation']
    return SimulationContext(
        config=config,
        scenario_name=scenario_name,
        scenario=scenario,
        outdir=outdir,
        inputs_dir=outdir / 'inputs',
        static_dir=outdir / 'static',
        crs=crs,
        pixel_size=pixel_size,
        polygon_mode=polygon_mode,
        nx=nx,
        ny=ny,
        spatial_bounds=spatial_bounds,
        transform=transform,
        aoi_geometries=aoi_geometries,
        aoi_mask=aoi_mask,
        roi_mask=roi_mask,
        roi_weight=roi_weight,
        roi_center=roi_center,
        x=x,
        y=y,
        bowl=bowl,
        years=simulation['duration_days'] / 365,
        revisit_days=simulation['revisit_days'],
        start_date=datetime.fromisoformat(simulation['start_date']).replace(
            tzinfo=timezone.utc
        ),
        incidence_deg=observation['incidence_angle_deg'],
        hydro_amp=config['baseline']['seasonal_amplitude_mm'] / 1000,
        atm_sigma=observation['atmospheric_sigma_mm'] / 1000,
        meas_sigma=observation['measurement_sigma_mm'] / 1000,
        random_seed=simulation['random_seed'],
    )


def create_output_dirs(ctx):
    ctx.outdir.mkdir(exist_ok=True, parents=True)
    for d in INPUT_SUBDIRS:
        (ctx.inputs_dir / d).mkdir(parents=True, exist_ok=True)
    ctx.static_dir.mkdir(parents=True, exist_ok=True)


def generate_coherence(ctx):
    """Return the spatial coherence field and the derived coherent-pixel mask."""
    coherence_config = ctx.config['observation']
    coherence_texture = gaussian_filter(
        np.random.normal(size=(ctx.ny, ctx.nx)),
        sigma=coherence_config['coherence_correlation_length_px'],
    )
    # Standardize over valid AOI pixels before imposing the requested distribution.
    texture_values = coherence_texture[ctx.aoi_mask]
    texture_std = texture_values.std()
    if texture_std == 0:
        raise ValueError('Cannot generate coherence: spatial texture has zero variance')
    coherence_texture = (coherence_texture - texture_values.mean()) / texture_std
    coh_field = np.clip(
        coherence_config['coherence_mean']
        + coherence_config['coherence_spatial_std'] * coherence_texture,
        coherence_config['coherence_min'],
        coherence_config['coherence_max'],
    )
    coh_field[~ctx.aoi_mask] = np.nan
    mask = (coh_field >= coherence_config['coherence_threshold']) & ctx.aoi_mask
    return coh_field, mask


### HELPERS ###
def write_tif(
    ctx,
    path,
    arr,
    acquisition_date=None,
    product_type=None,
    stage=0,
    dtype='float32',
    nodata=None,
    apply_aoi=True,
):
    """Write an aligned single-band synthetic GeoTIFF and optional STAC item.

    Args:
        ctx: Simulation context providing CRS, transform and AOI mask.
        path: Destination raster path.
        arr: Two-dimensional values on the global synthetic grid.
        acquisition_date: Optional acquisition time used for STAC metadata.
        product_type: Simulation product identifier stored in the STAC item.
        stage: Integer failure stage from zero (stable) to three (accelerating).
        dtype: Rasterio-compatible output data type.
        nodata: Optional output nodata value.
    """
    output = np.array(arr, copy=True)
    if apply_aoi and output.shape == ctx.aoi_mask.shape:
        if np.issubdtype(np.dtype(dtype), np.floating):
            output[~ctx.aoi_mask] = np.nan
            nodata = np.nan
        else:
            output[~ctx.aoi_mask] = 255
            nodata = 255
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=ctx.crs,
        transform=ctx.transform,
        nodata=nodata,
        compress='lzw',
    ) as dst:
        dst.write(output.astype(dtype), 1)

    write_stac_item(
        ctx,
        path,
        acquisition_date,
        product_type or path.stem,
        stage,
    )


def write_stac_item(
    ctx,
    tif_path,
    acquisition_date,
    product_type,
    stage,
):
    """
    Create STAC Item JSON next to a GeoTIFF.
    """

    transformer = Transformer.from_crs(
        ctx.crs,
        'EPSG:4326',
        always_xy=True,
    )

    xmin, ymin, xmax, ymax = ctx.spatial_bounds

    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)

    if ctx.aoi_geometries:
        wgs84_geometries = [
            transform_geom(ctx.crs, 'EPSG:4326', item.__geo_interface__)
            for item in ctx.aoi_geometries
        ]
        geometry = (
            wgs84_geometries[0]
            if len(wgs84_geometries) == 1
            else {
                'type': 'GeometryCollection',
                'geometries': wgs84_geometries,
            }
        )
    else:
        geometry = {
            'type': 'Polygon',
            'coordinates': [
                [
                    [lon_min, lat_min],
                    [lon_min, lat_max],
                    [lon_max, lat_max],
                    [lon_max, lat_min],
                    [lon_min, lat_min],
                ]
            ],
        }

    item = {
        'type': 'Feature',
        'stac_version': '1.0.0',
        'id': tif_path.stem,
        'bbox': [
            lon_min,
            lat_min,
            lon_max,
            lat_max,
        ],
        'geometry': geometry,
        'properties': {
            'platform': 'synthetic-tsf',
            'constellation': 'synthetic',
            'instruments': ['simulation'],
            'processing:level': 'L2',
            'proj:epsg': rasterio.crs.CRS.from_user_input(ctx.crs).to_epsg(),
            'simulation:type': product_type,
            'simulation:failure_stage': STAGE_NAMES[stage],
            'simulation:unit': 'm',
        },
        'assets': {
            'data': {
                'href': './' + tif_path.name,
                'type': 'image/tiff; application=geotiff',
                'roles': ['data'],
            }
        },
        'collection': 'gaia-tsf-synthetic',
    }
    if acquisition_date is not None:
        item['properties']['datetime'] = acquisition_date.strftime('%Y-%m-%dT00:00:00Z')

    json_file = tif_path.with_suffix('.json')

    with open(json_file, 'w') as f:
        json.dump(
            item,
            f,
            indent=4,
        )


def generate_tsf_dem(ctx):
    """Generate a deterministic terrain surface containing the synthetic TSF.

    The surface combines regional relief, a shallow impoundment depression,
    and a raised perimeter embankment aligned with the TSF mask boundary.
    A dedicated random generator keeps DEM creation from changing the temporal
    simulation's seeded random sequence.
    """
    x, y = ctx.x, ctx.y
    x_distance_m = (x - (ctx.nx - 1) / 2) * ctx.pixel_size
    y_distance_m = (y - (ctx.ny - 1) / 2) * ctx.pixel_size
    regional_terrain = (
        DEM_BASE_ELEVATION
        + DEM_EAST_SLOPE * x_distance_m
        + DEM_SOUTH_SLOPE * y_distance_m
    )

    distance_pixels = np.hypot(x - 50, y - 50)
    impoundment = -DEM_BASIN_DEPTH * ctx.bowl
    embankment = DEM_EMBANKMENT_HEIGHT * np.exp(
        -((distance_pixels - DEM_EMBANKMENT_RADIUS) ** 2)
        / (2 * DEM_EMBANKMENT_WIDTH**2)
    )

    dem_rng = np.random.default_rng(ctx.random_seed + 1)
    roughness = gaussian_filter(
        dem_rng.normal(0.0, DEM_ROUGHNESS_STD, (ctx.ny, ctx.nx)),
        sigma=5,
    )
    return (regional_terrain + impoundment + embankment + roughness).astype(np.float32)


### SYNTHETIC METEODATA ###
def generate_meteodata(ctx):
    """Generate raw daily weather observations for downstream DAG processing."""
    simulation_days = np.arange(0, ctx.years * 365)
    dates = pd.to_datetime(ctx.start_date) + pd.to_timedelta(simulation_days, unit='D')

    # A simple wet/dry process produces dry days and right-skewed rain amounts.
    wet_days = np.random.random(len(simulation_days)) < 0.38
    precipitation = np.where(
        wet_days,
        np.random.gamma(shape=1.5, scale=4.0, size=len(simulation_days)),
        0.0,
    )
    for event_day, event_precipitation in TRIGGER_EVENTS.items():
        if event_day < len(precipitation):
            precipitation[event_day] = event_precipitation

    # Seasonal temperature cycle, daily synoptic variability, and diurnal range.
    seasonal_temperature = 8.0 + 11.0 * np.sin(
        2 * np.pi * (simulation_days - 105) / 365
    )
    temperature_mean = seasonal_temperature + np.random.normal(
        0, 3.0, len(simulation_days)
    )
    diurnal_range = np.maximum(2.0, np.random.normal(8.0, 2.0, len(simulation_days)))
    temperature_min = temperature_mean - diurnal_range / 2
    temperature_max = temperature_mean + diurnal_range / 2

    meteo = pd.DataFrame(
        {
            'date': dates,
            'precipitation': precipitation,
            'temperature_mean': temperature_mean,
            'temperature_min': temperature_min,
            'temperature_max': temperature_max,
        }
    )

    meteo['date'] = meteo['date'].dt.strftime('%Y%m%d')
    numeric_columns = meteo.select_dtypes(include=[np.number]).columns
    meteo[numeric_columns] = meteo[numeric_columns].round(2)
    return meteo


### DATA SIMULATION ###
def scenario_state(ctx, t):
    """Return displacement, probability, stage and trigger fields in millimetres."""
    p, kind = ctx.scenario, ctx.scenario['archetype']
    x, y = ctx.x, ctx.y
    probability_threshold = ctx.config['classification'][
        'hotspot_probability_threshold'
    ]
    zero = np.zeros_like(ctx.bowl)
    if kind == 'no_failure':
        fluctuation = p['fluctuation_amplitude_mm'] * np.sin(
            2 * np.pi * t / p['primary_period_days']
        ) + p['secondary_amplitude_mm'] * np.sin(
            2 * np.pi * t / p['secondary_period_days'] + np.pi / 3
        )
        settlement = p['local_settlement_mm_per_year'] * t / 365.0
        # Negative control: apply stable motion uniformly. In particular, never
        # expose the future failure ROI through displacement or truth labels.
        stable_field = np.full_like(ctx.bowl, fluctuation + settlement)
        return stable_field, zero, zero.astype('uint8'), zero
    if kind == 'gradual_acceleration':
        if ctx.polygon_mode:
            h = ctx.roi_weight
        else:
            h = np.exp(
                -0.5
                * (
                    ((x - p['center_px'][0]) / p['sigma_px'][0]) ** 2
                    + ((y - p['center_px'][1]) / p['sigma_px'][1]) ** 2
                )
            )
        if t < p['onset_day']:
            d, stage = 0, 0
        elif t < p['velocity_increase_day']:
            d, stage = p['slow_velocity_mm_per_day'] * (t - p['onset_day']), 1
        elif t < p['acceleration_day']:
            dt = t - p['velocity_increase_day']
            d0 = p['slow_velocity_mm_per_day'] * (
                p['velocity_increase_day'] - p['onset_day']
            )
            d = (
                d0
                + p['slow_velocity_mm_per_day'] * dt
                + 0.5 * p['linear_acceleration_mm_per_day2'] * dt**2
            )
            stage = 2
        else:
            dt = t - p['acceleration_day']
            span = p['acceleration_day'] - p['velocity_increase_day']
            v = (
                p['slow_velocity_mm_per_day']
                + p['linear_acceleration_mm_per_day2'] * span
            )
            d0 = (
                p['slow_velocity_mm_per_day']
                * (p['velocity_increase_day'] - p['onset_day'])
                + p['slow_velocity_mm_per_day'] * span
                + 0.5 * p['linear_acceleration_mm_per_day2'] * span**2
            )
            d = (
                d0
                + v
                * (np.exp(p['exponential_growth_rate_per_day'] * dt) - 1)
                / p['exponential_growth_rate_per_day']
            )
            stage = 3
        d = max(d, p['maximum_displacement_mm'])
        stage = 4 if t >= p['failure_day'] else stage
        prob = np.clip(abs(d / p['maximum_displacement_mm']), 0, 1) * h
        return d * h, prob, np.where(prob > probability_threshold, stage, 0), zero
    if kind == 'spatial_propagation':
        if t < p['initiation_day']:
            return zero, zero, zero.astype('uint8'), zero
        initial_center = (
            ctx.roi_center if ctx.polygon_mode else np.array(p['initial_center_px'])
        )
        dt = t - p['initiation_day']
        c = initial_center + np.array(p['center_velocity_px_per_day']) * dt
        s = np.minimum(
            np.array(p['maximum_sigma_px']),
            np.array(p['initial_sigma_px'])
            + np.array(p['propagation_rate_px_per_day']) * dt,
        )
        h = (
            np.exp(-0.5 * (((x - c[0]) / s[0]) ** 2 + ((y - c[1]) / s[1]) ** 2))
            * ctx.roi_weight
        )
        d = (
            p['core_velocity_mm_per_day']
            * (dt + 0.5 * p['velocity_growth_per_day'] * dt**2)
            * h
        )
        prob = h * min(1, 0.15 + dt / (p['failure_day'] - p['initiation_day']))
        frac = np.mean(prob > probability_threshold)
        stage = (
            1
            if frac < p['failure_area_fraction'] * 0.25
            else 2
            if frac < p['failure_area_fraction'] * 0.7
            else 3
        )
        if t >= p['failure_day'] or frac >= p['failure_area_fraction']:
            stage = 4
        return d, prob, np.where(prob > probability_threshold, stage, 0), zero
    base = p['baseline_velocity_mm_per_day'] * t * ctx.bowl
    trigger = zero.copy()
    active = False
    for event in p['events']:
        age = t - event['day'] - event.get('response_delay_days', 0)
        if age < 0:
            continue
        event_center = (
            ctx.roi_center if ctx.polygon_mode else np.array(event['center_px'])
        )
        active = True
        h = (
            np.exp(
                -0.5
                * (
                    ((x - event_center[0]) / event['sigma_px'][0]) ** 2
                    + ((y - event_center[1]) / event['sigma_px'][1]) ** 2
                )
            )
            * ctx.roi_weight
        )
        if event['type'] == 'rainfall':
            response = (
                event['displacement_scale_mm']
                * (1 - np.exp(-age / event['response_rise_days']))
                * np.exp(-age / event['decay_days'])
            )
        else:
            response = event['instantaneous_offset_mm']
        trigger += response * h
    if active:
        age = t - min(e['day'] for e in p['events'])
        trigger += (
            p['post_trigger_velocity_mm_per_day'] * age
            + 0.5 * p['post_trigger_acceleration_mm_per_day2'] * age**2
        ) * ctx.roi_weight
    prob = np.clip(abs(trigger) / 50, 0, 1)
    stage = 3 if active else 0
    return (
        base + trigger,
        prob,
        np.where(prob > probability_threshold, stage, 0),
        trigger,
    )


def simulate_time_series(ctx, times, coherence_mask):
    """Simulate every acquisition and write its rasters and STAC items."""
    previous_disp = previous_velocity = None
    probability_threshold = ctx.config['classification'][
        'hotspot_probability_threshold'
    ]

    for t in times:
        acquisition_date = ctx.start_date + timedelta(days=int(t))

        date_str = acquisition_date.strftime('%Y%m%d')

        # ``no_failure`` defines its complete physical motion in scenario_state().
        # Applying the shared settlement and seasonality as well would introduce an
        # unintended trend and duplicate its configured periodic fluctuation.
        if ctx.scenario['archetype'] == 'no_failure':
            background = np.zeros_like(ctx.bowl)
            seasonal = np.zeros_like(ctx.bowl)
        else:
            background = (
                ctx.config['baseline']['final_settlement_mm']
                / 1000
                * (t / (ctx.years * 365))
                * ctx.bowl
            )
            seasonal = ctx.hydro_amp * np.sin(2 * np.pi * t / 365) * ctx.bowl

        scenario_disp_mm, hotspot, stage, trigger_mm = scenario_state(ctx, t)
        total_vertical = background + seasonal + scenario_disp_mm / 1000
        velocity = (
            np.zeros_like(ctx.bowl)
            if previous_disp is None
            else (scenario_disp_mm - previous_disp) / ctx.revisit_days / 1000
        )
        acceleration = (
            np.zeros_like(ctx.bowl)
            if previous_velocity is None
            else (velocity - previous_velocity) / ctx.revisit_days
        )
        previous_disp, previous_velocity = scenario_disp_mm.copy(), velocity.copy()

        true_los = total_vertical * np.cos(np.deg2rad(ctx.incidence_deg))

        atmosphere = gaussian_filter(
            np.random.normal(0, ctx.atm_sigma, (ctx.ny, ctx.nx)), sigma=10
        )

        measurement = np.random.normal(0, ctx.meas_sigma, (ctx.ny, ctx.nx))

        los_obs = true_los + atmosphere + measurement

        los_obs_masked = los_obs.copy()
        los_obs_masked[~coherence_mask] = np.nan

        label = (hotspot > probability_threshold).astype(np.uint8)

        stage = stage.astype(np.uint8)
        max_stage = int(stage.max())

        products = [
            ('los', f'tsf_los_{date_str}', los_obs_masked, 'los', 'float32', np.nan),
            ('true_los', f'true_los_{date_str}', true_los, 'true_los', 'float32', None),
            (
                'hotspot_prob',
                f'hotspot_prob_{date_str}',
                hotspot,
                'hotspot_probability',
                'float32',
                None,
            ),
            (
                'labels',
                f'hotspot_label_{date_str}',
                label,
                'binary_label',
                'uint8',
                None,
            ),
            (
                'failure_stage',
                f'failure_stage_{date_str}',
                stage,
                'failure_stage',
                'uint8',
                None,
            ),
            (
                'atmosphere',
                f'atmosphere_{date_str}',
                atmosphere,
                'atmosphere',
                'float32',
                None,
            ),
            (
                'rainfall',
                f'rainfall_trigger_{date_str}',
                trigger_mm / 1000,
                'rainfall_trigger',
                'float32',
                None,
            ),
            ('velocity', f'velocity_{date_str}', velocity, 'velocity', 'float32', None),
            (
                'acceleration',
                f'acceleration_{date_str}',
                acceleration,
                'acceleration',
                'float32',
                None,
            ),
            (
                'trigger_response',
                f'trigger_response_{date_str}',
                trigger_mm / 1000,
                'trigger_response',
                'float32',
                None,
            ),
        ]
        for subdir, name, arr, product_type, dtype, nodata in products:
            write_tif(
                ctx,
                ctx.inputs_dir / subdir / f'{name}.tif',
                arr,
                acquisition_date=acquisition_date,
                product_type=product_type,
                stage=max_stage,
                dtype=dtype,
                nodata=nodata,
            )


### EXPORTS ###
def write_meteodata(ctx, meteo_data, times):
    meteo_data.to_csv(ctx.inputs_dir / 'meteodata.csv', index=False)
    # Retain the acquisition-date precipitation export for existing consumers.
    acquisition_date_strings = {
        (ctx.start_date + timedelta(days=int(day))).strftime('%Y%m%d') for day in times
    }
    meteo_data.loc[
        meteo_data['date'].isin(acquisition_date_strings),
        ['date', 'precipitation'],
    ].rename(columns={'precipitation': 'precipitation_mm'}).to_csv(
        ctx.inputs_dir / 'meteo_precipitation.csv',
        index=False,
    )


def write_static_products(ctx, coh_field, coherence_mask):
    write_tif(
        ctx,
        ctx.inputs_dir / 'aux' / 'coherence.tif',
        coh_field,
        dtype='float32',
    )
    write_tif(
        ctx,
        ctx.inputs_dir / 'aux' / 'coherence_mask.tif',
        coherence_mask.astype(np.uint8),
        dtype='uint8',
        apply_aoi=False,
    )

    write_tif(
        ctx,
        ctx.static_dir / 'aoi_mask.tif',
        ctx.aoi_mask.astype(np.uint8),
        dtype='uint8',
        apply_aoi=False,
    )
    write_tif(
        ctx,
        ctx.static_dir / 'roi_failure_mask.tif',
        ctx.roi_mask.astype(np.uint8),
        dtype='uint8',
        apply_aoi=False,
    )
    write_tif(
        ctx, ctx.static_dir / 'roi_failure_weight.tif', ctx.roi_weight, dtype='float32'
    )
    write_tif(
        ctx,
        ctx.static_dir / 'tsf_mask.tif',
        ctx.aoi_mask.astype(np.uint8),
        dtype='uint8',
        apply_aoi=False,
    )

    tsf_dem = generate_tsf_dem(ctx)
    dem_path = ctx.static_dir / 'tsf_dem.tif'
    write_tif(ctx, dem_path, tsf_dem, dtype='float32', nodata=np.nan)
    with rasterio.open(dem_path, 'r+') as dem_dataset:
        dem_dataset.set_band_description(1, 'TSF elevation')
        dem_dataset.update_tags(1, units='m')


def build_failure_timeline(ctx):
    scenario, start_date = ctx.scenario, ctx.start_date

    def day_to_date(day):
        return (start_date + timedelta(days=day)).strftime('%Y-%m-%d')

    archetype = scenario['archetype']
    if archetype == 'no_failure':
        return [{'event': 'stable_period', 'date': start_date.strftime('%Y-%m-%d')}]
    if archetype == 'gradual_acceleration':
        return [
            {'event': event, 'date': day_to_date(scenario[key])}
            for event, key in [
                ('slow_phase', 'onset_day'),
                ('developing_phase', 'velocity_increase_day'),
                ('accelerating_phase', 'acceleration_day'),
                ('failure', 'failure_day'),
            ]
        ]
    if archetype == 'spatial_propagation':
        return [
            {
                'event': 'localized_instability',
                'date': day_to_date(scenario['initiation_day']),
            },
            {'event': 'failure', 'date': day_to_date(scenario['failure_day'])},
        ]
    return [
        {'event': event['type'], 'date': day_to_date(event['day'])}
        for event in scenario.get('events', [])
    ]


def write_metadata(ctx, coherence_mask):
    coherence_config = ctx.config['observation']
    pd.DataFrame(
        {
            'parameter': [
                'crs',
                'pixel_size',
                'years',
                'revisit_days',
                'failure_start_day',
                'final_hotspot_amplitude_m',
                'atmosphere_sigma_m',
                'measurement_sigma_m',
                'coherence_mean',
                'coherence_spatial_std',
                'coherence_threshold',
                'coherent_aoi_fraction',
                'dem_base_elevation_m',
                'dem_basin_depth_m',
                'dem_embankment_height_m',
            ],
            'value': [
                ctx.crs,
                ctx.pixel_size,
                ctx.years,
                ctx.revisit_days,
                FAILURE_START_DAY,
                A_MAX,
                ctx.atm_sigma,
                ctx.meas_sigma,
                coherence_config['coherence_mean'],
                coherence_config['coherence_spatial_std'],
                coherence_config['coherence_threshold'],
                float(coherence_mask.sum() / ctx.aoi_mask.sum()),
                DEM_BASE_ELEVATION,
                DEM_BASIN_DEPTH,
                DEM_EMBANKMENT_HEIGHT,
            ],
        }
    ).to_csv(ctx.inputs_dir / 'metadata.csv', index=False)


def write_scenario_config(ctx):
    # Preserve the exact scenario definition alongside every generated data set.
    with open(ctx.outdir / 'scenario_config.yaml', 'w', encoding='utf-8') as stream:
        yaml.safe_dump(
            {**ctx.config, 'active_scenario': ctx.scenario_name},
            stream,
            sort_keys=False,
        )


### MAIN ###
def run(config_path=CONFIG_FILE, scenario_name=None, output_root=None):
    """Generate the complete synthetic benchmark and return its output directory."""
    ctx = build_context(config_path, scenario_name, output_root)
    create_output_dirs(ctx)

    # The global NumPy random sequence is consumed in a fixed order (coherence,
    # meteodata, per-acquisition noise) so outputs are reproducible per seed.
    np.random.seed(ctx.random_seed)
    coh_field, coherence_mask = generate_coherence(ctx)
    meteo_data = generate_meteodata(ctx)

    times = np.arange(0, ctx.config['simulation']['duration_days'], ctx.revisit_days)
    simulate_time_series(ctx, times, coherence_mask)

    write_meteodata(ctx, meteo_data, times)
    write_static_products(ctx, coh_field, coherence_mask)
    pd.DataFrame(build_failure_timeline(ctx)).to_csv(
        ctx.inputs_dir / 'failure_timeline.csv', index=False
    )
    write_metadata(ctx, coherence_mask)
    write_scenario_config(ctx)
    return ctx.outdir


def main(argv=None):
    args = parse_args(argv)
    outdir = run(args.config, args.scenario, args.output_root)
    print('Finished.')
    print(f'Output: {outdir.resolve()}')


if __name__ == '__main__':
    main()
