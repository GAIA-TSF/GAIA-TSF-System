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
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

### CONFIGURATION ###

CONFIG_FILE = Path(__file__).with_name('synthetic_tsf_config.yaml')
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, default=CONFIG_FILE)
parser.add_argument('--scenario')
parser.add_argument('--output-root', type=Path)
args = parser.parse_args()
with args.config.open(encoding='utf-8') as stream:
    CONFIG = yaml.safe_load(stream)
SCENARIO_NAME = args.scenario or CONFIG['active_scenario']
if SCENARIO_NAME not in CONFIG['scenarios']:
    raise ValueError(f"Unknown scenario {SCENARIO_NAME!r}; choose one of {sorted(CONFIG['scenarios'])}")
SCENARIO = CONFIG['scenarios'][SCENARIO_NAME]
root = args.output_root or Path(CONFIG['simulation']['output_root'])
if not root.is_absolute():
    root = (args.config.resolve().parent / root).resolve()
OUTDIR = root / SCENARIO['output_directory']
OUTDIR.mkdir(exist_ok=True, parents=True)
INPUTS_DIR = OUTDIR / 'inputs'
STATIC_DIR = OUTDIR / 'static'

CRS = CONFIG['grid']['crs']
PIXEL_SIZE = CONFIG['grid']['pixel_size_m']

NX = CONFIG['grid']['width_px']
NY = CONFIG['grid']['height_px']

XMIN = CONFIG['grid']['xmin']
YMAX = CONFIG['grid']['ymax']

YEARS = CONFIG['simulation']['duration_days'] / 365
REVISIT_DAYS = CONFIG['simulation']['revisit_days']

START_DATE = datetime.fromisoformat(CONFIG['simulation']['start_date']).replace(tzinfo=timezone.utc)

INCIDENCE_DEG = CONFIG['observation']['incidence_angle_deg']
FAILURE_START_DAY = 730

A_MAX = -0.10
HYDRO_AMP = CONFIG['baseline']['seasonal_amplitude_mm'] / 1000

ATM_SIGMA = CONFIG['observation']['atmospheric_sigma_mm'] / 1000
MEAS_SIGMA = CONFIG['observation']['measurement_sigma_mm'] / 1000

RANDOM_SEED = CONFIG['simulation']['random_seed']

# Static TSF terrain model (metres).
DEM_BASE_ELEVATION = 450.0
DEM_EAST_SLOPE = 0.012
DEM_SOUTH_SLOPE = 0.006
DEM_BASIN_DEPTH = 8.0
DEM_EMBANKMENT_HEIGHT = 15.0
DEM_EMBANKMENT_RADIUS = 39.0
DEM_EMBANKMENT_WIDTH = 3.0
DEM_ROUGHNESS_STD = 0.75

np.random.seed(RANDOM_SEED)

transform = from_origin(XMIN, YMAX, PIXEL_SIZE, PIXEL_SIZE)

times = np.arange(0, CONFIG['simulation']['duration_days'], REVISIT_DAYS)

for d in [
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
]:
    (INPUTS_DIR / d).mkdir(parents=True, exist_ok=True)

STATIC_DIR.mkdir(parents=True, exist_ok=True)

Y, X = np.mgrid[0:NY, 0:NX]

sigma_bowl = 18

bowl = np.exp(-((X - 50) ** 2 + (Y - 50) ** 2) / (2 * sigma_bowl**2))

coh_field = gaussian_filter(np.random.rand(NY, NX), sigma=8)

mask = coh_field > 0.35

rain_events = [790, 860, 950, 1030]


### HELPERS ###
def write_tif(
    path,
    arr,
    acquisition_date=None,
    product_type=None,
    stage=0,
    dtype='float32',
    nodata=None,
):
    """Write an aligned single-band synthetic GeoTIFF and optional STAC item.

    Args:
        path: Destination raster path.
        arr: Two-dimensional values on the global synthetic grid.
        acquisition_date: Optional acquisition time used for STAC metadata.
        product_type: Simulation product identifier stored in the STAC item.
        stage: Integer failure stage from zero (stable) to three (accelerating).
        dtype: Rasterio-compatible output data type.
        nodata: Optional output nodata value.
    """
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=CRS,
        transform=transform,
        nodata=nodata,
        compress='lzw',
    ) as dst:
        dst.write(arr.astype(dtype), 1)

    if acquisition_date is not None:
        write_stac_item(
            path,
            acquisition_date,
            product_type,
            stage,
        )


def write_stac_item(
    tif_path,
    acquisition_date,
    product_type,
    stage,
):
    """
    Create STAC Item JSON next to a GeoTIFF.
    """

    transformer = Transformer.from_crs(
        CRS,
        'EPSG:4326',
        always_xy=True,
    )

    xmin = XMIN
    xmax = XMIN + NX * PIXEL_SIZE

    ymax = YMAX
    ymin = YMAX - NY * PIXEL_SIZE

    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)

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

    stage_name = {
        0: 'stable',
        1: 'latent',
        2: 'developing',
        3: 'accelerating',
        4: 'failure',
    }[stage]

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
            'datetime': acquisition_date.strftime('%Y-%m-%dT00:00:00Z'),
            'platform': 'synthetic-tsf',
            'constellation': 'synthetic',
            'instruments': ['simulation'],
            'processing:level': 'L2',
            'proj:epsg': 32633,
            'simulation:type': product_type,
            'simulation:failure_stage': stage_name,
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

    json_file = tif_path.with_suffix('.json')

    with open(json_file, 'w') as f:
        json.dump(
            item,
            f,
            indent=4,
        )


def generate_tsf_dem():
    """Generate a deterministic terrain surface containing the synthetic TSF.

    The surface combines regional relief, a shallow impoundment depression,
    and a raised perimeter embankment aligned with the TSF mask boundary.
    A dedicated random generator keeps DEM creation from changing the temporal
    simulation's seeded random sequence.
    """
    x_distance_m = (X - (NX - 1) / 2) * PIXEL_SIZE
    y_distance_m = (Y - (NY - 1) / 2) * PIXEL_SIZE
    regional_terrain = (
        DEM_BASE_ELEVATION
        + DEM_EAST_SLOPE * x_distance_m
        + DEM_SOUTH_SLOPE * y_distance_m
    )

    distance_pixels = np.hypot(X - 50, Y - 50)
    impoundment = -DEM_BASIN_DEPTH * bowl
    embankment = DEM_EMBANKMENT_HEIGHT * np.exp(
        -((distance_pixels - DEM_EMBANKMENT_RADIUS) ** 2)
        / (2 * DEM_EMBANKMENT_WIDTH**2)
    )

    dem_rng = np.random.default_rng(RANDOM_SEED + 1)
    roughness = gaussian_filter(
        dem_rng.normal(0.0, DEM_ROUGHNESS_STD, (NY, NX)),
        sigma=5,
    )
    return (regional_terrain + impoundment + embankment + roughness).astype(
        np.float32
    )


### SYNTHETIC METEODATA ###
# Rainfall trigger events that contribute to instability. Weather is exported
# daily so the DAG can derive rolling features over complete lookback periods.
trigger_events = {790: 45, 860: 62, 950: 78, 1030: 55}


def generate_meteodata():
    """Generate raw daily weather observations for downstream DAG processing."""
    simulation_days = np.arange(0, YEARS * 365)
    dates = pd.to_datetime(START_DATE) + pd.to_timedelta(simulation_days, unit='D')

    # A simple wet/dry process produces dry days and right-skewed rain amounts.
    wet_days = np.random.random(len(simulation_days)) < 0.38
    precipitation = np.where(
        wet_days,
        np.random.gamma(shape=1.5, scale=4.0, size=len(simulation_days)),
        0.0,
    )
    for event_day, event_precipitation in trigger_events.items():
        if event_day < len(precipitation):
            precipitation[event_day] = event_precipitation

    # Seasonal temperature cycle, daily synoptic variability, and diurnal range.
    seasonal_temperature = 8.0 + 11.0 * np.sin(
        2 * np.pi * (simulation_days - 105) / 365
    )
    temperature_mean = seasonal_temperature + np.random.normal(
        0, 3.0, len(simulation_days)
    )
    diurnal_range = np.maximum(
        2.0, np.random.normal(8.0, 2.0, len(simulation_days))
    )
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


meteo_data = generate_meteodata()

### DATA SIMULATION ###
def scenario_state(t):
    """Return displacement, probability, stage and trigger fields in millimetres."""
    p, kind = SCENARIO, SCENARIO['archetype']
    zero = np.zeros_like(bowl)
    if kind == 'no_failure':
        center, sigma = p['fluctuation_center_px'], p['fluctuation_sigma_px']
        stable_zone = np.exp(-0.5 * (((X-center[0])/sigma[0])**2 + ((Y-center[1])/sigma[1])**2))
        fluctuation = (
            p['fluctuation_amplitude_mm'] * np.sin(2*np.pi*t/p['primary_period_days'])
            + p['secondary_amplitude_mm'] * np.sin(2*np.pi*t/p['secondary_period_days'] + np.pi/3)
        )
        settlement = p['local_settlement_mm_per_year'] * t / 365.0
        # Negative control: motion may fluctuate, but all truth labels stay stable.
        return (fluctuation + settlement) * stable_zone, zero, zero.astype('uint8'), zero
    if kind == 'gradual_acceleration':
        h = np.exp(-0.5 * (((X-p['center_px'][0])/p['sigma_px'][0])**2 + ((Y-p['center_px'][1])/p['sigma_px'][1])**2))
        if t < p['onset_day']:
            d, stage = 0, 0
        elif t < p['velocity_increase_day']:
            d, stage = p['slow_velocity_mm_per_day']*(t-p['onset_day']), 1
        elif t < p['acceleration_day']:
            dt=t-p['velocity_increase_day']; d0=p['slow_velocity_mm_per_day']*(p['velocity_increase_day']-p['onset_day'])
            d=d0+p['slow_velocity_mm_per_day']*dt+.5*p['linear_acceleration_mm_per_day2']*dt**2; stage=2
        else:
            dt=t-p['acceleration_day']; span=p['acceleration_day']-p['velocity_increase_day']; v=p['slow_velocity_mm_per_day']+p['linear_acceleration_mm_per_day2']*span
            d0=p['slow_velocity_mm_per_day']*(p['velocity_increase_day']-p['onset_day'])+p['slow_velocity_mm_per_day']*span+.5*p['linear_acceleration_mm_per_day2']*span**2
            d=d0+v*(np.exp(p['exponential_growth_rate_per_day']*dt)-1)/p['exponential_growth_rate_per_day']; stage=3
        d=max(d,p['maximum_displacement_mm']); stage=4 if t>=p['failure_day'] else stage
        prob=np.clip(abs(d/p['maximum_displacement_mm']),0,1)*h
        return d*h, prob, np.where(prob>.2,stage,0), zero
    if kind == 'spatial_propagation':
        if t < p['initiation_day']: return zero, zero, zero.astype('uint8'), zero
        dt=t-p['initiation_day']; c=np.array(p['initial_center_px'])+np.array(p['center_velocity_px_per_day'])*dt
        s=np.minimum(np.array(p['maximum_sigma_px']),np.array(p['initial_sigma_px'])+np.array(p['propagation_rate_px_per_day'])*dt)
        h=np.exp(-.5*(((X-c[0])/s[0])**2+((Y-c[1])/s[1])**2)); d=p['core_velocity_mm_per_day']*(dt+.5*p['velocity_growth_per_day']*dt**2)*h
        prob=h*min(1,.15+dt/(p['failure_day']-p['initiation_day'])); frac=np.mean(prob>.2); stage=1 if frac<p['failure_area_fraction']*.25 else 2 if frac<p['failure_area_fraction']*.7 else 3
        if t>=p['failure_day'] or frac>=p['failure_area_fraction']: stage=4
        return d,prob,np.where(prob>.2,stage,0),zero
    base=p['baseline_velocity_mm_per_day']*t*bowl; trigger=zero.copy(); active=False
    for event in p['events']:
        age=t-event['day']-event.get('response_delay_days',0)
        if age<0: continue
        active=True; h=np.exp(-.5*(((X-event['center_px'][0])/event['sigma_px'][0])**2+((Y-event['center_px'][1])/event['sigma_px'][1])**2))
        if event['type']=='rainfall': response=event['displacement_scale_mm']*(1-np.exp(-age/event['response_rise_days']))*np.exp(-age/event['decay_days'])
        else: response=event['instantaneous_offset_mm']
        trigger += response*h
    if active:
        age=t-min(e['day'] for e in p['events']); trigger+=(p['post_trigger_velocity_mm_per_day']*age+.5*p['post_trigger_acceleration_mm_per_day2']*age**2)*bowl
    prob=np.clip(abs(trigger)/50,0,1); stage=3 if active else 0
    return base+trigger,prob,np.where(prob>.2,stage,0),trigger

previous_disp = previous_velocity = None

for t in times:
    acquisition_date = START_DATE + timedelta(days=int(t))

    date_str = acquisition_date.strftime('%Y%m%d')

    background = CONFIG['baseline']['final_settlement_mm'] / 1000 * (t / (YEARS * 365)) * bowl

    seasonal = HYDRO_AMP * np.sin(2 * np.pi * t / 365) * bowl

    scenario_disp_mm, hotspot, stage, trigger_mm = scenario_state(t)
    total_vertical = background + seasonal + scenario_disp_mm / 1000
    velocity = np.zeros_like(bowl) if previous_disp is None else (scenario_disp_mm-previous_disp)/REVISIT_DAYS/1000
    acceleration = np.zeros_like(bowl) if previous_velocity is None else (velocity-previous_velocity)/REVISIT_DAYS
    previous_disp, previous_velocity = scenario_disp_mm.copy(), velocity.copy()

    true_los = total_vertical * np.cos(np.deg2rad(INCIDENCE_DEG))

    atmosphere = gaussian_filter(np.random.normal(0, ATM_SIGMA, (NY, NX)), sigma=10)

    measurement = np.random.normal(0, MEAS_SIGMA, (NY, NX))

    los_obs = true_los + atmosphere + measurement

    los_obs_masked = los_obs.copy()
    los_obs_masked[~mask] = np.nan

    label = (hotspot > CONFIG['classification']['hotspot_probability_threshold']).astype(np.uint8)

    stage = stage.astype(np.uint8)

    # write_tif(INPUTS_DIR / 'los' / f'tsf_los_{date_str}.tif', los_obs_masked, nodata=np.nan)
    write_tif(
        INPUTS_DIR / 'los' / f'tsf_los_{date_str}.tif',
        los_obs_masked,
        acquisition_date=acquisition_date,
        product_type='los',
        stage=int(stage.max()),
        nodata=np.nan,
    )

    # write_tif(INPUTS_DIR / 'true_los' / f'true_los_{date_str}.tif', true_los)
    write_tif(
        INPUTS_DIR / 'true_los' / f'true_los_{date_str}.tif',
        true_los,
        acquisition_date=acquisition_date,
        product_type='true_los',
        stage=int(stage.max()),
    )

    # write_tif(INPUTS_DIR / 'hotspot_prob' / f'hotspot_prob_{date_str}.tif', hotspot)
    write_tif(
        INPUTS_DIR / 'hotspot_prob' / f'hotspot_prob_{date_str}.tif',
        hotspot,
        acquisition_date=acquisition_date,
        product_type='hotspot_probability',
        stage=int(stage.max()),
    )

    # write_tif(INPUTS_DIR / 'labels' / f'hotspot_label_{date_str}.tif', label, dtype='uint8')
    write_tif(
        INPUTS_DIR / 'labels' / f'hotspot_label_{date_str}.tif',
        label,
        acquisition_date=acquisition_date,
        product_type='binary_label',
        stage=int(stage.max()),
        dtype='uint8',
    )

    # write_tif(
    #     INPUTS_DIR / 'failure_stage' / f'failure_stage_{date_str}.tif', stage, dtype='uint8'
    # )
    write_tif(
        INPUTS_DIR / 'failure_stage' / f'failure_stage_{date_str}.tif',
        stage,
        acquisition_date=acquisition_date,
        product_type='failure_stage',
        stage=int(stage.max()),
        dtype='uint8',
    )

    # write_tif(INPUTS_DIR / 'atmosphere' / f'atmosphere_{date_str}.tif', atmosphere)
    write_tif(
        INPUTS_DIR / 'atmosphere' / f'atmosphere_{date_str}.tif',
        atmosphere,
        acquisition_date=acquisition_date,
        product_type='atmosphere',
        stage=int(stage.max()),
    )

    # write_tif(INPUTS_DIR / 'rainfall' / f'rainfall_trigger_{date_str}.tif', rainfall)
    write_tif(
        INPUTS_DIR / 'rainfall' / f'rainfall_trigger_{date_str}.tif',
        trigger_mm / 1000,
        acquisition_date=acquisition_date,
        product_type='rainfall_trigger',
        stage=int(stage.max()),
    )
    write_tif(INPUTS_DIR / 'velocity' / f'velocity_{date_str}.tif', velocity,
              acquisition_date, 'velocity', int(stage.max()))
    write_tif(INPUTS_DIR / 'acceleration' / f'acceleration_{date_str}.tif', acceleration,
              acquisition_date, 'acceleration', int(stage.max()))
    write_tif(INPUTS_DIR / 'trigger_response' / f'trigger_response_{date_str}.tif', trigger_mm / 1000,
              acquisition_date, 'trigger_response', int(stage.max()))


### METEODATA CSV ###
meteo_data.to_csv(INPUTS_DIR / 'meteodata.csv', index=False)
# Retain the acquisition-date precipitation export for existing consumers.
acquisition_date_strings = {
    (START_DATE + timedelta(days=int(day))).strftime('%Y%m%d') for day in times
}
meteo_data.loc[
    meteo_data['date'].isin(acquisition_date_strings),
    ['date', 'precipitation'],
].rename(columns={'precipitation': 'precipitation_mm'}).to_csv(
    INPUTS_DIR / 'meteo_precipitation.csv',
    index=False,
)

### AUXILIARY EXPORTS ###
write_tif(
    INPUTS_DIR / 'aux' / 'coherence_mask.tif',
    mask.astype(np.uint8),
    dtype='uint8',
)

write_tif(STATIC_DIR / 'tsf_mask.tif', (bowl > 0.1).astype(np.uint8), dtype='uint8')

tsf_dem = generate_tsf_dem()
write_tif(STATIC_DIR / 'tsf_dem.tif', tsf_dem, dtype='float32', nodata=np.nan)
with rasterio.open(STATIC_DIR / 'tsf_dem.tif', 'r+') as dem_dataset:
    dem_dataset.set_band_description(1, 'TSF elevation')
    dem_dataset.update_tags(1, units='m')

if SCENARIO['archetype'] == 'no_failure':
    timeline = [{'event': 'stable_period', 'date': START_DATE.strftime('%Y-%m-%d')}]
elif SCENARIO['archetype'] == 'gradual_acceleration':
    timeline = [
        {'event': event, 'date': (START_DATE + timedelta(days=SCENARIO[key])).strftime('%Y-%m-%d')}
        for event, key in [('slow_phase', 'onset_day'), ('developing_phase', 'velocity_increase_day'), ('accelerating_phase', 'acceleration_day'), ('failure', 'failure_day')]
    ]
elif SCENARIO['archetype'] == 'spatial_propagation':
    timeline = [
        {'event': 'localized_instability', 'date': (START_DATE + timedelta(days=SCENARIO['initiation_day'])).strftime('%Y-%m-%d')},
        {'event': 'failure', 'date': (START_DATE + timedelta(days=SCENARIO['failure_day'])).strftime('%Y-%m-%d')},
    ]
else:
    timeline = [
        {'event': event['type'], 'date': (START_DATE + timedelta(days=event['day'])).strftime('%Y-%m-%d')}
        for event in SCENARIO.get('events', [])
    ]
pd.DataFrame(timeline).to_csv(INPUTS_DIR / 'failure_timeline.csv', index=False)

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
            'dem_base_elevation_m',
            'dem_basin_depth_m',
            'dem_embankment_height_m',
        ],
        'value': [
            CRS,
            PIXEL_SIZE,
            YEARS,
            REVISIT_DAYS,
            FAILURE_START_DAY,
            A_MAX,
            ATM_SIGMA,
            MEAS_SIGMA,
            DEM_BASE_ELEVATION,
            DEM_BASIN_DEPTH,
            DEM_EMBANKMENT_HEIGHT,
        ],
    }
).to_csv(INPUTS_DIR / 'metadata.csv', index=False)

# Preserve the exact scenario definition alongside every generated data set.
with open(OUTDIR / 'scenario_config.yaml', 'w', encoding='utf-8') as stream:
    yaml.safe_dump({**CONFIG, 'active_scenario': SCENARIO_NAME}, stream, sort_keys=False)

print('Finished.')
print(f'Output: {OUTDIR.resolve()}')
