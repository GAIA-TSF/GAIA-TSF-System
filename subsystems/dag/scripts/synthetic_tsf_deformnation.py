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
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from pathlib import Path
from datetime import datetime, timedelta
import json
from pyproj import Transformer

### CONFIGURATION ###

# project_dir = '/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/'
project_dir = '/home/lukas/ownCloud/Projects/GAIA_TSF/tsf_experiments/'
OUTDIR = Path(os.path.join(project_dir, 'synthetic_tsf_deformation_meteo'))
OUTDIR.mkdir(exist_ok=True, parents=True)
INPUTS_DIR = OUTDIR / 'inputs'
STATIC_DIR = OUTDIR / 'static'

CRS = 'EPSG:32633'
PIXEL_SIZE = 10

NX = 100
NY = 100

XMIN = 486768  # 500000
YMAX = 6656348  # 6700000

YEARS = 3
REVISIT_DAYS = 12

START_DATE = datetime(2018, 1, 1)

INCIDENCE_DEG = 39.0
FAILURE_START_DAY = 730

A_MAX = -0.10
HYDRO_AMP = 0.01

ATM_SIGMA = 0.012
MEAS_SIGMA = 0.003

RANDOM_SEED = 42

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

times = np.arange(0, YEARS * 365, REVISIT_DAYS)

for d in [
    'los',
    'true_los',
    'hotspot_prob',
    'labels',
    'failure_stage',
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

for t in times:
    acquisition_date = START_DATE + timedelta(days=int(t))

    date_str = acquisition_date.strftime('%Y%m%d')

    background = -0.005 * (t / (YEARS * 365)) * bowl

    seasonal = HYDRO_AMP * np.sin(2 * np.pi * t / 365) * bowl

    if t < FAILURE_START_DAY:
        sigma_hot = 3

        hotspot = np.exp(-((X - 75) ** 2 + (Y - 40) ** 2) / (2 * sigma_hot**2))

        hotspot_disp = np.zeros_like(bowl)

    else:
        t_fail = t - FAILURE_START_DAY

        sigma_hot = 3 + 10 * (t_fail / 365)

        hotspot = np.exp(-((X - 75) ** 2 + (Y - 40) ** 2) / (2 * sigma_hot**2))

        growth = 1.0 / (1.0 + np.exp(-0.03 * (t_fail - 180)))

        hotspot_disp = A_MAX * growth * hotspot

    rainfall = np.zeros_like(bowl)

    for ev in rain_events:
        if t >= ev:
            rainfall += -0.015 * np.exp(-(t - ev) / 40) * hotspot

    total_vertical = background + seasonal + hotspot_disp + rainfall

    true_los = total_vertical * np.cos(np.deg2rad(INCIDENCE_DEG))

    atmosphere = gaussian_filter(np.random.normal(0, ATM_SIGMA, (NY, NX)), sigma=10)

    measurement = np.random.normal(0, MEAS_SIGMA, (NY, NX))

    los_obs = true_los + atmosphere + measurement

    los_obs_masked = los_obs.copy()
    los_obs_masked[~mask] = np.nan

    label = (hotspot > 0.2).astype(np.uint8)

    stage = np.zeros((NY, NX), dtype=np.uint8)

    if t >= FAILURE_START_DAY:
        if t < 850:
            stage[label == 1] = 1
        elif t < 970:
            stage[label == 1] = 2
        else:
            stage[label == 1] = 3

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
        rainfall,
        acquisition_date=acquisition_date,
        product_type='rainfall_trigger',
        stage=int(stage.max()),
    )


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

pd.DataFrame(
    {
        'event': [
            'failure_start',
            'latent_phase',
            'developing_phase',
            'accelerating_phase',
        ],
        'date': [
            (START_DATE + timedelta(days=730)).strftime('%Y-%m-%d'),
            (START_DATE + timedelta(days=730)).strftime('%Y-%m-%d'),
            (START_DATE + timedelta(days=850)).strftime('%Y-%m-%d'),
            (START_DATE + timedelta(days=970)).strftime('%Y-%m-%d'),
        ],
    }
).to_csv(INPUTS_DIR / 'failure_timeline.csv', index=False)

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

print('Finished.')
print(f'Output: {OUTDIR.resolve()}')
