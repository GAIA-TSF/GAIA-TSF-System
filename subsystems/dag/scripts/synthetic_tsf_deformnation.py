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
OUTDIR = Path(os.path.join(project_dir, 'synthetic_tsf_deformation_metadata'))
OUTDIR.mkdir(exist_ok=True, parents=True)

CRS = 'EPSG:32633'
PIXEL_SIZE = 10

NX = 100
NY = 100

XMIN = 500000
YMAX = 6700000

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
    (OUTDIR / d).mkdir(parents=True, exist_ok=True)

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
    dtype="float32",
    nodata=None,
):

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=CRS,
        transform=transform,
        nodata=nodata,
        compress="lzw",
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
        "EPSG:4326",
        always_xy=True,
    )

    xmin = XMIN
    xmax = XMIN + NX * PIXEL_SIZE

    ymax = YMAX
    ymin = YMAX - NY * PIXEL_SIZE

    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)

    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [lon_min, lat_min],
            [lon_min, lat_max],
            [lon_max, lat_max],
            [lon_max, lat_min],
            [lon_min, lat_min],
        ]],
    }

    stage_name = {
        0: "stable",
        1: "latent",
        2: "developing",
        3: "accelerating",
    }[stage]

    item = {

        "type": "Feature",

        "stac_version": "1.0.0",

        "id": tif_path.stem,

        "bbox": [
            lon_min,
            lat_min,
            lon_max,
            lat_max,
        ],

        "geometry": geometry,

        "properties": {

            "datetime":
                acquisition_date.strftime(
                    "%Y-%m-%dT00:00:00Z"
                ),

            "platform":
                "synthetic-tsf",

            "constellation":
                "synthetic",

            "instruments": [
                "simulation"
            ],

            "processing:level":
                "L2",

            "proj:epsg":
                32633,

            "simulation:type":
                product_type,

            "simulation:failure_stage":
                stage_name,

            "simulation:unit":
                "m",

        },

        "assets": {

            "data": {

                "href":
                    "./" + tif_path.name,

                "type":
                    "image/tiff; application=geotiff",

                "roles": [
                    "data"
                ],

            }

        },

        "collection":
            "gaia-tsf-synthetic",

    }

    json_file = tif_path.with_suffix(".json")

    with open(json_file, "w") as f:
        json.dump(
            item,
            f,
            indent=4,
        )


### SYNTHETIC PRECIPITATION ###
precip_records = []

# rainfall trigger events that contribute to instability
trigger_events = {790: 45, 860: 62, 950: 78, 1030: 55}

### DATA SIMULATION ###

for t in times:
    acquisition_date = START_DATE + timedelta(days=int(t))

    date_str = acquisition_date.strftime('%Y%m%d')

    # PRECIPITATION
    # background precipitation
    precip_mm = np.random.gamma(shape=1.5, scale=4.0)

    # stronger events used for instability triggering
    for ev_day, ev_precip in trigger_events.items():
        if abs(t - ev_day) <= 6:
            precip_mm = ev_precip

    precip_records.append({'date': date_str, 'precipitation_mm': round(precip_mm, 1)})

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

    # write_tif(OUTDIR / 'los' / f'tsf_los_{date_str}.tif', los_obs_masked, nodata=np.nan) 
    write_tif(
        OUTDIR / "los" / f"tsf_los_{date_str}.tif",
        los_obs_masked,
        acquisition_date=acquisition_date,
        product_type="los",
        stage=int(stage.max()),
        nodata=np.nan,
    )

    # write_tif(OUTDIR / 'true_los' / f'true_los_{date_str}.tif', true_los)
    write_tif(
        OUTDIR / "true_los" / f"true_los_{date_str}.tif",
        true_los,
        acquisition_date=acquisition_date,
        product_type="true_los",
        stage=int(stage.max()),
    )

    # write_tif(OUTDIR / 'hotspot_prob' / f'hotspot_prob_{date_str}.tif', hotspot)
    write_tif(
        OUTDIR / "hotspot_prob" / f"hotspot_prob_{date_str}.tif",
        hotspot,
        acquisition_date=acquisition_date,
        product_type="hotspot_probability",
        stage=int(stage.max()),
    )

    # write_tif(OUTDIR / 'labels' / f'hotspot_label_{date_str}.tif', label, dtype='uint8')
    write_tif(
        OUTDIR / "labels" / f"hotspot_label_{date_str}.tif",
        label,
        acquisition_date=acquisition_date,
        product_type="binary_label",
        stage=int(stage.max()),
        dtype="uint8",
    )

    # write_tif(
    #     OUTDIR / 'failure_stage' / f'failure_stage_{date_str}.tif', stage, dtype='uint8'
    # )
    write_tif(
        OUTDIR / "failure_stage" / f"failure_stage_{date_str}.tif",
        stage,
        acquisition_date=acquisition_date,
        product_type="failure_stage",
        stage=int(stage.max()),
        dtype="uint8",
    )

    # write_tif(OUTDIR / 'atmosphere' / f'atmosphere_{date_str}.tif', atmosphere)
    write_tif(
        OUTDIR / "atmosphere" / f"atmosphere_{date_str}.tif",
        atmosphere,
        acquisition_date=acquisition_date,
        product_type="atmosphere",
        stage=int(stage.max()),
    )

    # write_tif(OUTDIR / 'rainfall' / f'rainfall_trigger_{date_str}.tif', rainfall)
    write_tif(
        OUTDIR / "rainfall" / f"rainfall_trigger_{date_str}.tif",
        rainfall,
        acquisition_date=acquisition_date,
        product_type="rainfall_trigger",
        stage=int(stage.max()),
    )


### PRECIPITATION CSV ###
pd.DataFrame(precip_records).to_csv(OUTDIR / 'meteo_precipitation.csv', index=False)

### AUXILIARY EXPORTS ###
write_tif(OUTDIR / 'aux' / 'coherence_mask.tif', mask.astype(np.uint8), dtype='uint8')

write_tif(OUTDIR / 'aux' / 'tsf_mask.tif', (bowl > 0.1).astype(np.uint8), dtype='uint8')

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
).to_csv(OUTDIR / 'failure_timeline.csv', index=False)

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
        ],
    }
).to_csv(OUTDIR / 'metadata.csv', index=False)

print('Finished.')
print(f'Output: {OUTDIR.resolve()}')
