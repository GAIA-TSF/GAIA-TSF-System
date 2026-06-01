# Data Processing (DPR) Sub-system

The **Data Processing** sub-system serves as the central refinement
engine of the architecture, responsible for transforming raw inputs
into standardized, analysis-ready information products. It encompasses
three primary functional areas: metadata management, data
preprocessing, and advanced data analysis. The sub-system ensures that
all ingested data—whether from satellite imagery or in-situ
sensors—are accurately described, geometrically and atmospherically
corrected, and derived into meaningful indicators before storage. By
implementing modular pipelines, the system allows for the flexible
application of specific processing chains, such as atmospheric
correction for Earth Observation data or harmonized formatting for
sensor logs, ensuring compatibility with the Spatial Data
Infrastructure (SDI) and downstream Machine Learning (MAP)
sub-systems.

![Data Processor Architecture](../../images/dpr_subsystem.png)

## Usage

## SENTINEL-2 PROCESSING

## Sentinel2SafeProcessor

`Sentinel2SafeProcessor` performs the conversion of a Level2A Sentinel-2 SAFE product to a geotiff. Processing steps include:<br />
- (1) Location of the metadata files, extraction of key attributes, list of .jp2 files.<br />
- (2) Extraction of spectral and SCL bands from the GRANULE. Bands are then cropped and resampled using GDAL.<br />
- (3) Saving output geotiff and metadata to an output folder.<br />

**Parameters:**


`input_safe`<br />
`dtype` PosixPath
`description` Path to the Sentinel-2 SAFE product (can be a .zip file or an unzipped folder)

`output_folder`<br />
`dtype`: PosixPath
`description`Directory where the resulting geotiff and corresponding metadata will be saved.

`overwrite`<br />
`dtype`: bool
`description` If true overwrites any geotiff with same filename present in the output folder.
`default`: False

`roi`<br />
`dtype` str
`description` Bounding box as a POLYGON wkt string, coordinates must be in WGS84 (EPSG:4326).

`target_res`<br />
`dtype`: tuple
`description`: Pixel resolution (x_res, y_res) for the output raster.
`default`: (20, 20)

`resampling_alg`<br />
`dtype`: str
`description`: 'The GDAL resampling algorithm (e.g., bilinear, cubic, near).
`default`: 'near'

## Sentinel2CloudCoverPipeline

`Sentinel2CloudCoverPipeline` reads metadata JSON files, computes cloud cover and other land cover classes from the SCL band and write the percentages to the metadata file.<br />

**Parameters:**

`metadata_path`<br />
`dtype` PosixPath
`description` 'Path to the JSON file containing the path key.

`path_key`<br />
`dtype` str
`description` Name of the metadata dictionary key containing the path to the Sentinel-2 multiband raster.

`scl_band`<br />
`dtype` int
`description` The index of the Sentinel-2 SCL band.
`default` 13

## Sentinel2WaterMaskingPipeline

`Sentinel2WaterMaskingPipeline` generates water masks for Sentinel-2 scenes. Main processing steps include:<br />
- (1) Parsing metadata from input folder and create a list of scenes. Scenes with clouds/snow/dark are filtered out using threshold from `max_cloud_snow_dark` parameter. The user can apply a temporal filter (optional) using `start_date` and `end_date` parameters as well as specify months of the year (`input_months`).<br /><br />
- (2) Generating a "global" water mask. Filtered scenes are aggregated into a median raster. A water mask will be derived by thresholding spectral indices (`threshold_parameters`). This mask is then converted into a labeled array. This method is sensitive to topographic shadows. Winter months should be avoided. The optional `input_months` parameter can be used to filter scenes by months.<br />
Alternatively, the user can use the optional `input_water_mask` parameter to provide a vector file (e.g., *.shp, *.gpkg) with digitized water bodies (it is recommended to map the maximum extent of the water features). This shapefile will be converted into a labeled array and used as "global" water mask.<br /><br />
- (3) Generation of a water mask for each Sentinel-2 scenes. Thresholding on spectral indices (`threshold_parameters`) will be used to check the global water mask validity for a given scene. Pixels that do not fulfill thresholding conditions are removed to create a scene mask. If the scene mask contains valid pixels, it will be appended to the Sentinel-2 bands and the merged raster will be exported to the specified `output folder`.<br />

**Parameters:**

`input_folder`<br />
`dtype` *PosixPath*
`description` Path to the folder containing the processed (clipped) Sentinel-2 SAFE products'

`output_folder`<br />
`dtype` PosixPath
`description` Directory where the processed data will be saved.

`max_cloud_snow_dark`<br />
`dtype` float
`description` Parameter used to filter scenes based on the maximum percentage of pixels containing clouds/snow/shadows. Ratio between 0 and 1. Scenes with a ratio above this value will be filtered out.
`default` 0.2

`threshold_parameters`<br />
`dtype` dict
`description` Thresholds used to tag water bodies and discard soil/vegetation/other pixels.
`default` see self._config['threshold_parameters']

`input_water_mask`<br />
`dtype` PosixPath
`description` 'The user can provide a vector file (e.g., .shp, .gpkg) for water bodies instead of relying on the pipeline filtering.
`required` False

`input_months`<br />
`dtype` list,
`description` List of months (e.g., [1, 2, 5, 6]) to use for water masking.
`required` False

`start_date`<br />
`dtype` str
`description` Starting date for temporal filtering as "YYYY-MM-DD".
`required` False

`end_date`<br />
`dtype` str,
`description` End date for temporal filtering as "YYYY-MM-DD".
`required` False

`minimum_water_area_pixels`<br />
`dtype` int
`description` Minimum size of extracted water bodies in pixels.
`default` 4

`reference_spectra`<br />
`dtype` list
`description` List of reference spectra used as input for discarding pixels based on spectral similarity (i.e. Spectral Angle Mapper algorithm).
`default` see self._config['reference_spectra']

## Sentinel-2 workflow

```py
import os
import shutil
import glob
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from lib.config import ProjectConfigReader
from tests.utils import TestUtils
from subsystems.dpr.preprocessing_pipelines import Sentinel2SafeProcessor
from subsystems.dpr.preprocessing_pipelines import Sentinel2CloudCoverPipeline
from subsystems.dpr.data_analysis_pipelines import Sentinel2WaterMaskingPipeline

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' PART 1 - SENTINEL-2 PRODUCTS DOWNLOAD (SAFE)                                                                     '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

project_config = ProjectConfigReader(
    TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
)

dag_module = DataAcquisitionGateway(backend='eodag')

search_filter = {
    'provider': 'cop_dataspace',
    'start': '2025-05-01',
    'end': '2025-08-31',
    'productType': 'S2_MSI_L2A',
}

results = dag_module.backend.search(
    geom=project_config.aoi(),
    **search_filter,
)
data_path = dag_module.backend.download_all(results, target_dir='sentinel2', quicklook=False)
print(data_path)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' PART 2 - SENTINEL-2 CLIPPING AND RESAMPLING                                                                      '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Set output folder:
s2_processed_folder = os.path.abspath('tests/data/dpr/output_safe_processor')

# Set output resolution in meters
res = (10, 10)

# Set resampling algorithm
r_alg = 'bilinear'

# Get paths to SAFE files from data folder (we assume that they are still zipped)
base_dir = Path(data_path)
safe_list = [str(file) for file in list(base_dir.glob('*.zip'))]

# Get ROI from project_config
gdf = gpd.read_file(project_config.aoi(), layer=None)
gdf = gdf.to_crs(epsg=4326)
bounds = gdf.total_bounds
roi_geometry = box(*bounds)
roi_geometry = roi_geometry.wkt

# Run the pipeline
pipeline = Sentinel2SafeProcessor()
for safe_path in safe_list:
    pipeline.configure(
        input_safe=Path(safe_path),
        output_folder=Path(s2_processed_folder),
        roi=roi_geometry,
        target_res=res,
        resampling_alg=r_alg,
        overwrite=True,
    )
    pipeline.run()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' PART 3 - RETRIEVING CLOUD COVER PERCENTAGE FROM CLIPPED SCENES                                                  '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Path to the S2 processed folder containing the metadata JSON files (Same as PART2)
s2_processed_folder = os.path.abspath('tests/data/dpr/output_safe_processor')

# Retrieve all JSON files from the folder
meta_list = glob.glob(os.path.join(s2_processed_folder, "*.json"))
    
# Run the pipeline
pipeline = Sentinel2CloudCoverPipeline()
for meta_path in meta_list:
    pipeline.configure(metadata_path=meta_path, path_key='source_path')
    pipeline.run()
    
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' PART 4 - CREATION OF WATER MASKS                                                                                 '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Set path to folder containing the processed sentinel-2 scenes
input_folder = os.path.abspath('tests/data/dpr/output_safe_processor')

# Set path to the vector file containing the water bodies
input_water_mask = os.path.abspath('tests/data/dpr/yxsjoberg_lakes.gpkg')

# Set path to output folder (delete pre-existing folder)
output_folder = os.path.abspath('tests/data/dpr/output_water_masks')
if os.path.exists(output_folder) and os.path.isdir(output_folder):
    shutil.rmtree(output_folder)

# OPTION 1: Run the pipeline in default mode (without user input file, input_water_mask=None)
pipeline = Sentinel2WaterMaskingPipeline()
pipeline.configure(
    input_folder=Path(input_folder),
    output_folder=Path(output_folder),
    max_cloud_snow_dark=0.1,
    input_months=[4, 5, 6, 7, 8, 9, 10],
    start_date=project_config.monitoring_period.start(),
    end_date=project_config.monitoring_period.end(),
    input_water_mask=None,
    threshold_parameters={
        'shadow index': 0.9,
        'spectral angle': 0.15,
        'vnir regression slope': 1,
        'vnir regression intercept': -500,
        'band 2': 2000,
        'swir reflectance': 1000,
    },
)
pipeline.run()

# OPTION 2: Run the pipeline with vector file as input (input_water_mask provided)
pipeline = Sentinel2WaterMaskingPipeline()
pipeline.configure(
    input_folder=Path(input_folder),
    output_folder=Path(output_folder),
    input_water_mask=Path(input_water_mask),
    max_cloud_snow_dark=0.1,
    input_months=[4, 5, 6, 7, 8, 9, 10],
    start_date=project_config.monitoring_period.start(),
    end_date=project_config.monitoring_period.end(),
    threshold_parameters={
        'shadow index': 0.9,
        'spectral angle': 0.15,
        'vnir regression slope': 1,
        'vnir regression intercept': -500,
        'band 2': 2000,
        'swir reflectance': 1000,
    },
)
pipeline.run()
```

### Sentinel-1 preprocessing pipeline

Utilize the `Sentinel1Pipeline` from `PreprocessingPipelines` to automatically
process Sentinel-1 SLC BURST data to compute displacement maps. The example also
includes downloading Sentinel-1 SLC BURST data using the `DataAcquisitionGateway`
module.

```py
from pathlib import Path

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.preprocessing_pipelines import PreprocessingPipelines
from lib.config import ProjectConfigReader, SettingsReader
from tests.utils import TestUtils
base_dir = Path(SettingsReader()['storage']['data_dir']).resolve()
data_dir = base_dir / 'sentinel1'

if __name__ == '__main__':
    # download input data
    dag_module = DataAcquisitionGateway(backend='asf')
    results = dag_module.backend.search(
        geom=project_config.aoi(),
        start='2022-01-01',
        end='2022-01-31',
        direction='A',
    )
    dag_module.backend.download_all(results, target_dir=data_dir)

    # configure & run the pipeline
    pipeline = PreprocessingPipelines().pipelines['sentinel1']

    pipeline.configure(
        datadir=data_dir,
        aoi=project_config.aoi(),
        dem_path= data_dir / 'dem.nc',
        landmask_path= data_dir / 'landmask.nc',
        workdir= data_dir / 'workdir',
        result_dir= data_dir / 'results'
    )

    pipeline.run()
```

The results are stored in the `results` directory, which contains both
displacement and velocity data as well as environmental and risk
databases in CSV format.

`Sentinel1Pipeline` uses a Dask Cluster for its computations. The
parameters for this Dask Cluster can be configured in the global
`config.yaml` file (`dask_parameters` section). These should be
increased when processing larger volumes of data.
