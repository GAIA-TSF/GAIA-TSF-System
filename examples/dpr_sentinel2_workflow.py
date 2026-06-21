from pathlib import Path

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from lib.config import ProjectConfigReader, SettingsReader
from tests.utils import TestUtils
from subsystems.dpr.preprocessing_pipelines import Sentinel2SafeProcessor
from subsystems.dpr.preprocessing_pipelines import Sentinel2CloudCoverPipeline
from subsystems.dpr.data_analysis_pipelines import Sentinel2WaterMaskingPipeline

#
# PART 1 - SENTINEL-2 PRODUCTS DOWNLOAD (SAFE)
#
project_config = ProjectConfigReader(
    TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
)

dag_module = DataAcquisitionGateway(backend='eodag')

search_filter = {
    'provider': 'cop_dataspace',
    'start': '2025-06-01',
    'end': '2025-08-31',
    'productType': 'S2_MSI_L2A',
}

results = dag_module.backend.search(
    geom=project_config.aoi(),
    **search_filter,
)
safe_list = dag_module.backend.download_all(
    results, target_dir='sentinel2', quicklook=False
)

#
# PART 2 - SENTINEL-2 CLIPPING AND RESAMPLING
#

# Set output folder:
s2_processed_folder = (
    Path(SettingsReader()['storage']['data_dir'])
    / 'sentinel2'
    / 'output_safe_processor'
)

# Set output resolution in meters
res = (10, 10)

# Set resampling algorithm
r_alg = 'bilinear'

# Run the pipeline
pipeline = Sentinel2SafeProcessor()
for safe_path in safe_list:
    pipeline.configure(
        input_safe=safe_path,
        output_folder=s2_processed_folder,
        roi=project_config.aoi(roi=True).wkt,
        target_res=res,
        resampling_alg=r_alg,
        overwrite=True,
    )
    pipeline.run()

#
# PART 3 - RETRIEVING CLOUD COVER PERCENTAGE FROM CLIPPED SCENES
#

# Retrieve all JSON files from the folder
meta_list = s2_processed_folder.glob('*.json')

# Run the pipeline
pipeline = Sentinel2CloudCoverPipeline()
for meta_path in meta_list:
    pipeline.configure(metadata_path=meta_path, path_key='source_path')
    pipeline.run()

#
# PART 4 - CREATION OF WATER MASKS
#

# Set path to the vector file containing the water bodies
input_water_mask = TestUtils.get_data_path('dpr') / 'yxsjoberg_lakes.gpkg'

# Set path to output folder (delete pre-existing folder)
output_folder = (
    Path(SettingsReader()['storage']['data_dir']) / 'sentinel2' / 'output_water_masks'
)

# OPTION 1: Run the pipeline in default mode (without user input file, input_water_mask=None)
pipeline = Sentinel2WaterMaskingPipeline()
pipeline.configure(
    input_folder=s2_processed_folder,
    output_folder=output_folder,
    max_cloud_snow_dark=0.1,
    input_months=[4, 5, 6, 7, 8, 9, 10],
    start_date=project_config.monitoring_period.start,
    end_date=project_config.monitoring_period.end,
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
    input_folder=s2_processed_folder,
    output_folder=output_folder,
    input_water_mask=input_water_mask,
    max_cloud_snow_dark=0.1,
    input_months=[4, 5, 6, 7, 8, 9, 10],
    start_date=project_config.monitoring_period.start,
    end_date=project_config.monitoring_period.end,
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
