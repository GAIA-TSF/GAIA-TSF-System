"""
Task: run EOU-DPR subsystems to download Sentinel-1 images for slope stability monitoring 
and prepo-cessing the images for given test site.

Usage: 
cd docker/

docker compose exec -T -u "$(id -u):$(id -g)" gaiatesting python3 -u -m subsystems.dpr.run_s1_eou_dpr 2>&1 | tee s1_run.log

"""
import os
from pathlib import Path

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.preprocessing_pipelines import PreprocessingPipelines
from lib.config import ProjectConfigReader, SettingsReader
from tests.utils import TestUtils

project_config = ProjectConfigReader(
    TestUtils.get_project_config_path('slope_monitoring_western_platinum')
)
# amd_monitoring_yxsjoberg

if __name__ == '__main__':

    # download input data
    data_dir = Path(project_config['project']['data_dir'], 'sentinel1')
    print(data_dir)
    
    dag_module = DataAcquisitionGateway(backend='asf')
    results = dag_module.backend.search(
        geom=project_config.aoi(),
        start='2022-01-01',
        end='2022-12-31',
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

 
