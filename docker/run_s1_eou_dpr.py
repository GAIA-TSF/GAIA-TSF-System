"""
Task: run EOU-DPR subsystems to download Sentinel-1 images for slope stability monitoring 
and prepo-cessing the images for given test site.

Prep: 
1. set start & end dates 
2. create project /home/lukas/GAIA-TSF/src/GAIA-TSF-System/tests/projects/slope_monitoring_cadia
3. update project config.yaml 
4. set config.yaml data_dir:  /home/lukas/GAIA-TSF/tsf_experiments/slope_monitoring_cadia 


Usage: 
cd docker/
docker compose exec -u $(id -u):$(id -g) gaiatesting python3 run_s1_eou_dpr.py
new way: 
docker compose exec -T -u "$(id -u):$(id -g)" gaiatesting bash -c 'ulimit -Sn 65536 && exec python3 -u -m subsystems.dpr.run_s1_eou_dpr' 2>&1 | tee s1_run.log
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
        start='2015-01-01',
        end='2018-01-31',
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

### 
