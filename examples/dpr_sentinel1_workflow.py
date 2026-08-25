from pathlib import Path

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.preprocessing_pipelines import PreprocessingPipelines
from lib.config import ProjectConfigReader, SettingsReader
from tests.utils import TestUtils

project_config = ProjectConfigReader(
    TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
)

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
        dem_path=data_dir / 'dem.nc',
        landmask_path=data_dir / 'landmask.nc',
        workdir=data_dir / 'workdir',
        result_dir=data_dir / 'results',
    )

    pipeline.run()
