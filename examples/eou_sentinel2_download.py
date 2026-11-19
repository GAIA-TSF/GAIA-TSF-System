from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from lib.config import ProjectConfigReader
from tests.utils import TestUtils

project_config = ProjectConfigReader(
    TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
)

dag_module = DataAcquisitionGateway(backend='eodag')

search_filter = {
    'provider': 'cop_dataspace',
    'start': '2025-06-01',
    'end': '2025-06-05',
    'productType': 'S2_MSI_L2A',
}

# search & download is logged as a single task (TASK_STARTED ... TASK_FINISHED,
# or TASK_FAILED when an exception is raised)
with dag_module.logger.task(dag_module):
    results = dag_module.backend.search(
        geom=project_config.aoi(),
        **search_filter,
    )
    data_path = dag_module.backend.download_all(
        results, target_dir='sentinel2', quicklook=False
    )
