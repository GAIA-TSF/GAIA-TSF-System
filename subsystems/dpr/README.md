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

Get list of preprocessing pipelines:

```py
from subsystems.dpr.preprocessing_pipelines import PreprocessingPipelines
print(PreprocessingPipelines().pipelines.keys())
```

Select preprocessing pipeline and get it's usage:

```
print(PreprocessingPipelines().pipelines['sentinel2_cloudcover'].metadata)
```

### Sentinel-1 processing

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

### Sentinel-2 processing

Preprocessing pipelines:

- `Sentinel2SafeProcessor` performs the conversion of a Level2A Sentinel-2 SAFE product to a image data file.
- `Sentinel2CloudCoverPipeline` reads metadata JSON files, computes cloud cover and other land cover classes from the SCL band and write the percentages to the metadata file.

Data analytical pipelines:

- `Sentinel2WaterMaskingPipeline` generates water masks for Sentinel-2 scenes.

The code example below demonstrates downloading Sentinel-2 data using
the `DataAcquisitionGateway` and applying the preprocessing pipelines
`Sentinel2SafeProcessor` and `Sentinel2CloudCoverPipeline`. Finally, the
data analysis pipeline `Sentinel2WaterMaskingPipeline` is executed.

See
[dpr_sentinel2_workflow.py](../../examples/dpr_sentinel2_workflow.py)
for complete Sentinel-2 workflow.