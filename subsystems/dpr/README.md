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

```py
print(PreprocessingPipelines().pipelines['sentinel2_cloudcover'].metadata)
```

### Sentinel-1 processing

Utilize the `Sentinel1Pipeline` from `PreprocessingPipelines` to automatically
process Sentinel-1 SLC BURST data to compute displacement maps. The example also
includes downloading Sentinel-1 SLC BURST data using the `DataAcquisitionGateway`
module.

See
[dpr_sentinel1_workflow.py](../../examples/dpr_sentinel1_workflow.py)
for complete Sentinel-1 workflow.

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
