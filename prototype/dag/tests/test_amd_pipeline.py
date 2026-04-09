
from subsystems.dag.pipelines.amd_pipeline import AMDPipeline
from subsystems.dag.core.data_model import DataContainer


def test_amd_pipeline_runs():
    pipeline = AMDPipeline()

    input_data = DataContainer(data=None, metadata={})
    output = pipeline.run(input_data)

    assert output is not None 


""" 
from subsystems.dag.pipelines.amd_pipeline import AMDPipeline


def test_amd_pipeline_runs():
    pipeline = AMDPipeline()

    inputs = {
        's2': ['dummy.tif'],
        'aoi': 'aoi.geojson',
        'water_mask': 'mask.tif',
    }

    output = pipeline.run(inputs)

    assert output is not None
""" 