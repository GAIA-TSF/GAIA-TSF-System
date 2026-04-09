
from subsystems.dag.pipelines.slope_pipeline import SlopePipeline
from subsystems.dag.core.data_model import DataContainer


def test_slope_pipeline_runs():
    pipeline = SlopePipeline()

    input_data = DataContainer(data=None, metadata={})
    output = pipeline.run(input_data)

    assert output is not None 

