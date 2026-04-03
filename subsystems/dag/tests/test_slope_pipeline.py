from subsystems.dag.pipelines.slope_pipeline import SlopeStabilityPipeline


def test_slope_pipeline_runs():
    pipeline = SlopeStabilityPipeline()

    inputs = {
        's1': ['dummy.tif'],
        'aoi': 'aoi.geojson',
    }

    output = pipeline.run(inputs)

    assert output is not None
