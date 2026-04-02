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
