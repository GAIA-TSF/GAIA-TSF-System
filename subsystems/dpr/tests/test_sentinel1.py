import sys
from pathlib import Path


# to be removed when https://github.com/GAIA-TSF/GAIA-TSF-System/issues/97 is solved
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from lib.config import ProjectConfigReader
from eou.data_acquisition_gateway import DataAcquisitionGateway
from dpr.preprocessing_pipelines import PreprocessingPipelines


class TestSentinel1Workflow:
    search_filter = {
        'provider': 'cop_dataspace',
        'start': '2026-01-01',
        'end': '2026-01-29',
        'productType': 'S1_SAR_SLC',
    }

    def test_download(self):
        """Test EOU Data Acquisition Gateway to download Sentinel-1 data."""
        config = ProjectConfigReader(
            str(
                Path(__file__).parent.parent.parent.parent
                / 'tests'
                / 'projects'
                / 'jagersfontein.yml'
            )
        )

        assert config.is_valid() is True

        module = DataAcquisitionGateway()
        results = module.search(
            geom=config['project']['aoi']['geom'],
            **self.search_filter,
        )

        assert len(results) > 0

        config_eodag = str(
            Path(__file__).parent.parent.parent / 'eou' / 'tests' / 'eodag_config.yml'
        )
        module.set_config(config_eodag)

        ql_path = None
        try:
            ql_path = module.download(results[0], quicklook=True)
            assert Path(ql_path).exists()
        finally:
            if ql_path and Path(ql_path).exists():
                Path(ql_path).unlink()

    def test_run_workflow(self):
        module = PreprocessingPipelines()
        # TBD: propapage project configuration file
        pipeline = module.pipeline['sentinel1']
        pipeline.run() # TBD: to be implemented

        # TBD: check results
