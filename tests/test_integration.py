import glob
import json
import pytest
import tempfile

from pathlib import Path
from unittest.mock import MagicMock

from lib.config import ProjectConfigReader
from subsystems.dpr import DataProcessing
from subsystems.dpr.metadata_processor import MetadataGenerator
from subsystems.dpr.data_export import DataExporter
from subsystems.eou.manual_file_loader import ManualFileLoader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.isu import InSituDataUploader
from subsystems.isu.etl_engine.pipeline import ETLEngine
from subsystems.qcl import QCLayer
from subsystems.sdi import SpatialDataInfrastructure
from subsystems.sdi.loader import EarthObservationDataLoader, InSituDataLoader
from tests.utils import TestUtils


def generate_eou_metadata_and_import(product_path, metadata_path, output_file_path):
    module = MetadataGenerator()
    module.set_datasource(product_path)
    module.stac.create_item()
    module.stac.save(metadata_path)

    create_sdi_package(product_path, metadata_path, output_file_path)

    importer = EarthObservationDataLoader(zip_path=output_file_path)
    importer.import_zip()


def create_sdi_package(product_path, metadata_path, output_file_path):
    extractor = DataExporter(product_path, metadata_path)
    extractor.create_sdi_package(output_file_path)


@pytest.fixture(scope='class')
def project_config():
    return ProjectConfigReader(
        TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
    )


class TestConfig:
    def test_integration_EOU_001(self, tmp_path):
        """Test full-system integration for manual data from EOU -> DPR -> SDI."""
        metadata_temp = tempfile.NamedTemporaryFile(dir=tmp_path, suffix='.json')
        exported_temp = tempfile.NamedTemporaryFile(dir=tmp_path, suffix='.zip')

        test_file = TestUtils.get_data_path('eou/ENMAP01_sample.tif')

        module = ManualFileLoader()
        module.check_file_validity(test_file)

        generate_eou_metadata_and_import(
            test_file, metadata_temp.name, exported_temp.name
        )

    @pytest.mark.skip(reason='MetadataGenerator does not support S1 so far')
    def test_integration_EOU_002(self, tmp_path, project_config):
        """Test full-system integration for S1 from EOU -> DPR -> SDI."""
        metadata_temp = tempfile.NamedTemporaryFile(dir=tmp_path, suffix='.json')
        exported_temp = tempfile.NamedTemporaryFile(dir=tmp_path, suffix='.zip')

        search_filter = {
            'provider': 'cop_dataspace',
            'start': '2026-01-01',
            'end': '2026-01-29',
            'productType': 'S1_SAR_SLC',
            'orbitDirection': 'ascending',
        }

        credentials = 'subsystems/eou/tests/eodag_config.yml'

        module = DataAcquisitionGateway()
        module.set_config(credentials)

        results = module.search(
            geom=project_config.aoi(),
            **search_filter,
        )

        s1_path = module.download(results[0], output_dir=tmp_path)

        generate_eou_metadata_and_import(
            s1_path, metadata_temp.name, exported_temp.name
        )

    @pytest.mark.skip(reason='MetadataGenerator does not support S2 so far')
    def test_integration_EOU_003(self, tmp_path, project_config):
        """Test full-system integration for S2 from EOU -> DPR -> SDI."""
        metadata_temp = tempfile.NamedTemporaryFile(dir=tmp_path, suffix='.json')
        exported_temp = tempfile.NamedTemporaryFile(dir=tmp_path, suffix='.zip')

        search_filter = {
            'provider': 'cop_dataspace',
            'start': '2026-01-01',
            'end': '2026-01-29',
            'productType': 'S2_MSI_L2A',
        }

        credentials = 'subsystems/eou/tests/eodag_config.yml'

        module = DataAcquisitionGateway()
        module.set_config(credentials)

        results = module.search(
            geom=project_config.aoi(),
            **search_filter,
        )

        s2_path = module.download(results[0], output_dir=tmp_path)

        generate_eou_metadata_and_import(
            s2_path, metadata_temp.name, exported_temp.name
        )

    @pytest.mark.skip(reason='MetadataGenerator does not support ISU output so far')
    def test_integration_ISU_001(self, tmp_path):
        """Test full-system integration from ISU -> DPR -> SDI."""
        metadata_temp = tempfile.NamedTemporaryFile(dir=tmp_path, suffix='.json')
        exported_temp = tempfile.NamedTemporaryFile(dir=tmp_path, suffix='.zip')

        test_data = (
            Path(__file__).parent.parent / 'subsystems' / 'isu' / 'tests' / 'test_data'
        )

        isu = InSituDataUploader(input_dir=str(test_data), processed_dir=str(tmp_path))

        isu.scheduler = MagicMock()
        isu.start()
        isu._scan_and_process_files()
        isu.scheduler.start.assert_called_once()

        # Test Stop
        isu.stop()

        # TODO: Is there a better way to get the product and metadata path?
        subsystem_output = glob.glob(str(tmp_path / '*.csv'))[0]

        create_sdi_package(subsystem_output, metadata_temp.name, exported_temp.name)

        importer = InSituDataLoader(zip_path=exported_temp.name)
        importer.import_zip()


class TestISUIntegration:
    """Full-pipeline integration tests using real in-situ datasets.

    Pipeline under test: ISU (parse + QC) → QCL (validate + SDI log) → DPR (STAC) → SDI (import).
    Runs against live PostGIS and STAC API services in the Docker test environment.
    """

    @pytest.mark.parametrize(
        'filename,expected_sensor_type',
        [
            (
                'Piezometer1.csv',
                'piezometer',
            ),  # UTF-8, multi-depth T/P deployment sensors
            (
                'Piezometer2.csv',
                'piezometer',
            ),  # GBK, dam monitoring DataSetI-IV + X/Y(mm)
        ],
    )
    def test_integration_ISU_002(self, tmp_path, filename, expected_sensor_type):
        """Full pipeline with real in-situ data: ISU → QCL → DPR → SDI."""
        content = TestUtils.get_data_path(f'isu/{filename}').read_bytes()

        sdi = SpatialDataInfrastructure()
        etl = ETLEngine(
            qc_layer=QCLayer(sdi_service=sdi),
            dpr_service=DataProcessing(),
        )

        result = etl.process_file(content, filename)

        # --- ISU: file parsed and not quarantined ---
        assert result is not None, (
            f'{filename} was quarantined — parser score below threshold'
        )
        meta = result['metadata']['data']
        assert meta['sensor_type'] == expected_sensor_type
        assert meta['time_range'] is not None

        # --- QCL: data passed quality control ---
        assert result['qc_result']['final_status'] in ('Pass', 'Warn')

        # --- DPR: STAC item generated ---
        assert result['dpr_result']['ready_for_sdi'] is True
        stac = result['dpr_result']['stac_item']
        assert stac['type'] == 'Feature'
        assert stac['properties'].get('sensor_type') == expected_sensor_type

        # --- SDI: assemble package and run full import against live services ---
        stem = Path(filename).stem
        csv_path = tmp_path / filename
        stac_path = tmp_path / f'{stem}_stac.json'
        zip_path = tmp_path / f'{stem}_package.zip'

        result['data'].to_csv(csv_path, index=False)
        stac_path.write_text(json.dumps(stac))

        create_sdi_package(str(csv_path), str(stac_path), str(zip_path))
        assert zip_path.exists()

        loader = InSituDataLoader(zip_path=str(zip_path))
        loader.import_zip()
