import glob
import pytest
import tempfile

from pathlib import Path
from unittest.mock import MagicMock

from lib.config import ProjectConfigReader
from subsystems.eou.manual_file_loader import ManualFileLoader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.metadata_processor import MetadataGenerator
from subsystems.dpr.data_export import DataExporter
from subsystems.isu import InSituDataUploader
from subsystems.sdi.loader import EarthObservationDataLoader, InSituDataLoader


def generate_eou_metadata_and_import(product_path, metadata_path, output_file_path):
    module = MetadataGenerator(product_path)
    module.stac.create_item()
    module.stac.save(metadata_path)

    create_sdi_package(product_path, metadata_path, output_file_path)

    importer = EarthObservationDataLoader(zip_path=output_file_path)
    importer.import_zip()


def create_sdi_package(product_path, metadata_path, output_file_path):
    extractor = DataExporter(product_path, metadata_path)
    extractor.create_sdi_package(output_file_path)


class TestConfig:
    def test_integration_EOU_001(self, tmp_path):
        """Test full-system integration for manual data from EOU -> DPR -> SDI."""
        metadata_temp = tempfile.NamedTemporaryFile(dir=tmp_path, suffix='.json')
        exported_temp = tempfile.NamedTemporaryFile(dir=tmp_path, suffix='.zip')

        test_file = (
            Path(__file__).parent.parent
            / 'subsystems'
            / 'eou'
            / 'tests'
            / 'sample_data'
            / 'ENMAP01_sample.tif'
        )

        module = ManualFileLoader()
        module.check_file_validity(test_file)

        generate_eou_metadata_and_import(
            test_file, metadata_temp.name, exported_temp.name
        )

    @pytest.mark.skip(reason='MetadataGenerator does not support S1 so far')
    def test_integration_EOU_002(self, tmp_path):
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

        config = ProjectConfigReader(
            Path(__file__).parent / 'projects' / 'jagersfontein.yml'
        )
        credentials = 'subsystems/eou/tests/eodag_config.yml'

        module = DataAcquisitionGateway()
        module.set_config(credentials)

        results = module.search(
            geom=config['project']['aoi']['geom'],
            **search_filter,
        )

        s1_path = module.download(results[0], output_dir=tmp_path)

        generate_eou_metadata_and_import(
            s1_path, metadata_temp.name, exported_temp.name
        )

    @pytest.mark.skip(reason='MetadataGenerator does not support S2 so far')
    def test_integration_EOU_003(self, tmp_path):
        """Test full-system integration for S2 from EOU -> DPR -> SDI."""
        metadata_temp = tempfile.NamedTemporaryFile(dir=tmp_path, suffix='.json')
        exported_temp = tempfile.NamedTemporaryFile(dir=tmp_path, suffix='.zip')

        search_filter = {
            'provider': 'cop_dataspace',
            'start': '2026-01-01',
            'end': '2026-01-29',
            'productType': 'S2_MSI_L2A',
        }

        config = ProjectConfigReader(
            Path(__file__).parent / 'projects' / 'jagersfontein.yml'
        )
        credentials = 'subsystems/eou/tests/eodag_config.yml'

        module = DataAcquisitionGateway()
        module.set_config(credentials)

        results = module.search(
            geom=config['project']['aoi']['geom'],
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

