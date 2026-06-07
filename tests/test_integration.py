import glob
import pytest
import tempfile
import requests

from pathlib import Path
from unittest.mock import MagicMock

from lib.config import ProjectConfigReader, SettingsReader
from subsystems.eou.manual_file_loader import ManualFileLoader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.metadata_processor import MetadataGenerator
from subsystems.dpr.data_export import DataExporter
from subsystems.isu import InSituDataUploader
from subsystems.sdi.loader import EarthObservationDataLoader, InSituDataLoader
from subsystems.sdi.utils import SdiUtils
from tests.utils import TestUtils


def generate_eou_metadata_and_import(product_path):
    # generate metadata
    module = MetadataGenerator()
    module.set_datasource(product_path)
    module.stac.create_item()
    metadata_path = module.stac.save()

    # TODO: MetadataValidator

    # create package for upload
    extractor = DataExporter(product_path, metadata_path)
    package_path = extractor.create_sdi_package()

    # upload package into SDI
    importer = EarthObservationDataLoader(zip_path=package_path)
    importer.import_zip()

    # TBD: Copied from subsystem.sdi.tests.test_import -> how to avoid code duplication
    
    # STAC query: search by bbox and datetime
    stac_api_url = importer.stac_api_url
    bbox = importer.stac_json['bbox']
    datetime = importer.stac_json['properties']['datetime']
    
    query_url = (
        f'{stac_api_url}/search?bbox={",".join(map(str, bbox))}&datetime={datetime}'
    )
    
    # Send request to STAC API
    resp = requests.post(query_url, json={})
    resp.raise_for_status()
    items = resp.json().get('features', [])
    assert items, 'STAC query returned no items'
    
    # Find the asset B01
    for stac_item in items:
        if 'data' in stac_item['assets']:
            asset = stac_item['assets']['data'] # B01?
            asset_url = asset['href']
            
    assert asset_url, 'STAC asset does not contain href'
        
    # Download the file from STAC asset URL
    temp_file = SettingsReader().temp_file()
    r = requests.get(asset_url, stream=True)
    r.raise_for_status()
    with open(temp_file, 'wb') as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    # Compare MD5 hash of downloaded file and input data_file
    utils = SdiUtils()
    md5_input = utils.file_md5(importer.raster_files[0])
    md5_downloaded = utils.file_md5(temp_file)
    assert md5_input == md5_downloaded, (
        'Downloaded file does not match the original data file'
    )


@pytest.fixture(scope='class')
def project_config():
    return ProjectConfigReader(
        TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
    )


class TestConfig:
    def test_integration_EOU_001_manual_loader(self, tmp_path):
        """Test full-system integration for manual data from EOU -> DPR -> SDI."""
        test_file = TestUtils.get_data_path('eou/ENMAP01_sample.tif')

        module = ManualFileLoader()
        module.check_file_validity(test_file)

        generate_eou_metadata_and_import(test_file)


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
