import json
import pytest
import requests

from pathlib import Path

from lib.config import ProjectConfigReader, SettingsReader
from subsystems.eou.manual_file_loader import ManualFileLoader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from subsystems.dpr.metadata_processor import MetadataGenerator
from subsystems.dpr.data_export import DataExporter
from subsystems.dpr import DataProcessing
from subsystems.isu.etl_engine.pipeline import ETLEngine
from subsystems.qcl import QCLayer
from subsystems.sdi import SpatialDataInfrastructure
from subsystems.sdi.loader import EarthObservationDataLoader, InSituDataLoader
from subsystems.sdi.utils import SdiUtils
from tests.utils import TestUtils


def create_sdi_package(product_path, metadata_path):
    extractor = DataExporter(product_path, metadata_path)
    zip_path = extractor.create_sdi_package()
    assert zip_path.exists()

    return zip_path


def generate_eou_metadata_and_import(product_path):
    # generate metadata
    module = MetadataGenerator()
    module.set_datasource(product_path)
    module.stac.create_item()
    metadata_path = module.stac.save()

    # TODO: MetadataValidator

    # create package for upload
    package_path = create_sdi_package(product_path, metadata_path)

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
            asset = stac_item['assets']['data']  # B01?
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


class TestEOUIntegration:
    def test_integration_EOU_001_manual_loader(self):
        """Test full-system integration for manual data from EOU -> DPR -> SDI."""
        test_file = TestUtils.get_data_path('eou/ENMAP01_sample.tif')

        module = ManualFileLoader()
        module.check_file_validity(test_file)

        generate_eou_metadata_and_import(test_file)

    @pytest.mark.slow
    def test_integration_EOU_002_sentinel1(self, tmp_path, project_config):
        """Test full-system integration for S1 from EOU -> DPR -> SDI."""
        search_filter = {
            'provider': 'cop_dataspace',
            'start': '2026-01-01',
            'end': '2026-01-29',
            'productType': 'S1_SAR_SLC',
            'orbitDirection': 'ascending',
        }

        module = DataAcquisitionGateway()

        results = module.backend.search(
            geom=project_config.aoi(),
            **search_filter,
        )

        s1_path = module.backend.download(results[0], target_dir='sentinel1')

        generate_eou_metadata_and_import(s1_path)

    @pytest.mark.slow
    def test_integration_EOU_003_sentinel2(self, project_config):
        """Test full-system integration for S2 from EOU -> DPR -> SDI."""

        search_filter = {
            'provider': 'cop_dataspace',
            'start': '2026-01-01',
            'end': '2026-01-29',
            'productType': 'S2_MSI_L2A',
        }

        module = DataAcquisitionGateway()

        results = module.backend.search(
            geom=project_config.aoi(),
            **search_filter,
        )

        s2_path = module.backend.download(results[0], target_dir='sentinel2')

        generate_eou_metadata_and_import(s2_path)


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

        zip_path = create_sdi_package(csv_path, stac_path)

        loader = InSituDataLoader(zip_path=str(zip_path))
        loader.import_zip()
