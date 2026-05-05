import requests
import tempfile

import psycopg
import pytest

from subsystems.sdi.loader import InSituDataLoader
from subsystems.sdi.loader import EarthObservationDataLoader
from subsystems.sdi.utils import SdiUtils

from lib.config import SettingsReader

from config import INSITU_COLLECTION, INSITU_ITEM_ID
from config import EO_COLLECTION, EO_ITEM_ID
from tests.utils import get_data_path

TEST_DATA_DIR = get_data_path('sdi')


class TestInSituDataLoader:
    @pytest.fixture(scope='module')
    def db_connection(self):
        settings = SettingsReader()
        conn = psycopg.connect(**settings['sdi']['db'])
        yield conn
        conn.close()

    def test_import(self, db_connection):
        utils = SdiUtils()

        zip_path = TEST_DATA_DIR / 'isu_sample_data.zip'

        assert zip_path.exists()

        importer = InSituDataLoader(zip_path=zip_path)
        utils.delete_item_and_collection(
            importer.stac_api_url, INSITU_COLLECTION, INSITU_ITEM_ID
        )

        importer.import_zip()

        with db_connection.cursor() as cur:
            cur.execute("""
                        SELECT COUNT(*)
                        FROM prague.measurement_ph_202602_data;
                        """)
            count = cur.fetchone()[0]
            assert count == 20

    def test_import_append_data(self, db_connection):
        utils = SdiUtils()

        zip_path = TEST_DATA_DIR / 'isu_sample_data.zip'

        assert zip_path.exists()

        importer = InSituDataLoader(zip_path=zip_path)
        utils.delete_item_and_collection(
            importer.stac_api_url, INSITU_COLLECTION, INSITU_ITEM_ID
        )

        importer.import_zip(append_data=True)
        importer.import_zip(append_data=True)

        with db_connection.cursor() as cur:
            cur.execute("""
                        SELECT COUNT(*)
                        FROM prague.measurement_ph_202602_data;
                        """)
            count = cur.fetchone()[0]
            assert count > 20


class TestEarthObservationDataLoader:
    def test_import_via_stac(self):
        utils = SdiUtils()

        zip_path = TEST_DATA_DIR / 'eou_sample_data.zip'
        assert zip_path.exists()

        # Run the import: uploads raster to S3 and updates STAC
        importer = EarthObservationDataLoader(zip_path=zip_path)
        utils.delete_item_and_collection(
            importer.stac_api_url, EO_COLLECTION, EO_ITEM_ID
        )
        importer.import_zip()

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
            if 'B01' in stac_item['assets']:
                asset = stac_item['assets']['B01']
                asset_url = asset['href']

        assert asset_url, 'STAC asset does not contain href'

        # Download the file from STAC asset URL
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        r = requests.get(asset_url, stream=True)
        r.raise_for_status()
        with open(temp_file.name, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        # Compare MD5 hash of downloaded file and input GeoTIFF

        md5_input = utils.file_md5(importer.raster_files[0])
        md5_downloaded = utils.file_md5(temp_file.name)
        assert (
            md5_input == md5_downloaded
        ), 'Downloaded file does not match the original GeoTIFF'

    def test_failing_import(self):
        zip_path = TEST_DATA_DIR / 'eou_sample_data_bad_metadata.zip'
        assert zip_path.exists()

        importer = EarthObservationDataLoader(zip_path=zip_path)

        # Expecting expection
        with pytest.raises(ValueError) as excinfo:
            importer.import_zip()

        # Checking text of the exception
        assert 'Missing required fields' in str(excinfo.value)
