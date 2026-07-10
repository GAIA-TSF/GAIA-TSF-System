import psycopg
import pytest

from subsystems.sdi.loader import InSituDataLoader
from subsystems.sdi.loader import EarthObservationDataLoader
from subsystems.sdi.utils import SdiUtils

from lib.config import SettingsReader

from config import INSITU_COLLECTION, INSITU_ITEM_ID
from config import EO_COLLECTION, EO_ITEM_ID
from tests.utils import TestUtils

TEST_DATA_DIR = TestUtils.get_data_path('sdi')


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

            # Test that id and site_id columns exist
            cur.execute("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'prague'
                          AND table_name = 'measurement_ph_202602_data'
                          AND column_name IN ('id', 'site_id');
                        """)
            columns = [row[0] for row in cur.fetchall()]

            assert 'id' in columns, "Column 'id' not found in table"
            assert 'site_id' in columns, "Column 'site_id' not found in table"

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
        utils.import_via_stac(
            importer.stac_api_url,
            importer.stac_json['bbox'],
            importer.stac_json['properties']['datetime'],
            importer.raster_files[0],
        )

    def test_failing_import(self):
        zip_path = TEST_DATA_DIR / 'eou_sample_data_bad_metadata.zip'
        assert zip_path.exists()

        importer = EarthObservationDataLoader(zip_path=zip_path)

        # Expecting expection
        with pytest.raises(ValueError) as excinfo:
            importer.import_zip()

        # Checking text of the exception
        assert 'Missing required fields' in str(excinfo.value)
