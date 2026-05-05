from pathlib import Path

from subsystems.sdi.loader import EarthObservationDataLoader
from subsystems.sdi.utils import SdiUtils
from subsystems.sdi.reader import SdiReader

from config import EO_COLLECTION, EO_ITEM_ID


class TestEarthObservationDataReader:
    def test_import_via_stac(self):
        utils = SdiUtils()

        base_dir = Path(__file__).parent
        zip_path = base_dir / 'assets' / 'eou_sample_data.zip'
        assert zip_path.exists()

        # Run the import: uploads raster to S3 and updates STAC
        importer = EarthObservationDataLoader(zip_path=zip_path)
        utils.delete_item_and_collection(
            importer.stac_api_url, EO_COLLECTION, EO_ITEM_ID
        )
        importer.import_zip()

        # Prepare STAC query parameters
        bbox = importer.stac_json['bbox']
        datetime_value = importer.stac_json['properties']['datetime']

        query_string = f'bbox={",".join(map(str, bbox))}&datetime={datetime_value}'

        # Use SdiReader to search for assets
        sdi_client = SdiReader()
        assets = sdi_client.search_assets(query_string, asset_name='B01')

        assert assets, 'STAC query returned no matching assets'

        asset = assets[0]
        asset_url = asset.get('href')
        assert asset_url, 'STAC asset does not contain href'

        # Download asset using StacClient
        downloaded_path = sdi_client.download_asset(asset_url)

        # Compare MD5 hash of downloaded file and original GeoTIFF
        utils = SdiUtils()
        md5_input = utils.file_md5(importer.raster_files[0])
        md5_downloaded = utils.file_md5(downloaded_path)

        assert md5_input == md5_downloaded, (
            'Downloaded file does not match the original GeoTIFF'
        )
