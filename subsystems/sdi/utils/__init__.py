import hashlib
import requests
import tempfile

from lib.base import GaiaBase, SubsystemId


class SdiUtils(GaiaBase):
    def __init__(self):
        """Simple SDI client for searching and downloading assets."""
        GaiaBase.__init__(self, SubsystemId.SDI)

    def file_md5(self, path):
        """
        Compute MD5 hash of a file.
        """
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def delete_item_and_collection(self, stac_url, collection_id, item_id):
        requests.delete(
            f'{stac_url}/collections/{collection_id}/items/{item_id}',
        )

        requests.delete(
            f'{stac_url}/collections/{collection_id}',
        )

    def import_via_stac(self, stac_api_url, bbox, datetime, raster_file):
        """Verify STAC publication and asset integrity.

        Queries the STAC API for items intersecting the given bounding box and
        datetime, downloads the band asset, verifies that its content matches
        the provided raster file using an MD5 checksum, and checks that the
        corresponding COG asset is present.

        :param str stac_api_url: Base URL of the STAC API.
        :param bbox: Bounding box in the form ``(minx, miny, maxx, maxy)``.
        :type bbox: tuple[float, float, float, float] | list[float]
        :param str datetime: STAC datetime query (RFC 3339 timestamp or interval).
        :param str raster_file: Path to the reference GeoTIFF file.
        :raises requests.HTTPError: If the STAC API request or asset download fails.
        :raises AssertionError: If no matching STAC item is found, the required
        assets are missing, or the downloaded asset differs from the reference
        GeoTIFF.
        """

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
        md5_input = self.file_md5(raster_file)
        md5_downloaded = self.file_md5(temp_file.name)
        assert md5_input == md5_downloaded, (
            'Downloaded file does not match the original GeoTIFF'
        )

        # Search for COG item
        for stac_item in items:
            if 'B01_cog' in stac_item['assets']:
                asset = stac_item['assets']['B01_cog']
                asset_cog_url = asset['href']

        assert asset_cog_url, (
            'STAC asset for COG does not contain href or assets does not exist'
        )
