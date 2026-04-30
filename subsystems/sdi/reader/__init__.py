import requests
import tempfile
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from lib.base import GaiaBase, SubsystemId

STAC_URL = 'http://stacapi:8000'


class SdiReader(GaiaBase):
    def __init__(self):
        """Simple SDI client for searching and downloading assets."""
        super().__init__(SubsystemId.SDI)

    def search_assets(
            self, query_string: str, asset_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform a STAC search request and return a list of matching assets with metadata.

        :param query_string: Query string part for STAC search
        :param asset_name: Optional asset key (e.g. 'B01')
        :return: List of asset dictionaries with timestamp, item_id, and other metadata
        """
        search_url = f'{STAC_URL}/search?{query_string}'

        response = requests.post(search_url, json={})
        response.raise_for_status()

        features = response.json().get('features', [])
        if not features:
            return []

        results = []

        for item in features:
            # Extract metadata from item
            properties = item.get('properties', {})
            timestamp = properties.get('datetime')
            item_id = item.get('id')
            collection = item.get('collection')

            # Additional useful metadata
            cloud_cover = properties.get('eo:cloud_cover')
            platform = properties.get('platform')
            instruments = properties.get('instruments', [])

            assets = item.get('assets', {})

            if asset_name:
                # Return only the specified asset
                if asset_name in assets:
                    asset = assets[asset_name].copy()
                    asset.update({
                        'timestamp': timestamp,
                        'item_id': item_id,
                        'collection': collection,
                        'cloud_cover': cloud_cover,
                        'platform': platform,
                        'instruments': instruments,
                        'asset_key': asset_name
                    })
                    results.append(asset)
            else:
                # Return all assets
                for key, asset_data in assets.items():
                    asset = asset_data.copy()
                    asset.update({
                        'timestamp': timestamp,
                        'item_id': item_id,
                        'collection': collection,
                        'cloud_cover': cloud_cover,
                        'platform': platform,
                        'instruments': instruments,
                        'asset_key': key
                    })
                    results.append(asset)

        return results

    def download_asset(self, asset_href: str) -> Path:
        """
        Download a STAC asset to a temporary file and return its local path.

        :param asset_href: Asset URL (href) to download.
        :return: Path to the downloaded temporary file.
        """
        response = requests.get(asset_href, stream=True)
        response.raise_for_status()

        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        tmp_path = Path(tmp_file.name)

        # Stream download to avoid loading the whole file into memory
        with open(tmp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return tmp_path
