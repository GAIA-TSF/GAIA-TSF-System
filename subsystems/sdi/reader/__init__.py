import requests
import tempfile
from pathlib import Path
from typing import Optional, List, Dict

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
        Perform a STAC search request and return a list of matching assets.

        :param query_string: Query string part for STAC search
                             (e.g. "bbox=...&datetime=...")
        :param asset_name: Optional asset key (e.g. 'B01').
                           If provided, only assets with this name are returned.
                           If None, all assets from matching items are returned.
        :return: List of asset dictionaries (each containing at least 'href' and metadata).
        """
        search_url = f'{STAC_URL}/search?{query_string}'

        response = requests.post(search_url, json={})
        response.raise_for_status()

        features = response.json().get('features', [])
        if not features:
            return []

        results = []

        for item in features:
            assets = item.get('assets', {})
            if asset_name:
                # Return only the specified asset if it exists
                if asset_name in assets:
                    results.append(assets[asset_name])
            else:
                # Return all assets from the item
                results.extend(assets.values())

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
