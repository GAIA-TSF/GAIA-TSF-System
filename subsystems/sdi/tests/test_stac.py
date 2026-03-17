import requests
import uuid
import pytest

from lib.config import SettingsReader

class TestSTAC:
    @pytest.fixture(scope='class')
    def stac_url(self):
        settings = SettingsReader()
        return settings['sdi']['stac']['url']

    def test_stac_alive(self, stac_url):
        """STAC API is running on /"""
        r = requests.get(stac_url)
        assert r.status_code == 200

    def test_create_collection_and_item(self, stac_url):
        # create a new collection
        collection_id = f'testcollection{uuid.uuid4().hex[:8]}'
        collection_payload = {
            'id': collection_id,
            'title': 'Test Collection',
            'description': 'Collection created for pytest test',
            'extent': {
                'spatial': {'bbox': [[0, 0, 1, 1]]},
                'temporal': {
                    'interval': [['2025-10-01T00:00:00Z', '2025-10-31T23:59:59Z']]
                },
            },
            'license': 'proprietary',
        }

        r = requests.post(f'{stac_url}/collections', json=collection_payload)
        assert r.status_code in (200, 201), f'Failed to create collection: {r.text}'

        # verify the collection exists
        r = requests.get(f'{stac_url}/collections/{collection_id}')
        assert r.status_code == 200, f'Collection not found: {r.text}'
        data = r.json()
        assert data['id'] == collection_id

        # create a new item in the collection
        item_id = f'testitem{uuid.uuid4().hex[:8]}'
        item_payload = {
            'id': item_id,
            'collection': collection_id,
            'geometry': {'type': 'Point', 'coordinates': [0.5, 0.5]},
            'bbox': [0.5, 0.5, 0.5, 0.5],
            'properties': {'datetime': '2025-10-15T12:00:00Z'},
            'links': [],
            'assets': {},
        }

        r = requests.post(
            f'{stac_url}/collections/{collection_id}/items', json=item_payload
        )
        assert r.status_code in (200, 201), f'Failed to create item: {r.text}'

        # verify the item is available
        r = requests.get(f'{stac_url}/collections/{collection_id}/items/{item_id}')
        assert r.status_code == 200, f'Item not found: {r.text}'
        item_data = r.json()
        assert item_data['id'] == item_id
        assert item_data['collection'] == collection_id
