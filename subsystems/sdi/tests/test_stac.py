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

    def test_default_collections_exist(self, stac_url):
        """Verify that default collections were created on startup"""
        # Define expected default collections
        expected_collections = ['sentinel-2', 'sentinel-1', 'in-situ', 'landsat-8']

        # Get all collections
        r = requests.get(f'{stac_url}/collections')
        assert r.status_code == 200, f'Failed to get collections: {r.text}'

        data = r.json()
        assert 'collections' in data, 'Response missing collections field'

        # Extract collection IDs
        existing_collection_ids = [col['id'] for col in data['collections']]

        # Verify each expected collection exists
        for collection_id in expected_collections:
            assert collection_id in existing_collection_ids, (
                f"Default collection '{collection_id}' was not created"
            )

            # Verify individual collection details
            r = requests.get(f'{stac_url}/collections/{collection_id}')
            assert r.status_code == 200, (
                f"Collection '{collection_id}' exists in list but cannot be retrieved"
            )

            collection_data = r.json()
            assert collection_data['id'] == collection_id
            assert 'title' in collection_data
            assert 'description' in collection_data
            assert 'extent' in collection_data

        print(
            f'All {len(expected_collections)} default collections verified successfully'
        )

    def test_default_collection_details(self, stac_url):
        """Verify specific properties of default collections"""
        # Test Sentinel-2 collection
        r = requests.get(f'{stac_url}/collections/sentinel-2')
        assert r.status_code == 200
        sentinel_data = r.json()

        assert sentinel_data['id'] == 'sentinel-2'
        assert sentinel_data['title'] == 'Sentinel-2 Data'
        assert 'extent' in sentinel_data
        assert 'spatial' in sentinel_data['extent']
        assert 'temporal' in sentinel_data['extent']

        # Test Landsat-8 collection
        r = requests.get(f'{stac_url}/collections/landsat-8')
        assert r.status_code == 200
        landsat_data = r.json()

        assert landsat_data['id'] == 'landsat-8'
        assert landsat_data['title'] == 'Landsat 8 Data'
        assert 'extent' in landsat_data
        assert 'spatial' in landsat_data['extent']
        assert 'temporal' in landsat_data['extent']

    def test_create_collection_and_item(self, stac_url):
        """Create and verify a test collection and item"""
        # Create a new collection
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

        # Verify the collection exists
        r = requests.get(f'{stac_url}/collections/{collection_id}')
        assert r.status_code == 200, f'Collection not found: {r.text}'
        data = r.json()
        assert data['id'] == collection_id

        # Create a new item in the collection
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

        # Verify the item is available
        r = requests.get(f'{stac_url}/collections/{collection_id}/items/{item_id}')
        assert r.status_code == 200, f'Item not found: {r.text}'
        item_data = r.json()
        assert item_data['id'] == item_id
        assert item_data['collection'] == collection_id

    def test_default_collections_not_empty(self, stac_url):
        """Verify that at least some default collections exist"""
        r = requests.get(f'{stac_url}/collections')
        assert r.status_code == 200

        data = r.json()
        assert 'collections' in data
        assert len(data['collections']) >= 2, (
            'Expected at least 2 default collections (sentinel-2, landsat-8)'
        )

    def test_default_collections_are_valid_stac(self, stac_url):
        """Verify that default collections conform to STAC spec"""
        expected_collections = ['sentinel-2', 'sentinel-1', 'in-situ', 'landsat-8']

        for collection_id in expected_collections:
            r = requests.get(f'{stac_url}/collections/{collection_id}')
            assert r.status_code == 200

            collection = r.json()


            # Check required STAC Collection fields
            required_fields = [
                'id',
                'type',
                'description',
                'license',
                'extent',
                'links',
            ]
            for field in required_fields:
                assert field in collection, (
                    f"Collection '{collection_id}' missing required field '{field}'"
                )

            # Verify type is Collection
            assert collection['type'] == 'Collection', (
                f"Collection '{collection_id}' has wrong type: {collection.get('type')}"
            )

            # Verify extent structure
            assert 'spatial' in collection['extent']
            assert 'bbox' in collection['extent']['spatial']
            assert 'temporal' in collection['extent']
            assert 'interval' in collection['extent']['temporal']
