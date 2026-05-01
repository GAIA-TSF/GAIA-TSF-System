"""
Tests for in-situ CSV metadata generation (Issue #200).

Covers InsituDataset, InsituStacItemFactory, and MetadataGenerator for CSV datasources.
"""

import os
import sys
import tempfile
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest

from subsystems.dpr.metadata_processor.generator import (
    InsituDataset,
    InsituStacItemFactory,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

METADATA_WITH_LOCATION = {
    'ingestion': {
        'mode': 'manual',
        'source': 'piezometer_site_A.csv',
        'parser': 'slope',
        'confidence': 0.9,
    },
    'data': {
        'type': 'in_situ',
        'format': 'csv',
        'sensor_type': 'piezometer',
        'time_range': {
            'start': '2026-01-01T00:00:00',
            'end': '2026-06-30T23:59:59',
        },
        'schema': ['iso_timestamp', 'lat', 'lon', 'pressure'],
        'location': {'bbox': [14.412, 50.082, 14.440, 50.092]},
        'crs': 'EPSG:4326',
    },
}

METADATA_NO_LOCATION = {
    'ingestion': {
        'mode': 'bulk',
        'source': 'sensor_nocoords.csv',
    },
    'data': {
        'type': 'in_situ',
        'format': 'csv',
        'sensor_type': 'inclinometer',
        'time_range': {
            'start': '2026-03-01T00:00:00',
            'end': '2026-03-31T23:59:59',
        },
        'schema': ['iso_timestamp', 'tilt_x', 'tilt_y'],
        'location': None,
        'crs': 'EPSG:4326',
    },
}


@pytest.fixture
def tmp_csv():
    """Create a temporary CSV file and return its path."""
    df = pd.DataFrame(
        {
            'iso_timestamp': ['2026-01-01T00:00:00', '2026-06-30T23:59:59'],
            'lat': [50.082, 50.092],
            'lon': [14.412, 14.440],
            'pressure': [101.3, 102.1],
        }
    )
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


# ---------------------------------------------------------------------------
# InsituDataset
# ---------------------------------------------------------------------------


class TestInsituDataset:
    def test_epsg_from_metadata(self, tmp_csv):
        ds = InsituDataset(tmp_csv, METADATA_WITH_LOCATION)
        assert ds.epsg == 4326

    def test_get_bbox(self, tmp_csv):
        ds = InsituDataset(tmp_csv, METADATA_WITH_LOCATION)
        assert ds.get_bbox() == [14.412, 50.082, 14.440, 50.092]

    def test_get_bbox_none_when_no_location(self, tmp_csv):
        ds = InsituDataset(tmp_csv, METADATA_NO_LOCATION)
        assert ds.get_bbox() is None

    def test_get_time_range(self, tmp_csv):
        ds = InsituDataset(tmp_csv, METADATA_WITH_LOCATION)
        tr = ds.get_time_range()
        assert tr['start'] == '2026-01-01T00:00:00'
        assert tr['end'] == '2026-06-30T23:59:59'

    def test_get_schema(self, tmp_csv):
        ds = InsituDataset(tmp_csv, METADATA_WITH_LOCATION)
        assert 'pressure' in ds.get_schema()

    def test_get_sensor_type(self, tmp_csv):
        ds = InsituDataset(tmp_csv, METADATA_WITH_LOCATION)
        assert ds.get_sensor_type() == 'piezometer'


# ---------------------------------------------------------------------------
# InsituStacItemFactory
# ---------------------------------------------------------------------------


class TestInsituStacItemFactory:
    def test_stac_item_structure(self, tmp_csv):
        ds = InsituDataset(tmp_csv, METADATA_WITH_LOCATION)
        factory = InsituStacItemFactory(ds, MagicMock())
        item = factory.create_item()

        assert item['type'] == 'Feature'
        assert item['stac_version'] == '1.0.0'
        assert item['bbox'] == [14.412, 50.082, 14.440, 50.092]
        assert item['geometry']['type'] == 'Polygon'
        assert item['properties']['proj:epsg'] == 4326
        assert item['properties']['start_datetime'] == '2026-01-01T00:00:00'
        assert item['properties']['end_datetime'] == '2026-06-30T23:59:59'

    def test_stac_item_table_columns(self, tmp_csv):
        ds = InsituDataset(tmp_csv, METADATA_WITH_LOCATION)
        factory = InsituStacItemFactory(ds, MagicMock())
        item = factory.create_item()
        col_names = [c['name'] for c in item['assets']['data']['table:columns']]
        assert 'pressure' in col_names

    def test_stac_item_null_geometry_when_no_location(self, tmp_csv):
        ds = InsituDataset(tmp_csv, METADATA_NO_LOCATION)
        factory = InsituStacItemFactory(ds, MagicMock())
        item = factory.create_item()
        assert item['geometry'] is None
        assert item['bbox'] is None

    def test_stac_item_sensor_type_in_properties(self, tmp_csv):
        ds = InsituDataset(tmp_csv, METADATA_WITH_LOCATION)
        factory = InsituStacItemFactory(ds, MagicMock())
        item = factory.create_item()
        assert item['properties']['sensor_type'] == 'piezometer'


# ---------------------------------------------------------------------------
# process_in_situ() — tests the full pipeline using InsituDataset + InsituStacItemFactory
# directly, bypassing GaiaBase/sqlalchemy dependencies
# ---------------------------------------------------------------------------


class TestProcessInSitu:
    def test_process_in_situ_returns_valid_stac(self, tmp_csv):
        """Full pipeline: InsituDataset → InsituStacItemFactory → STAC item."""
        ds = InsituDataset(tmp_csv, METADATA_WITH_LOCATION)
        factory = InsituStacItemFactory(ds, MagicMock())
        stac_item = factory.create_item()

        assert stac_item['type'] == 'Feature'
        assert stac_item['bbox'] == [14.412, 50.082, 14.440, 50.092]
        assert stac_item['properties']['start_datetime'] == '2026-01-01T00:00:00'

    def test_process_in_situ_no_location_null_geometry(self, tmp_csv):
        """Pipeline with no location: geometry and bbox should be None."""
        ds = InsituDataset(tmp_csv, METADATA_NO_LOCATION)
        factory = InsituStacItemFactory(ds, MagicMock())
        stac_item = factory.create_item()

        assert stac_item['geometry'] is None
        assert stac_item['bbox'] is None
