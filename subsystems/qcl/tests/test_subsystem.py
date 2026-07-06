import os
import pytest

import pandas as pd
from shapely.geometry import box

from subsystems.qcl.layer import QualityControlLoggingLayer
from subsystems.qcl import QCLayer
from subsystems.qcl.asset_checker import AssetChecker

from lib.base import SubsystemId
from tests.utils import TestUtils
from lib.config import ProjectConfigReader


class TestSubsystem:
    def test_QCL_001(self):
        """Test QualityControlLoggingLayer subsystem.
        Example of unit test.
        """
        subsystem = QualityControlLoggingLayer()
        assert subsystem.sid == SubsystemId.QCL

    # 1.Test Cases for insitu
    def test_in_situ_pass(self):
        """Test compliant in-situ data: The final status should be 'Pass'."""
        qc = QCLayer()

        # Construct a perfect DataFrame with pH within 0-14 and unique indices
        df = pd.DataFrame(
            {'ph': [7.2, 6.8, 8.1], 'temperature': [20.1, 19.5, 22.0]},
            index=[101, 102, 103],
        )
        metadata = {'sensor': 'piezometer'}

        result = qc.check('in_situ', df, metadata, 'TEST_ISU_PASS_001')

        assert result['final_status'] == 'Pass'
        assert len(result['errors']) == 0
        assert result['metrics']['missing_values_pct'] == 0.0

    def test_in_situ_fail_ph_range(self):
        """Test data with physical range anomalies: pH > 14 should result in 'Fail'."""
        qc = QCLayer()

        # 15.5 is out of the acceptable physical range (0-14)
        df = pd.DataFrame({'ph': [7.2, 15.5, 8.1]}, index=[101, 102, 103])

        result = qc.check('in_situ', df, {}, 'TEST_ISU_FAIL_001')

        assert result['final_status'] == 'Fail'
        # Verify that the specific error message is present
        assert any('out of range' in e and 'ph' in e.lower() for e in result['errors'])

    def test_in_situ_fail_duplicate_ids(self):
        """Test duplicate identifiers: Duplicated indices should result in 'Fail'."""
        qc = QCLayer()

        # Index 101 is duplicated
        df = pd.DataFrame({'ph': [7.0, 7.1, 7.2]}, index=[101, 101, 103])

        result = qc.check('in_situ', df, {}, 'TEST_ISU_FAIL_002')

        assert result['final_status'] == 'Fail'
        assert any('Duplicate identifiers' in e for e in result['errors'])

    # 2. Test Cases for EO

    def test_eo_raster_pass(self):
        """Test compliant EO raster metadata: The final status should be 'Pass'."""
        qc = QCLayer()
        metadata = {
            'null_pixel_pct': 0.01,
            'is_geometrically_aligned': True,
            'crs': 'EPSG:4326',
            'cloud_cover_pct': 5.0,
            'sensor_type': 'multispectral',
        }

        # Passing None for data_array as current EO rules primarily check metadata
        result = qc.check('eo_raster', None, metadata, 'TEST_EO_PASS_001')
        assert result['final_status'] == 'Pass'

    def test_eo_raster_fail_null_pixels(self):
        """Test EO data with excessive null pixels: Should result in 'Fail'."""
        qc = QCLayer()
        metadata = {
            'null_pixel_pct': 0.05,  # 5% null pixels exceeds the 2% (0.02) limit
            'is_geometrically_aligned': True,
            'crs': 'EPSG:4326',
            'cloud_cover_pct': 5.0,
            'sensor_type': 'multispectral',
        }

        result = qc.check('eo_raster', None, metadata, 'TEST_EO_FAIL_001')

        assert result['final_status'] == 'Fail'
        assert any('exceed 2% limit' in e for e in result['errors'])

    def test_eo_raster_warn_incomplete_metadata(self):
        """Test EO images missing critical metadata: Should result in 'Warn'."""
        qc = QCLayer()
        metadata = {
            'null_pixel_pct': 0.01,
            'is_geometrically_aligned': True,
            # Intentionally omitting 'crs', 'sensor_type', etc.
        }

        result = qc.check('eo_raster', None, metadata, 'TEST_EO_WARN_001')

        assert result['final_status'] == 'Warn'
        assert any('Incomplete EO metadata' in e for e in result['errors'])

    # 3. Exception Handling Tests
    def test_invalid_data_type(self):
        """Test passing an unsupported data type: Should raise a ValueError."""
        qc = QCLayer()

        # Standard pytest way to check if an exception is raised
        with pytest.raises(ValueError) as exc_info:
            qc.check('unknown_type', None, {}, 'TEST_ERR_001')

        assert 'No QC Controller found for unknown_type' in str(exc_info.value)

    def test_db_logging(self):
        """Verify that the logger persists logs to the database"""
        import psycopg
        import logging
        from lib.base import SubsystemId

        os.environ['GAIA_PROJECT_PATH'] = str(
            TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
        )
        project_config = ProjectConfigReader(os.environ['GAIA_PROJECT_PATH'])

        qc = QCLayer()
        log_message = 'Test'
        qc.logger.info(log_message)

        with psycopg.connect(**qc.settings['qcl']['logger']['db']) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT subsystem_id,level_id,message,pid,site_id,project_name FROM log ORDER BY id DESC LIMIT 1'
                )
                row = cur.fetchone()
                # subsystem
                assert row[0] == SubsystemId.QCL.name
                # logging level
                assert row[1] == logging.INFO
                # message
                assert row[2] == log_message
                # pid
                assert row[3] == os.getpid()
                # site_id
                assert row[4] == project_config['project']['site_id']
                # project_name
                assert row[5] == project_config['project']['name']


_S2_ASSET = {
    'platform': 'sentinel-2a',
    'cloud_cover': 5.0,
    'timestamp': '2024-01-01T10:00:00+00:00',
}
_INSITU_ASSET = {
    'type': 'application/x-postgresql',
    'timestamp': '2024-01-01T10:00:00+00:00',
}


class TestAssetChecker:
    def test_AC_001_perfect_score(self):
        """Full-conformance dataset: 3 S2 dates + insitu + low cloud cover should score 100."""
        checker = AssetChecker()
        assets = [
            {**_S2_ASSET, 'cloud_cover': 5.0, 'timestamp': '2024-01-01T10:00:00+00:00'},
            {**_S2_ASSET, 'cloud_cover': 7.0, 'timestamp': '2024-01-15T10:00:00+00:00'},
            {**_S2_ASSET, 'cloud_cover': 3.0, 'timestamp': '2024-02-01T10:00:00+00:00'},
            _INSITU_ASSET,
        ]
        result = checker.analyze_assets(assets, key_variable='amd', verbose=False)
        assert result['conformance'] == 100

    def test_AC_002_missing_insitu_returns_zero(self):
        """Missing required type (insitu) must short-circuit to conformance=0."""
        checker = AssetChecker()
        assets = [_S2_ASSET]
        result = checker.analyze_assets(assets, key_variable='amd', verbose=False)
        assert result['conformance'] == 0
        assert 'insitu' in result['result']['missing_types']

    def test_AC_003_high_cloud_cover_penalised(self):
        """Avg cloud cover >30% should score 0 for cloud criterion."""
        checker = AssetChecker()
        assets = [
            {
                **_S2_ASSET,
                'cloud_cover': 40.0,
                'timestamp': '2024-01-01T10:00:00+00:00',
            },
            {
                **_S2_ASSET,
                'cloud_cover': 50.0,
                'timestamp': '2024-01-15T10:00:00+00:00',
            },
            {
                **_S2_ASSET,
                'cloud_cover': 45.0,
                'timestamp': '2024-02-01T10:00:00+00:00',
            },
            _INSITU_ASSET,
        ]
        result = checker.analyze_assets(assets, key_variable='amd', verbose=False)
        assert result['result']['scores']['cloud_cover'] == 0

    def test_AC_004_insufficient_time_spread(self):
        """Fewer than 3 distinct dates should score 0 for time-spread criterion."""
        checker = AssetChecker()
        assets = [
            {**_S2_ASSET, 'timestamp': '2024-01-01T10:00:00+00:00'},
            {**_S2_ASSET, 'timestamp': '2024-01-01T14:00:00+00:00'},
            _INSITU_ASSET,
        ]
        result = checker.analyze_assets(assets, key_variable='amd', verbose=False)
        assert result['result']['scores']['time_spread'] == 0

    def test_AC_005_empty_assets_returns_zero(self):
        """Empty asset list should return conformance=0 without raising."""
        checker = AssetChecker()
        result = checker.analyze_assets([], key_variable='amd', verbose=False)
        assert result['conformance'] == 0
        assert result['result'] == {}

    def test_AC_006_unknown_key_variable_raises(self):
        """Unrecognised key_variable must raise ValueError."""
        checker = AssetChecker()
        with pytest.raises(ValueError, match='Unknown key_variable'):
            checker.analyze_assets(
                [_S2_ASSET], key_variable='unknown_task', verbose=False
            )


# AOI: 1°×1° box used across map coverage tests
_AOI = box(14.0, 50.0, 15.0, 51.0)

# S2 asset with bbox that fully covers the AOI
_S2_FULL = {**_S2_ASSET, 'bbox': [13.5, 49.5, 15.5, 51.5]}

# S2 asset with bbox that only partially overlaps the AOI (~25%)
_S2_PARTIAL = {**_S2_ASSET, 'bbox': [14.0, 50.0, 14.5, 50.5]}


class TestAssetCheckerCoverage:
    def _base_assets(self, s2_asset):
        """Three dated S2 assets + insitu to satisfy all other scoring rules."""
        return [
            {**s2_asset, 'timestamp': '2024-01-01T10:00:00+00:00'},
            {**s2_asset, 'timestamp': '2024-01-15T10:00:00+00:00'},
            {**s2_asset, 'timestamp': '2024-02-01T10:00:00+00:00'},
            _INSITU_ASSET,
        ]

    def test_AC_007_full_coverage_scores_10(self):
        """Assets whose bboxes fully enclose the AOI should score 10 for map coverage."""
        checker = AssetChecker()
        result = checker.analyze_assets(
            self._base_assets(_S2_FULL), aoi=_AOI, verbose=False
        )
        assert result['result']['scores']['map_coverage'] == 10
        assert result['result']['coverage_ratio'] >= 0.95

    def test_AC_008_partial_coverage_scores_0(self):
        """Assets covering only part of the AOI should score 0 for map coverage."""
        checker = AssetChecker()
        result = checker.analyze_assets(
            self._base_assets(_S2_PARTIAL), aoi=_AOI, verbose=False
        )
        assert result['result']['scores']['map_coverage'] == 0
        assert result['result']['coverage_ratio'] < 0.95

    def test_AC_009_no_aoi_fallback_scores_10(self):
        """When no AOI is provided the coverage check is skipped and scores 10."""
        checker = AssetChecker()
        result = checker.analyze_assets(
            self._base_assets(_S2_PARTIAL), aoi=None, verbose=False
        )
        assert result['result']['scores']['map_coverage'] == 10
        assert result['result']['coverage_ratio'] is None

    def test_AC_010_no_bbox_in_assets_fallback_scores_10(self):
        """When S2 assets have no bbox field the coverage check cannot run and scores 10."""
        checker = AssetChecker()
        assets = self._base_assets(_S2_ASSET)  # _S2_ASSET has no bbox key
        result = checker.analyze_assets(assets, aoi=_AOI, verbose=False)
        assert result['result']['scores']['map_coverage'] == 10
