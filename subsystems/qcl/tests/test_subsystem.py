import pytest
import pandas as pd
from qcl.layer import QualityControlLoggingLayer
from qcl import QCLayer

class TestSubsystem:
    def test_QCL_001(self):
        """Test QualityControlLoggingLayer subsystem.
        Example of unit test.
        """
        subsystem = QualityControlLoggingLayer()
        assert subsystem.id == 'QCL'
    #1.Test Cases for insitu
    def test_in_situ_pass(self):
        """Test compliant in-situ data: The final status should be 'Pass'."""
        qc = QCLayer()
        
        # Construct a perfect DataFrame with pH within 0-14 and unique indices
        df = pd.DataFrame(
            {'ph': [7.2, 6.8, 8.1], 'temperature': [20.1, 19.5, 22.0]}, 
            index=[101, 102, 103]
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
        assert any('pH values out of 0-14' in e for e in result['errors'])

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
            'sensor_type': 'multispectral'
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
            'sensor_type': 'multispectral'
        }
        
        result = qc.check('eo_raster', None, metadata, 'TEST_EO_FAIL_001')
        
        assert result['final_status'] == 'Fail'
        assert any('exceed 2% limit' in e for e in result['errors'])

    def test_eo_raster_warn_incomplete_metadata(self):
        """Test EO images missing critical metadata: Should result in 'Warn'."""
        qc = QCLayer()
        metadata = {
            'null_pixel_pct': 0.01,
            'is_geometrically_aligned': True
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