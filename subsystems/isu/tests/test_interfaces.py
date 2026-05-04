"""
Integration tests for ISU external interfaces.

ISU_I_1 → DPR (DataProcessing): ISU delivers validated, standardised datasets
  to DPR via process_in_situ(). SDI integration is indirect through DPR (ISU_R_07).

ISU_I_2 → QCL (QualityControlLogging): ISU reports ingestion status, validation
  results, and error metadata to QCL as structured events via qc_layer.check().
"""

import pandas as pd
from unittest.mock import MagicMock

from subsystems.isu.etl_engine.pipeline import ETLEngine


# Minimal CSV payload that triggers SlopeStabilityParser (score > 0.6)
_SLOPE_CSV = (
    b'timestamp,displacement\n2024-01-01T00:00:00Z,0.5\n2024-01-02T00:00:00Z,0.6'
)
_SLOPE_FILENAME = 'slope_sensor_data.csv'


class TestInterfaces:
    # ------------------------------------------------------------------
    # ISU_I_1 – DPR (DataProcessor)
    # ISU_IR_02: deliver validated and standardised datasets to DPR
    # ISU_R_07: raw data converted to SDI-compatible objects before handoff
    # ------------------------------------------------------------------

    def test_IF1_001(self):
        """ISU_I_1: ETLEngine delivers a parsed file to DPR via process_in_situ()."""
        mock_dpr = MagicMock()
        mock_dpr.process_in_situ.return_value = {
            'status': 'processed',
            'ready_for_sdi': True,
        }

        engine = ETLEngine(dpr_service=mock_dpr)
        result = engine.process_file(_SLOPE_CSV, _SLOPE_FILENAME)

        mock_dpr.process_in_situ.assert_called_once()
        assert result is not None
        assert result['dpr_result']['ready_for_sdi'] is True

    def test_IF1_002(self):
        """ISU_I_1: ETLEngine delivers streaming data to DPR via process_in_situ().

        Streaming data is written to a temporary CSV file before handoff,
        so DPR receives a file path (str) rather than a DataFrame directly.
        """
        mock_dpr = MagicMock()
        mock_dpr.process_in_situ.return_value = {
            'status': 'processed',
            'ready_for_sdi': True,
        }

        engine = ETLEngine(dpr_service=mock_dpr)

        df = pd.DataFrame([{'timestamp': '2024-01-01', 'displacement': 0.5}])
        metadata = {
            'sensor_type': 'slope',
            'source_topic_or_stream': 'test_stream',
            'ingestion_mode': 'streaming',
        }
        qc_result = {'final_status': 'Pass'}

        engine.process_data(df, metadata, qc_result)

        mock_dpr.process_in_situ.assert_called_once()
        call_args = mock_dpr.process_in_situ.call_args
        file_path_arg = call_args[0][0]
        assert isinstance(file_path_arg, str)
        assert file_path_arg.endswith('.csv')

    # ------------------------------------------------------------------
    # ISU_I_2 – QCL (QualityControlLogging)
    # ISU_IR_03: report ingestion status, errors, and metadata to QCL
    # ------------------------------------------------------------------

    def test_IF2_001(self):
        """ISU_I_2: ETLEngine calls qc_layer.check() with correct data_type and dataset_id."""
        mock_qc = MagicMock()
        mock_qc.check.return_value = {'final_status': 'Pass', 'errors': []}

        engine = ETLEngine(qc_layer=mock_qc)
        engine.process_file(_SLOPE_CSV, _SLOPE_FILENAME)

        mock_qc.check.assert_called_once()
        _, kwargs = mock_qc.check.call_args
        assert kwargs['data_type'] == 'in_situ'
        assert kwargs['dataset_id'] == _SLOPE_FILENAME
