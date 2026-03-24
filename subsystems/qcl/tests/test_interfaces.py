"""
Integration tests for QCL active output interfaces.

QCL_I_1  → SDI: QC results are pushed to SDI after every validation cycle.
QC_IR_04 → NTF: Notification Service is triggered on Fail and Warn outcomes.
QCL_I_2  → VID: Health status events are pushed to the VID dashboard.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, call

from subsystems.qcl import QCLayer


_PASS_DF = pd.DataFrame(
    {'ph': [7.0, 6.8], 'temperature': [20.0, 21.0]},
    index=[1, 2],
)
_FAIL_DF = pd.DataFrame({'ph': [7.0, 15.5]}, index=[1, 2])  # pH > 14


class TestActiveOutputInterfaces:

    # ------------------------------------------------------------------
    # QCL_I_1 – SDI
    # ------------------------------------------------------------------

    def test_QCL_I1_001_sdi_called_on_pass(self):
        """QCL_I_1: sdi_service.store_qc_result() is called after a Pass."""
        mock_sdi = MagicMock()
        qc = QCLayer(sdi_service=mock_sdi)

        qc.check('in_situ', _PASS_DF, {}, 'TEST_SDI_PASS')

        mock_sdi.store_qc_result.assert_called_once()
        result_arg = mock_sdi.store_qc_result.call_args[0][0]
        assert result_arg['final_status'] == 'Pass'
        assert result_arg['dataset_id'] == 'TEST_SDI_PASS'

    def test_QCL_I1_002_sdi_called_on_fail(self):
        """QCL_I_1: sdi_service.store_qc_result() is called even after a Fail."""
        mock_sdi = MagicMock()
        qc = QCLayer(sdi_service=mock_sdi)

        qc.check('in_situ', _FAIL_DF, {}, 'TEST_SDI_FAIL')

        mock_sdi.store_qc_result.assert_called_once()
        result_arg = mock_sdi.store_qc_result.call_args[0][0]
        assert result_arg['final_status'] == 'Fail'

    def test_QCL_I1_003_no_sdi_no_error(self):
        """QCL_I_1: QC works normally when no SDI service is configured."""
        qc = QCLayer()  # no sdi_service
        result = qc.check('in_situ', _PASS_DF, {}, 'TEST_SDI_NONE')
        assert result['final_status'] == 'Pass'

    # ------------------------------------------------------------------
    # QC_IR_04 – NTF (Notification Service)
    # ------------------------------------------------------------------

    def test_QC_IR04_001_alert_on_fail(self):
        """QC_IR_04: notification_service.send_alert() is called with severity='critical' on Fail."""
        mock_ntf = MagicMock()
        qc = QCLayer(notification_service=mock_ntf)

        qc.check('in_situ', _FAIL_DF, {}, 'TEST_NTF_FAIL')

        mock_ntf.send_alert.assert_called_once()
        kwargs = mock_ntf.send_alert.call_args[1]
        assert kwargs['severity'] == 'critical'
        assert kwargs['dataset_id'] == 'TEST_NTF_FAIL'

    def test_QC_IR04_002_alert_on_warn(self):
        """QC_IR_04: notification_service.send_alert() is called with severity='warning' on Warn."""
        mock_ntf = MagicMock()
        qc = QCLayer(notification_service=mock_ntf)

        # EO raster with incomplete metadata triggers Warn
        metadata = {'null_pixel_pct': 0.01, 'is_geometrically_aligned': True}
        qc.check('eo_raster', None, metadata, 'TEST_NTF_WARN')

        mock_ntf.send_alert.assert_called_once()
        kwargs = mock_ntf.send_alert.call_args[1]
        assert kwargs['severity'] == 'warning'

    def test_QC_IR04_003_no_alert_on_pass(self):
        """QC_IR_04: notification_service.send_alert() is NOT called on Pass."""
        mock_ntf = MagicMock()
        qc = QCLayer(notification_service=mock_ntf)

        qc.check('in_situ', _PASS_DF, {}, 'TEST_NTF_PASS')

        mock_ntf.send_alert.assert_not_called()

    # ------------------------------------------------------------------
    # QCL_I_2 – VID (Visualisation Dashboard)
    # ------------------------------------------------------------------

    def test_QCL_I2_001_vid_receives_status_event(self):
        """QCL_I_2: vid_service.push_status() is called with a structured event."""
        mock_vid = MagicMock()
        qc = QCLayer(vid_service=mock_vid)

        qc.check('in_situ', _PASS_DF, {}, 'TEST_VID_001')

        mock_vid.push_status.assert_called_once()
        event = mock_vid.push_status.call_args[0][0]
        assert event['dataset_id'] == 'TEST_VID_001'
        assert event['status'] == 'Pass'
        assert 'metrics' in event
        assert 'timestamp' in event

    def test_QCL_I2_002_vid_called_on_every_outcome(self):
        """QCL_I_2: vid_service.push_status() is called regardless of Pass/Fail/Warn."""
        mock_vid = MagicMock()
        qc = QCLayer(vid_service=mock_vid)

        qc.check('in_situ', _PASS_DF, {}, 'DS_PASS')
        qc.check('in_situ', _FAIL_DF, {}, 'DS_FAIL')

        assert mock_vid.push_status.call_count == 2
