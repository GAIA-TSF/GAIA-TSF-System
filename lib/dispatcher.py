import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class DispatcherBase(ABC):
    """
    Abstract base class for active output dispatchers.

    Provides shared initialisation and a guard utility so that concrete
    dispatcher implementations only need to define their ``dispatch()`` logic.
    """

    def __init__(self, logger: Any):
        """
        :param logger: Logger instance used for info/warning messages.
        """
        self._logger = logger

    @abstractmethod
    def dispatch(self, *args: Any, **kwargs: Any) -> None:
        """Send data to the target downstream service."""

    def _guard(self, service: Any, interface_id: str) -> bool:
        """
        Check that a downstream service is available before dispatching.

        Logs a warning and returns ``False`` when the service is ``None``,
        allowing callers to exit early with a single ``if not self._guard(...)``
        check.

        :param service: The downstream service instance (may be ``None``).
        :param interface_id: Interface label used in the warning message (e.g. ``'QCL_I_1'``).
        :returns: ``True`` if the service is available, ``False`` otherwise.
        :rtype: bool
        """
        if service is None:
            self._logger.warning(f'[{interface_id}] No service configured. Skipping.')
            return False
        return True


class SdiOutputDispatcher(DispatcherBase):
    """
    QCL_I_1: Actively pushes QC results and quality metrics to SDI after each
    validation cycle. SDI receives a structured record for auditability and
    traceability.
    """

    def dispatch(self, qc_result: Dict[str, Any], sdi_service: Any) -> None:
        """
        Push the QC result record to SDI.

        :param qc_result: Full QC result dict (dataset_id, final_status, metrics, errors).
        :param sdi_service: SDI service instance exposing ``store_qc_result()``.
        """
        if not self._guard(sdi_service, 'QCL_I_1'):
            return
        sdi_service.store_qc_result(qc_result)
        self._logger.info(
            f'[QCL_I_1] QC result for {qc_result.get("dataset_id")} stored in SDI.'
        )


class NotificationDispatcher(DispatcherBase):
    """
    QC_IR_04: Actively triggers the Notification Service (NTF) when QC detects
    failures or warnings. Fires on every Fail or Warn — does not wait to be asked.
    """

    def dispatch(
        self,
        dataset_id: str,
        status: str,
        errors: List[str],
        notification_service: Any,
    ) -> None:
        """
        Send an alert via NTF for Fail or Warn outcomes.

        :param dataset_id: Unique identifier of the dataset that triggered the alert.
        :param status: QC outcome — ``'Fail'`` or ``'Warn'``.
        :param errors: List of error/warning messages to include in the alert.
        :param notification_service: NTF service instance exposing ``send_alert()``.
        """
        if not self._guard(notification_service, 'QC_IR_04'):
            return
        if status == 'Fail':
            notification_service.send_alert(
                dataset_id=dataset_id, errors=errors, severity='critical'
            )
            self._logger.info(f'[QC_IR_04] Critical alert sent for {dataset_id}.')
        elif status == 'Warn':
            notification_service.send_alert(
                dataset_id=dataset_id, errors=errors, severity='warning'
            )
            self._logger.info(f'[QC_IR_04] Warning alert sent for {dataset_id}.')


class VidOutputDispatcher(DispatcherBase):
    """
    QCL_I_2: Actively pushes system health status and QC events to the VID
    (Visualisation Dashboard) after every validation cycle.
    """

    def dispatch(self, qc_result: Dict[str, Any], vid_service: Any) -> None:
        """
        Push a health status event to the VID dashboard.

        :param qc_result: Full QC result dict to derive the status event from.
        :param vid_service: VID service instance exposing ``push_status()``.
        """
        if not self._guard(vid_service, 'QCL_I_2'):
            return
        status_event = {
            'dataset_id': qc_result.get('dataset_id'),
            'status': qc_result.get('final_status'),
            'metrics': qc_result.get('metrics'),
            'errors': qc_result.get('errors'),
            'timestamp': datetime.datetime.now().isoformat(),
        }
        vid_service.push_status(status_event)
        self._logger.info(
            f'[QCL_I_2] Health status for {qc_result.get("dataset_id")} pushed to VID.'
        )


class EouDprDispatcher(DispatcherBase):
    """
    EOU_I_1: Actively forwards newly acquired EO data to the
    Data Processing subsystem (DPR). The dispatcher attempts to call the most
    appropriate ingest method on the DPR service depending on the provided payload
    (keeps behavior defensive to avoid tight coupling). This dispatcher does not
    handle in-situ datasets — EOU is EO-only.
    """

    def dispatch(self, payload: Dict[str, Any], dpr_service: Any) -> None:
        """
        Forward EO payload to DPR.

        :param payload: Dictionary containing keys such as 'data_type', 'file_path', 'metadata', 'dataset_id'.
        :param dpr_service: DPR service instance - expected to expose an ingest method.
        """
        if not self._guard(dpr_service, 'EOU_I_1'):
            return

        dataset_id = payload.get('dataset_id')

        # Preferred: explicit EO ingest method if available
        if hasattr(dpr_service, 'ingest_eo'):
            dpr_service.ingest_eo(payload)
            self._logger.info(
                f'[EOU_I_1] EO payload for {dataset_id} forwarded to DPR.ingest_eo.'
            )
            return

        # Generic hook: look for a generic ingest method
        if hasattr(dpr_service, 'ingest_raw'):
            dpr_service.ingest_raw(payload)
            self._logger.info(
                f'[EOU_I_1] Payload for {dataset_id} forwarded to DPR.ingest_raw.'
            )
            return

        # Last resort: try a catch-all method name used by some DPR implementations
        for candidate in ('receive_raw', 'receive_raw_data', 'ingest'):
            if hasattr(dpr_service, candidate):
                getattr(dpr_service, candidate)(payload)
                self._logger.info(
                    f'[EOU_I_1] Payload for {dataset_id} forwarded to DPR.{candidate}.'
                )
                return

        # If no suitable method found, warn and skip
        self._logger.warning(
            '[EOU_I_1] DPR service does not expose a known ingest API. Payload skipped.'
        )


class EouQcDispatcher(DispatcherBase):
    """
    EOU_I_2: Forwards newly acquired raw data (or metadata references) to the
    Quality Control layer (QCL) for validation. QCL is expected to expose
    ``process_incoming_data(data_type, data, metadata, dataset_id)``.
    """

    def dispatch(
        self,
        data_type: str,
        data: Any,
        metadata: Dict[str, Any],
        dataset_id: str,
        qcl_service: Any,
    ) -> None:
        """
        Send raw data or a pointer to the QC layer.

        :param data_type: Category of the data (e.g., 'in_situ', 'eo_raster').
        :param data: Either the raw payload or a pointer (file path) to the data.
        :param metadata: Associated metadata dict.
        :param dataset_id: Unique dataset identifier.
        :param qcl_service: QCL service instance exposing ``process_incoming_data()``.
        """
        if not self._guard(qcl_service, 'EOU_I_2'):
            return

        if hasattr(qcl_service, 'process_incoming_data'):
            qcl_service.process_incoming_data(data_type, data, metadata, dataset_id)
            self._logger.info(
                f'[EOU_I_2] {dataset_id} forwarded to QCL for validation.'
            )
            return

        # Fallback patterns
        for candidate in ('receive_for_qc', 'validate_raw', 'submit_raw'):
            if hasattr(qcl_service, candidate):
                getattr(qcl_service, candidate)(data_type, data, metadata, dataset_id)
                self._logger.info(
                    f'[EOU_I_2] {dataset_id} forwarded to QCL.{candidate}.'
                )
                return

        self._logger.warning(
            '[EOU_I_2] QCL service does not expose a known validation API. Skipping.'
        )
