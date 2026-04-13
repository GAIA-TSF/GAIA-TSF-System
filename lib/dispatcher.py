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
            self._logger.warning(
                f'[{interface_id}] No service configured. Skipping.'
            )
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
