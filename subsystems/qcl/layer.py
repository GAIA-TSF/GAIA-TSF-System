import datetime
import json
from typing import Dict, Any, List, Tuple

from .logger import Logger

from lib.base import GaiaBase, SubsystemId
from lib.dispatcher import DispatcherBase


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


class RuleRepository:
    """
    Rule Repository module to define thresholds specific to each data type.
    """

    def __init__(self):
        self._rules = {
            'in_situ': {
                'ph_range': {'min': 0, 'max': 14},
                'require_unique_id': True,
            },
            'eo_raster': {
                'max_null_pixels': 0.02,
                'min_snr': 30,
                'require_geo_alignment': True,
            },
        }

    def get_rules(self, data_type: str) -> Dict[str, Any]:
        """
        Retrieves the validation rules for a specific data type.

        :param data_type: The category of the data (e.g., ``'in_situ'``).
        :type data_type: str
        :returns: A dictionary of validation rules.
        :rtype: typing.Dict[str, typing.Any]
        """
        return self._rules.get(data_type, {})


class QcMetrics:
    """
    QC Metrics Engine to compute summary quality indicators.
    """

    def compute_in_situ_metrics(self, df: Any) -> Dict[str, float]:
        """Computes basic quality metrics for in-situ DataFrames."""
        return {
            'missing_values_pct': float(df.isnull().sum().sum() / df.size)
            if df.size > 0
            else 0.0,
        }

    def compute_eo_metrics(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Extracts and computes quality metrics for EO Raster data."""
        return {
            'cloud_cover_pct': metadata.get('cloud_cover_pct', 0.0),
            'null_pixel_pct': metadata.get('null_pixel_pct', 0.0),
        }


class DataLineageLogger:
    """
    Records all applied rules, transformations, filters, and QC actions to ensure
    auditability and compliance with SDI standards.
    """

    def __init__(self, logger_instance: Logger):
        self._logger = logger_instance

    def log(self, dataset_id: str, actions: List[str], status: str) -> None:
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'dataset_id': dataset_id,
            'actions': actions,
            'status': status,
        }
        self._logger.info(f'[QC DB LOG] Lineage recorded: {json.dumps(log_entry)}')


class ErrorNotification:
    """
    Error Reporting & Notification System that triggers alerts.
    """

    def __init__(self, logger_instance: Logger):
        self._logger = logger_instance

    def send_alert(self, dataset_id: str, errors: List[str]) -> None:
        self._logger.error(
            f'[NOTIFICATION] Critical QC Failure on {dataset_id}. Errors: {errors}'
        )


class InSituQualityController:
    """
    Validates in-situ data against physical ranges and unique identifiers.
    """

    def __init__(self, rules: Dict[str, Any], logger_instance: Logger):
        self._rules = rules
        self._logger = logger_instance
        self._metrics_engine = QcMetrics()

    def validate(
        self, df: Any, metadata: Dict[str, Any], dataset_id: str
    ) -> Tuple[str, Dict[str, float], List[str], List[str]]:
        self._logger.info(f'Starting in-situ validation for {dataset_id}')

        status = 'Pass'
        errors = []
        actions_applied = [
            'Missing Values Check',
            'Unique ID Check',
            'Physical Range Check',
        ]

        # 1. Check Unique ID
        if self._rules.get('require_unique_id') and df.index.duplicated().any():
            status = 'Fail'
            errors.append('Duplicate identifiers found.')

        # 2. Check Physical Ranges (e.g., pH)
        if 'ph' in df.columns:
            ph_rule = self._rules.get('ph_range', {'min': 0, 'max': 14})
            if not df['ph'].between(ph_rule['min'], ph_rule['max']).all():
                status = 'Fail'
                errors.append('pH values out of 0-14 physical range.')

        metrics = self._metrics_engine.compute_in_situ_metrics(df)
        return status, metrics, errors, actions_applied


class EoRasterQualityController:
    """
    Validates Earth Observation (EO) raster data.
    """

    def __init__(self, rules: Dict[str, Any], logger_instance: Logger):
        self._rules = rules
        self._logger = logger_instance
        self._metrics_engine = QcMetrics()

    def validate(
        self, data_array: Any, metadata: Dict[str, Any], dataset_id: str
    ) -> Tuple[str, Dict[str, float], List[str], List[str]]:
        self._logger.info(f'Starting EO raster validation for {dataset_id}')

        status = 'Pass'
        errors = []
        actions_applied = [
            'Null Pixel Check',
            'Geometric Alignment Check',
            'Metadata Completeness',
            'SNR Check',
        ]

        # 1. Null Pixel Check
        null_pct = metadata.get('null_pixel_pct', 0)
        if null_pct > self._rules.get('max_null_pixels', 0.02):
            status = 'Fail'
            errors.append(f'Null pixels ({null_pct}) exceed 2% limit.')

        # 2. Geometric Alignment Check
        if self._rules.get('require_geo_alignment') and not metadata.get(
            'is_geometrically_aligned', False
        ):
            status = 'Fail'
            errors.append('Geometric alignment validation failed.')

        # 3. Metadata Completeness
        required_meta = ['crs', 'cloud_cover_pct', 'sensor_type']
        if not all(k in metadata for k in required_meta):
            status = 'Warn'
            errors.append('Incomplete EO metadata.')

        # 4. SNR Check for Hyperspectral
        if metadata.get('sensor_type') == 'hyperspectral':
            if metadata.get('snr', 0) < self._rules.get('min_snr', 30):
                status = 'Fail'
                errors.append('Hyperspectral SNR is below 30.')

        metrics = self._metrics_engine.compute_eo_metrics(metadata)

        # Elevate status to Warn if there are non-critical errors but no failure
        if status != 'Fail' and errors:
            status = 'Warn'

        return status, metrics, errors, actions_applied


class QcCatalog:
    """
    Catalogue of Quality Controllers to route validation based on data type.
    """

    def __init__(self, controllers: Dict[str, Any], logger_instance: Logger):
        self._controllers = controllers
        self._logger = logger_instance

    def route_and_validate(
        self, data_type: str, data: Any, metadata: Dict[str, Any], dataset_id: str
    ) -> Tuple[str, Dict[str, float], List[str], List[str]]:
        if data_type in self._controllers:
            return self._controllers[data_type].validate(data, metadata, dataset_id)

        self._logger.error(f'No QC Controller found for {data_type}')
        raise ValueError(f'No QC Controller found for {data_type}')


class QualityControlLoggingLayer(GaiaBase):
    """
    The Quality Control Layer serves as the critical validation gatekeeper within
    the GAIA-TSF monitoring architecture, situated between the ingestion/ETL
    processes and the Spatial Data Infrastructure (SDI) storage.
    """

    def __init__(
        self,
        sdi_service: Any = None,
        notification_service: Any = None,
        vid_service: Any = None,
    ):
        super().__init__(SubsystemId.QCL)

        self._rule_repo = RuleRepository()
        self._lineage_logger = DataLineageLogger(self.logger)
        self._notifier = ErrorNotification(self.logger)

        self._catalog = QcCatalog(
            {
                'in_situ': InSituQualityController(
                    self._rule_repo.get_rules('in_situ'),
                    self.logger,
                ),
                'eo_raster': EoRasterQualityController(
                    self._rule_repo.get_rules('eo_raster'),
                    self.logger,
                ),
            },
            self.logger,
        )

        # Active output dispatchers (QCL_I_1, QC_IR_04, QCL_I_2)
        self._sdi_dispatcher = SdiOutputDispatcher(logger=self.logger)
        self._notification_dispatcher = NotificationDispatcher(logger=self.logger)
        self._vid_dispatcher = VidOutputDispatcher(logger=self.logger)

        # Injected downstream service references
        self._sdi_service = sdi_service
        self._notification_service = notification_service
        self._vid_service = vid_service

    def process_incoming_data(
        self, data_type: str, data: Any, metadata: Dict[str, Any], dataset_id: str
    ) -> Dict[str, Any]:
        """Automated Validation Mode."""
        self.logger.info(f'Processing incoming data {dataset_id} of type {data_type}')

        # Route to appropriate controller
        status, metrics, errors, actions = self._catalog.route_and_validate(
            data_type, data, metadata, dataset_id
        )

        # Always log lineage
        self._lineage_logger.log(dataset_id, actions, status)

        # Trigger notification on critical failure (legacy internal log)
        if status == 'Fail':
            self._notifier.send_alert(dataset_id, errors)

        qc_result = {
            'dataset_id': dataset_id,
            'final_status': status,
            'metrics': metrics,
            'errors': errors,
        }

        # --- Active outputs ---
        # QCL_I_1: Push QC result to SDI
        self._sdi_dispatcher.dispatch(qc_result, self._sdi_service)

        # QC_IR_04: Trigger NTF alert on Fail or Warn
        self._notification_dispatcher.dispatch(
            dataset_id, status, errors, self._notification_service
        )

        # QCL_I_2: Push health status to VID dashboard
        self._vid_dispatcher.dispatch(qc_result, self._vid_service)

        return qc_result

    def manual_review(
        self, dataset_id: str, action: str, new_status: str = None
    ) -> str:
        """
        Manual Review Interface allowing data managers to inspect, override,
        or re-run QC checks.
        """
        self.logger.info(
            f'Manual review triggered for {dataset_id} with action: {action}'
        )

        if action == 'inspect':
            return f'Inspecting logs for {dataset_id}...'
        elif action == 'override' and new_status:
            self._lineage_logger.log(
                dataset_id, [f'Manual Override to {new_status}'], new_status
            )
            return f'{dataset_id} status overridden to {new_status}.'
        elif action == 're-run':
            self._lineage_logger.log(
                dataset_id, ['Manual QC Re-run Triggered'], 'Pending'
            )
            return f'Re-running QC pipeline for {dataset_id}...'

        self.logger.warning(f'Invalid manual review action provided: {action}')
        return 'Invalid manual review action.'
