import datetime
import json
from typing import Dict, Any, List, Tuple

from .logger import Logger

from lib.base import GaiaBase, SubsystemId
from lib.dispatcher import (
    SdiOutputDispatcher,
    NotificationDispatcher,
    VidOutputDispatcher,
)


_DEFAULT_RULES: Dict[str, Any] = {
    'in_situ': {
        'require_unique_id': True,
        'ph_range': {'min': 0, 'max': 14},
        'do_range': {'min': 0, 'max': 20},
        'conductivity_range': {'min': 0, 'max': 5000},
        'turbidity_range': {'min': 0, 'max': 1000},
        'temperature_range': {'min': -5, 'max': 50},
    },
    'eo_raster': {
        'max_null_pixels': 0.02,
        'min_snr': 30,
        'require_geo_alignment': True,
    },
}

# Mapping from column name keywords to rule keys
_COLUMN_RULE_MAP: Dict[str, str] = {
    'ph': 'ph_range',
    'do': 'do_range',
    'conductivity': 'conductivity_range',
    'turbidity': 'turbidity_range',
    'temperature': 'temperature_range',
}


class RuleRepository:
    """
    Rule Repository module to define thresholds specific to each data type.
    Rules are loaded from config.yaml (qcl.validation_rules) with hardcoded
    defaults as fallback.
    """

    def __init__(self, settings: Any = None):
        config_rules = {}
        if settings:
            config_rules = settings.get('qcl', {}).get('validation_rules', {})
        self._rules = {**_DEFAULT_RULES, **config_rules}

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
            'Unique ID Check',
            'Office Record Detection (metadata or QC=0+PLATFORM=*OFFICE*)',
            'Sensor Fault Detection (QC=0+PLATFORM!=*OFFICE*)',
            'Sensor NaN Check',
            'Physical Range Check',
        ]

        # 1. Check Unique ID
        if self._rules.get('require_unique_id') and df.index.duplicated().any():
            status = 'Fail'
            errors.append('Duplicate identifiers found.')

        # 2. Identify office-interpreted records
        # Primary: read from metadata['ingestion']['data_source'] set by ISU.
        # TODO: ISU to populate data_source field (pending PR review).
        # Fallback: detect via QC=0 + PLATFORM=*OFFICE* columns for datasets
        # that include these fields (e.g. Lukáš's water quality format).
        if metadata.get('ingestion', {}).get('data_source') == 'office_interpreted':
            if status != 'Fail':
                status = 'Warn'
            errors.append(
                'Office-interpreted record (data_source=office_interpreted): '
                'no measurements available, skipping range checks.'
            )
            metrics = self._metrics_engine.compute_in_situ_metrics(df)
            return status, metrics, errors, actions_applied

        qc_col = next((c for c in df.columns if c.upper() == 'QC'), None)
        platform_col = next((c for c in df.columns if c.upper() == 'PLATFORM'), None)

        if qc_col and platform_col:
            office_mask = (df[qc_col] == 0) & df[platform_col].str.contains(
                'OFFICE', case=False, na=False
            )
            fault_mask = (df[qc_col] == 0) & ~df[platform_col].str.contains(
                'OFFICE', case=False, na=False
            )
            sensor_mask = ~office_mask & ~fault_mask
        else:
            office_mask = df.index != df.index  # all False
            fault_mask = df.index != df.index  # all False
            sensor_mask = ~office_mask

        office_count = office_mask.sum()
        if office_count > 0:
            if status != 'Fail':
                status = 'Warn'
            errors.append(
                f'{office_count} office-interpreted row(s) (QC=0, PLATFORM=*OFFICE*): '
                'no measurements available, skipping range checks for these rows.'
            )

        fault_count = fault_mask.sum()
        if fault_count > 0:
            status = 'Fail'
            errors.append(
                f'{fault_count} sensor fault row(s) detected (QC=0, PLATFORM!=*OFFICE*): '
                'invalid or missing sensor data.'
            )

        # 3. Sensor rows: check for unexpected NaN (real data loss)
        sensor_df = df[sensor_mask]
        non_meta_cols = [
            c
            for c in sensor_df.columns
            if not any(
                k in c.lower()
                for k in ('lat', 'lon', 'timestamp', 'date', 'platform', 'qc')
            )
        ]
        if not sensor_df.empty and non_meta_cols:
            nan_rows = sensor_df[non_meta_cols].isnull().any(axis=1).sum()
            if nan_rows > 0:
                if status != 'Fail':
                    status = 'Warn'
                errors.append(
                    f'{nan_rows} sensor row(s) contain missing measurement values.'
                )

        # 4. Physical range checks — sensor rows only, skip NaN
        for col in sensor_df.columns:
            col_lower = col.lower().split(' ')[0]
            rule_key = _COLUMN_RULE_MAP.get(col_lower)
            if rule_key and rule_key in self._rules:
                rule = self._rules[rule_key]
                valid = sensor_df[col].dropna()
                if (
                    not valid.empty
                    and not valid.between(rule['min'], rule['max']).all()
                ):
                    status = 'Fail'
                    errors.append(
                        f'{col} values out of range [{rule["min"]}, {rule["max"]}].'
                    )

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

        self._rule_repo = RuleRepository(settings=self.settings)
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
