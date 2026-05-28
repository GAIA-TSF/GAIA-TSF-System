import tempfile
from typing import Any, Dict, Optional
import pandas as pd

from lib.base import GaiaBase, SubsystemId

from .parsers import ParsingEngine

# Keywords used to infer sensor type from filename
_SENSOR_TYPE_FILENAME_KEYWORDS: Dict[str, list] = {
    'piezometer': ['piezo', 'piezometer'],
    'inclinometer': ['inclin', 'tilt'],
    'gnss': ['gnss', 'gps'],
    'insar': ['insar', 'sar', 'egms', 'ground_motion'],
    'sonde': ['sonde', 'hydro'],
    'water_quality': ['water', 'quality', 'wq'],
}

# Fallback: keywords matched against column names
_SENSOR_TYPE_COLUMN_KEYWORDS: Dict[str, list] = {
    'piezometer': [
        'pressure',
        'kpa',
        'pore_water',
        'pore',
        'dataset',
        'kilopascal',
        'pascal',
    ],
    'inclinometer': ['tilt', 'inclin', 'angle', 'deflection'],
    'gnss': ['displacement', 'velocity', 'def_x', 'def_y', 'def_z'],
    'insar': ['coherence', 'los', 'deformation'],
    'sonde': ['ph', 'turbidity', 'conductivity', 'dissolved', 'orp'],
    'water_quality': ['nitrate', 'sulfate', 'tds', 'mg/l'],
}


def _infer_sensor_type(filename: str, columns: Optional[list] = None) -> Optional[str]:
    # Primary: filename keywords
    name = filename.lower()
    for sensor, keywords in _SENSOR_TYPE_FILENAME_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return sensor
    # Fallback: column name keywords
    if columns:
        cols_str = ' '.join(c.lower() for c in columns)
        for sensor, keywords in _SENSOR_TYPE_COLUMN_KEYWORDS.items():
            if any(kw in cols_str for kw in keywords):
                return sensor
    return None


def _extract_location(
    df: pd.DataFrame,
    filename: str = '',
    settings: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    # Primary: lat/lon columns in the CSV
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    if lat_col and lon_col:
        lats = pd.to_numeric(df[lat_col], errors='coerce').dropna()
        lons = pd.to_numeric(df[lon_col], errors='coerce').dropna()
        if not lats.empty and not lons.empty:
            return {'bbox': [lons.min(), lats.min(), lons.max(), lats.max()]}
    # Fallback: site coordinates from settings.yaml
    if settings and filename:
        sites = settings.get('isu', {}).get('sites', {})
        name_lower = filename.lower()
        for site_name, site_config in sites.items():
            if site_name.lower() in name_lower:
                bbox = site_config.get('bbox')
                if bbox:
                    return {'bbox': bbox, 'source': 'settings'}
    return None


def _extract_time_range(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    if 'iso_timestamp' in df.columns:
        ts = pd.to_datetime(df['iso_timestamp'], errors='coerce').dropna()
        if not ts.empty:
            return {'start': ts.min().isoformat(), 'end': ts.max().isoformat()}
    return None


class ETLEngine(GaiaBase):
    """
    GAIA-TSF ISU: ETL Engine Hub.

    The internal core of the In-Situ module, responsible for receiving raw data
    from three primary entry points: Manual, Bulk, and Streaming.
    It orchestrates underlying parsers for data extraction, timestamp
    standardization, and prepares payloads for the external Data Processing module.

    :param project_file: Optional path to a project-specific configuration file.
    :type project_file: Optional[str]
    :param qc_layer: External Quality Control layer instance for data validation.
    :type qc_layer: Any
    """

    def __init__(
        self,
        project_file: Optional[str] = None,
        qc_layer: Any = None,
        dpr_service: Any = None,
    ):
        """
        Initialize the ETLEngine using GaiaBase.
        """
        # Step 1: Initialize base class and register subsystem identity as ISU
        super().__init__(SubsystemId.ISU, project_file=project_file)

        # Dependency Injection: Receive external service instances
        # qc_layer  → ISU_I_2: Quality Control and Logging (QCL)
        # dpr_service → ISU_I_1: Data Processing (DPR); SDI is reached indirectly via DPR
        self.qc_layer = qc_layer
        self.dpr_service = dpr_service

        # Step 2: Utilize self.logger automatically provided by GaiaBase
        encodings = (
            self.settings.get('isu', {}).get('csv_encodings') if self.settings else None
        )
        self.parsing_engine = ParsingEngine(logger=self.logger, encodings=encodings)
        self.logger.info('ETL Engine initialized with ParsingEngine.')

    def process_file(
        self,
        file_content: bytes,
        filename: str,
        ingestion_mode: str = 'manual',
    ) -> Optional[Dict[str, Any]]:
        """
        Interface 1: Dedicated for ManualFileLoader and BulkUploadScheduler.
        Processes physical files such as CSV or Excel.

        :param file_content: The raw bytes of the uploaded file.
        :type file_content: bytes
        :param filename: The original name of the file.
        :type filename: str
        :param ingestion_mode: How the file was ingested ('manual' or 'bulk').
        :type ingestion_mode: str
        :return: A standardized payload dictionary or None if quarantined/failed QC.
        :rtype: Optional[Dict[str, Any]]
        """
        self.logger.info(f'ETL Engine processing file: {filename}')

        # Dispatch file to the parsing engine for identification and data extraction
        result = self.parsing_engine.route_and_parse(file_content, filename)

        if result.get('status') == 'success':
            # Extract the successfully parsed DataFrame
            df = result.get('data')

            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'unknown'
            parser_applied = result.get('parser_applied', '')
            parser_short = (
                'slope'
                if 'slope' in parser_applied.lower()
                else 'water'
                if 'water' in parser_applied.lower()
                else parser_applied
            )

            # Assemble structured metadata
            # TODO: set data_source='office_interpreted' for manually created records
            # once the UI/API layer can pass this information to process_file().
            metadata = {
                'ingestion': {
                    'mode': ingestion_mode,
                    'source': filename,
                    'parser': parser_short,
                    'confidence': result.get('confidence'),
                },
                'data': {
                    'type': 'in_situ',
                    'format': ext,
                    'sensor_type': _infer_sensor_type(filename, list(df.columns)),
                    'collection': _infer_sensor_type(filename, list(df.columns))
                    or 'insitu',
                    'time_range': _extract_time_range(df),
                    'schema': [
                        {
                            'name': c,
                            'type': 'number'
                            if pd.api.types.is_numeric_dtype(df[c])
                            else 'text',
                        }
                        for c in df.columns
                    ],
                    'location': _extract_location(df, filename, self.settings),
                    'crs': 'EPSG:4326',
                },
            }

            # Architectural Compliance: Intercept data via the Quality Control (QC) Layer
            if self.qc_layer:
                # Retrieve expected data type from settings provided by GaiaBase
                target_data_type = (
                    self.settings.get('data_type', 'in_situ')
                    if self.settings
                    else 'in_situ'
                )

                qc_result = self.qc_layer.check(
                    data_type=target_data_type,
                    data=df,
                    metadata=metadata,
                    dataset_id=filename,
                )

                qc_status = qc_result.get('final_status')

                # Handle QC return statuses: Pass, Warn, or Fail
                if qc_status == 'Fail':
                    self.logger.warning(
                        f'File {filename} failed QC. Reason: {qc_result.get("errors", "Unknown")}. '
                        'Dropping dataset.'
                    )
                    return None  # Drop the data entirely

                elif qc_status == 'Warn':
                    self.logger.warning(
                        f'File {filename} passed QC with warnings: {qc_result.get("warnings", "None")}. '
                        'Proceeding with caution.'
                    )

                elif qc_status == 'Pass':
                    self.logger.info(
                        f'Successfully processed and QC-checked {filename}.'
                    )

                else:
                    self.logger.error(
                        f"Unknown QC status '{qc_status}' returned for {filename}."
                    )
            else:
                self.logger.warning('No QC layer configured. Skipping quality control.')
                qc_result = {'final_status': 'Skipped (No QC Layer)'}

            # ISU_I_1: Route validated, standardised dataset to DPR (ISU_IR_02 / ISU_R_07).
            # SDI integration is indirect — DPR is responsible for SDI ingestion.
            if self.dpr_service:
                dpr_result = self.dpr_service.process_in_situ(filename, metadata)
                self.logger.info(
                    f"[ISU_I_1] Dataset '{filename}' delivered to DPR: "
                    f'ready_for_sdi={dpr_result.get("ready_for_sdi")}'
                )
            else:
                self.logger.warning(
                    'No DPR service configured (ISU_I_1). Skipping data processing.'
                )
                dpr_result = {
                    'status': 'Skipped (No DPR Service)',
                    'ready_for_sdi': False,
                }

            # Return standardized payload for external interfaces (ISU_R_07)
            return {
                'metadata': metadata,
                'qc_result': qc_result,
                'dpr_result': dpr_result,
                'data': df,
            }
        else:
            # If the parser cannot identify the format, quarantine the file
            self.logger.warning(
                f'File {filename} quarantined. Reason: {result.get("reason")}'
            )
            return None

    def process_data(
        self, df: pd.DataFrame, metadata: Dict[str, Any], qc_result: Dict[str, Any]
    ) -> None:
        """
        Interface 2: Dedicated for StreamingDataHandler (as an etl_callback).
        Handles high-frequency stream data already converted to DataFrame and
        passed through initial Quality Control (QC).

        :param df: The data payload as a pandas DataFrame.
        :type df: pd.DataFrame
        :param metadata: Contextual metadata (sensor type, stream ID, etc.).
        :type metadata: Dict[str, Any]
        :param qc_result: Results from the preliminary Quality Control check.
        :type qc_result: Dict[str, Any]
        :return: None
        """
        dataset_id = metadata.get('source_topic_or_stream', 'unknown_stream')

        # Enrich streaming metadata with the same structured format
        data_meta = metadata.setdefault('data', {})
        data_meta.setdefault('type', 'in_situ')
        data_meta.setdefault('format', 'stream')
        data_meta.setdefault('schema', list(df.columns))
        data_meta.setdefault('crs', 'EPSG:4326')
        if 'time_range' not in data_meta:
            data_meta['time_range'] = _extract_time_range(df)
        if 'location' not in data_meta:
            data_meta['location'] = _extract_location(df, dataset_id, self.settings)

        ingestion_meta = metadata.setdefault('ingestion', {})
        ingestion_meta.setdefault('mode', 'streaming')
        ingestion_meta.setdefault('source', dataset_id)
        ingestion_meta.setdefault('data_source', 'sensor')

        sensor_type = data_meta.get('sensor_type', 'unknown')

        self.logger.info(
            f'ETL Engine received valid streaming payload from {sensor_type} ({dataset_id}).'
        )
        self.logger.debug(
            f'Stream data shape: {df.shape}, QC Status: {qc_result.get("final_status")}'
        )

        # ISU_I_1: Route validated, standardised streaming dataset to DPR (ISU_IR_02 / ISU_R_07).
        # SDI integration is indirect — DPR is responsible for SDI ingestion.
        if self.dpr_service:
            with tempfile.NamedTemporaryFile(
                suffix='.csv', delete=False, prefix=f'{dataset_id}_'
            ) as tmp:
                tmp_path = tmp.name
            df.to_csv(tmp_path, index=False)
            self.logger.debug(
                f'[ISU_I_1] Streaming data written to temporary file: {tmp_path}'
            )
            dpr_result = self.dpr_service.process_in_situ(tmp_path, metadata)
            self.logger.info(
                f'[ISU_I_1] Stream data from {dataset_id} delivered to DPR: '
                f'ready_for_sdi={dpr_result.get("ready_for_sdi")}'
            )
        else:
            self.logger.warning(
                'No DPR service configured (ISU_I_1). Skipping data processing.'
            )
            dpr_result = {'status': 'Skipped (No DPR Service)', 'ready_for_sdi': False}
