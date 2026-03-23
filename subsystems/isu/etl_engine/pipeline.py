from typing import Any, Dict, Optional
import pandas as pd

from lib.base import GaiaBase, SubsystemId

from .parsers import ParsingEngine 

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
    
    def __init__(self, project_file: Optional[str] = None, qc_layer: Any = None):
        """
        Initialize the ETLEngine using GaiaBase.
        """
        # Step 1: Initialize base class and register subsystem identity as ISU
        super().__init__(SubsystemId.ISU, project_file=project_file)
        
        # Dependency Injection: Receive the QC layer instance
        self.qc_layer = qc_layer
        
        # Step 2: Utilize self.logger automatically provided by GaiaBase
        self.parsing_engine = ParsingEngine(logger=self.logger)
        self.logger.info("ETL Engine initialized with ParsingEngine.")

    def process_file(self, file_content: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """
        Interface 1: Dedicated for ManualFileLoader and BulkUploadScheduler.
        Processes physical files such as CSV or Excel.

        :param file_content: The raw bytes of the uploaded file.
        :type file_content: bytes
        :param filename: The original name of the file.
        :type filename: str
        :return: A standardized payload dictionary or None if quarantined/failed QC.
        :rtype: Optional[Dict[str, Any]]
        """
        self.logger.info(f"ETL Engine processing file: {filename}")
        
        # Dispatch file to the parsing engine for identification and data extraction
        result = self.parsing_engine.route_and_parse(file_content, filename)
        
        if result.get('status') == 'success':
            # Extract the successfully parsed DataFrame
            df = result.get('data') 
            
            # Assemble data context metadata
            metadata = {
                'source_file': filename,
                'parser_applied': result.get('parser_applied'),
                'confidence': result.get('confidence'),
                'ingestion_mode': 'manual_or_bulk'
            }
            
            # Architectural Compliance: Intercept data via the Quality Control (QC) Layer
            if self.qc_layer:
                # Retrieve expected data type from settings provided by GaiaBase
                target_data_type = self.settings.get('data_type', 'in_situ') if self.settings else 'in_situ'
                
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
                        f"File {filename} failed QC. Reason: {qc_result.get('errors', 'Unknown')}. "
                        "Dropping dataset."
                    )
                    return None  # Drop the data entirely
                    
                elif qc_status == 'Warn':
                    self.logger.warning(
                        f"File {filename} passed QC with warnings: {qc_result.get('warnings', 'None')}. "
                        "Proceeding with caution."
                    )
                    
                elif qc_status == 'Pass':
                    self.logger.info(f"Successfully processed and QC-checked {filename}.")
                    
                else:
                    self.logger.error(f"Unknown QC status '{qc_status}' returned for {filename}.")
            else:
                self.logger.warning("No QC layer configured. Skipping quality control.")
                qc_result = {'final_status': 'Skipped (No QC Layer)'}

            # Return standardized payload for external interfaces (ISU_R_07)
            return {
                'metadata': metadata,
                'qc_result': qc_result,
                'data': df
            }
        else:
            # If the parser cannot identify the format, quarantine the file
            self.logger.warning(f"File {filename} quarantined. Reason: {result.get('reason')}")
            return None

    def process_data(self, df: pd.DataFrame, metadata: Dict[str, Any], qc_result: Dict[str, Any]) -> None:
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
        sensor_type = metadata.get('sensor_type', 'unknown')
        dataset_id = metadata.get('source_topic_or_stream', 'unknown_stream')
        
        self.logger.info(f"ETL Engine received valid streaming payload from {sensor_type} ({dataset_id}).")
        
        # Log data shape and QC status. 
        # Note: In a production environment, this triggers the DPR_I_1 interface 
        # to dispatch data to the external Data Processor.
        self.logger.debug(
            f"Stream data shape: {df.shape}, "
            f"QC Status: {qc_result.get('final_status')}"
        )
        
        # Integration Hook: Place logic here for downstream handoff.
        # Example: self.dpr_service.send_to_processor(df, metadata)
        pass