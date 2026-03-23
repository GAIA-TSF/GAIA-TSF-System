import os
from typing import Any, Optional

from lib.base import GaiaBase, SubsystemId


class ManualFileLoader(GaiaBase):
    """
    GAIA-TSF ISU: Manual File Loader Module.

    Handles sensor data files manually uploaded via the system interface or API.
    Once file content is received, it is dispatched directly to the central
    ETL Engine for parsing and standardization. Supports reading from local
    filesystem paths or processing raw byte streams from API endpoints.

    :param etl_engine: The central ETL Engine instance for data parsing.
    :type etl_engine: Any
    :param project_file: Optional path to a specific project configuration file.
    :type project_file: Optional[str]
    """

    def __init__(self, etl_engine: Any, project_file: Optional[str] = None):
        """
        Initialize the Manual File Loader using GaiaBase.
        """
        super().__init__(SubsystemId.ISU, project_file=project_file)

        self.etl_engine = etl_engine
        self.logger.debug('ManualFileLoader initialized successfully via GaiaBase.')

    def load_from_path(self, file_path: str) -> bool:
        """
        Load a file from a local filesystem path and dispatch it to the ETL Engine.

        :param file_path: The absolute or relative path to the manually uploaded file.
        :type file_path: str
        :return: True if the file was successfully processed, False otherwise.
        :rtype: bool
        """
        if not os.path.exists(file_path):
            self.logger.error(f'Manual upload failed: File not found at {file_path}')
            return False

        filename = os.path.basename(file_path)
        self.logger.info(f'Manually loading file: {filename}')

        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            return self.load_from_bytes(content, filename)

        except (OSError, IOError) as e:
            self.logger.error(
                f'File access error during manual load of {filename}: {str(e)}'
            )
            return False

    def load_from_bytes(self, content: bytes, filename: str) -> bool:
        """
        Load a file directly from a byte stream (e.g., from a web API upload)
        and route it to the ETL Engine.

        :param content: The raw bytes of the uploaded file.
        :type content: bytes
        :param filename: The original name of the uploaded file.
        :type filename: str
        :return: True if the file was successfully processed, False otherwise.
        :rtype: bool
        """
        self.logger.debug(
            f'Routing manually uploaded bytes ({len(content)} bytes) '
            f'for {filename} to ETL Engine.'
        )

        try:
            df_result = self.etl_engine.process_file(
                file_content=content, filename=filename
            )

            if df_result is not None:
                self.logger.info(f'Manual upload of {filename} processed successfully.')
                return True
            else:
                self.logger.warning(
                    f'Manual upload of {filename} was rejected or quarantined by ETL Engine.'
                )
                return False

        except Exception as e:
            self.logger.critical(
                f'Unexpected error processing manual upload {filename}: {str(e)}',
                exc_info=True,
            )
            return False
