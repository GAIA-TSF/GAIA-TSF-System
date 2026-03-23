import os
import shutil
from typing import Any, Optional


from lib.base import GaiaBase, SubsystemId

from .scheduler import Scheduler


class BulkUploadScheduler(GaiaBase):
    """
    GAIA-TSF ISU: Bulk Upload Scheduler Module.

    Monitors a designated local input directory (or mounted FTP directory)
    periodically in the background. Upon detecting new sensor data files,
    it reads and dispatches them to the ETL Engine. Once processed,
    files are securely moved to an archive (processed) directory.

    :param etl_engine: The central ETL Engine instance for data parsing.
    :type etl_engine: Any
    :param project_file: Optional path to a specific project configuration file.
    :type project_file: Optional[str]
    """

    def __init__(self, etl_engine: Any, project_file: Optional[str] = None):
        """
        Initialize the Bulk Upload Scheduler using GaiaBase.
        """
        # Step 1: Initialize base class and register as an ISU (In-Situ Unit) component
        super().__init__(SubsystemId.ISU, project_file=project_file)

        self.etl_engine = etl_engine

        # Step 2: Extract configurations from self.settings provided by GaiaBase
        isu_config = self.settings.get('isu', {}) if self.settings else {}
        self.input_dir = isu_config.get('input_dir', 'data/input')
        self.processed_dir = isu_config.get('processed_dir', 'data/processed')
        interval = isu_config.get('bulk_scan_interval_sec', 10)

        # Pass the auto-generated logger to the underlying thread scheduler
        self.scheduler = Scheduler(interval_seconds=interval, logger=self.logger)

        self._ensure_directories()
        self.logger.debug('BulkUploadScheduler initialized successfully via GaiaBase.')

    def _ensure_directories(self) -> None:
        """
        Verify the existence of required directories and create them if missing.

        :raises OSError: If directory creation fails due to permissions or OS errors.
        :return: None
        :rtype: None
        """
        for path in [self.input_dir, self.processed_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
                self.logger.info(f'Created directory for bulk upload: {path}')

    def start(self) -> None:
        """
        Start the scheduled background file monitoring task.

        :return: None
        :rtype: None
        """
        self.logger.info(
            f'Bulk Upload Scheduler starting (monitoring {self.input_dir})...'
        )
        self.scheduler.start(self._scan_and_process_files)

    def stop(self) -> None:
        """
        Gracefully stop the scheduled monitoring tasks.

        :return: None
        :rtype: None
        """
        self.scheduler.stop()
        self.logger.info('Bulk Upload Scheduler stopped.')

    def _scan_and_process_files(self) -> None:
        """
        Scan the input directory for new files and trigger the ETL pipeline.

        :return: None
        :rtype: None
        """
        self.logger.debug('Bulk scanner checking for new files...')
        try:
            files = [
                f
                for f in os.listdir(self.input_dir)
                if os.path.isfile(os.path.join(self.input_dir, f))
            ]
        except OSError as e:
            self.logger.error(f'Failed to access input directory {self.input_dir}: {e}')
            return

        if not files:
            return

        self.logger.info(f'Found {len(files)} new files in bulk folder. Processing...')

        for filename in files:
            file_path = os.path.join(self.input_dir, filename)
            self._process_single_file(file_path, filename)

    def _process_single_file(self, file_path: str, filename: str) -> None:
        """
        Read a single file from disk and route its content to the ETL Engine.

        :param file_path: The absolute or relative path to the file.
        :type file_path: str
        :param filename: The name of the file being processed.
        :type filename: str
        :return: None
        :rtype: None
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            # Dispatch the file content to the ETL Engine
            df_result = self.etl_engine.process_file(
                file_content=content, filename=filename
            )

            if df_result is not None:
                self.logger.debug(
                    f'File {filename} successfully processed by ETL Engine.'
                )
                self._archive_file(filename)
            else:
                self.logger.warning(
                    f'File {filename} rejected or quarantined by ETL Engine.'
                )
                self._archive_file(filename)

        except (OSError, IOError) as e:
            self.logger.error(f'File access error on {filename}: {str(e)}')
        except Exception as e:
            self.logger.critical(
                f'Unexpected error processing {filename}: {str(e)}', exc_info=True
            )

    def _archive_file(self, filename: str) -> None:
        """
        Move a file from the input directory to the processed (archive) directory.

        :param filename: The name of the file to archive.
        :type filename: str
        :return: None
        :rtype: None
        """
        src_path = os.path.join(self.input_dir, filename)
        dst_path = os.path.join(self.processed_dir, filename)
        try:
            shutil.move(src_path, dst_path)
            self.logger.debug(f'Archived {filename} to {self.processed_dir}')
        except (OSError, shutil.Error) as e:
            self.logger.error(f'Failed to move file {filename}: {e}')
