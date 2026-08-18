import os
import shutil
from typing import Any, List, Optional, Tuple


from lib.base import GaiaBase, SubsystemId
from lib.scheduler import BaseScheduler

from .source_fetchers import (
    fetch_from_ftp,
    fetch_from_https,
    fetch_from_s3,
    fetch_from_sftp,
)


class BulkUploadScheduler(GaiaBase):
    """
    GAIA-TSF ISU: Bulk Upload Scheduler Module.

    Periodically scans for new sensor data files and dispatches them to the
    ETL Engine. Depending on `bulk_source_type`, files are either read from a
    local input directory (and archived to a processed directory once handled)
    or fetched remotely over HTTPS, S3, FTP or SFTP.

    :param etl_engine: The central ETL Engine instance for data parsing.
    :type etl_engine: Any
    :param project_path: Optional path to a specific project configuration file.
    :type project_path: Optional[str]
    """

    def __init__(self, etl_engine: Any, project_path: Optional[str] = None):
        """
        Initialize the Bulk Upload Scheduler using GaiaBase.
        """
        # Step 1: Initialize base class and register as an ISU component
        super().__init__(SubsystemId.ISU, project_path=project_path)

        self.etl_engine = etl_engine

        # Step 2: Extract configurations from self.settings provided by GaiaBase
        self.config = self.settings.get('isu', {}) if self.settings else {}
        self.input_dir = self.config.get('input_dir', 'data/input')
        self.processed_dir = self.config.get('processed_dir', 'data/processed')
        interval = self.config.get('bulk_scan_interval_sec', 10)

        # Valid options: 'local', 'https', 's3', 'ftp', 'sftp'
        self.source_type = self.config.get('bulk_source_type', 'local').lower()
        # Tracks remote filenames already dispatched to the ETL Engine, so that
        # repeated scans don't re-fetch and reprocess the same remote files.
        self._seen_remote_files = set()

        # Pass the auto-generated logger to the underlying thread scheduler
        self.scheduler = BaseScheduler(interval_seconds=interval, logger=self.logger)

        if self.source_type == 'local':
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
        Scan for new files at the configured `bulk_source_type` and trigger the ETL pipeline.

        :return: None
        :rtype: None
        """
        if self.source_type == 'local':
            self._scan_local_directory()
        else:
            self._fetch_and_process_remote_files()

    def _scan_local_directory(self) -> None:
        """
        Scan the local input directory for new files and trigger the ETL pipeline.

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

    def _fetch_and_process_remote_files(self) -> None:
        """
        Fetch new files from the configured remote source (HTTPS/S3/FTP/SFTP)
        and dispatch any not seen in a previous scan to the ETL Engine.

        :return: None
        :rtype: None
        """
        self.logger.debug(f'Bulk scanner fetching new files via {self.source_type}...')
        fetched = self._fetch_remote_files()
        new_files = [
            (filename, content)
            for filename, content in fetched
            if filename not in self._seen_remote_files
        ]

        if not new_files:
            return

        self.logger.info(
            f'Fetched {len(new_files)} new files via {self.source_type}. Processing...'
        )
        for filename, content in new_files:
            self._process_fetched_file(filename, content)
            self._seen_remote_files.add(filename)

    def _fetch_remote_files(self) -> List[Tuple[str, bytes]]:
        """
        Dispatch to the fetch function matching the configured `bulk_source_type`.

        :return: List of (filename, content) tuples retrieved from the remote source.
        :rtype: List[Tuple[str, bytes]]
        """
        if self.source_type == 'https':
            return fetch_from_https(
                urls=self.config.get('https_urls', []),
                logger=self.logger,
            )

        if self.source_type == 's3':
            if not self.config.get('s3_bucket'):
                self.logger.error(
                    'Missing s3_bucket in settings. S3 fetch will be DISABLED.'
                )
                return []
            return fetch_from_s3(
                bucket=self.config.get('s3_bucket'),
                prefix=self.config.get('s3_prefix', ''),
                region_name=self.config.get('s3_region'),
                logger=self.logger,
            )

        if self.source_type == 'ftp':
            return fetch_from_ftp(
                host=self.config.get('ftp_host'),
                user=self.config.get('ftp_user'),
                password=self.config.get('ftp_password'),
                remote_dir=self.config.get('ftp_remote_dir', '/'),
                port=self.config.get('ftp_port', 21),
                logger=self.logger,
            )

        if self.source_type == 'sftp':
            return fetch_from_sftp(
                host=self.config.get('sftp_host'),
                user=self.config.get('sftp_user'),
                password=self.config.get('sftp_password'),
                key_path=self.config.get('sftp_key_path'),
                remote_dir=self.config.get('sftp_remote_dir', '.'),
                port=self.config.get('sftp_port', 22),
                logger=self.logger,
            )

        self.logger.error(
            f'Unsupported bulk_source_type: {self.source_type}. No files fetched.'
        )
        return []

    def _process_fetched_file(self, filename: str, content: bytes) -> None:
        """
        Route the content of a remotely fetched file to the ETL Engine.

        :param filename: The name of the fetched file.
        :type filename: str
        :param content: The raw file content.
        :type content: bytes
        :return: None
        :rtype: None
        """
        try:
            df_result = self.etl_engine.process_file(
                file_content=content, filename=filename
            )

            if df_result is not None:
                self.logger.debug(
                    f'File {filename} successfully processed by ETL Engine.'
                )
            else:
                self.logger.warning(
                    f'File {filename} rejected or quarantined by ETL Engine.'
                )

        except Exception as e:
            self.logger.critical(
                f'Unexpected error processing {filename}: {str(e)}', exc_info=True
            )

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
