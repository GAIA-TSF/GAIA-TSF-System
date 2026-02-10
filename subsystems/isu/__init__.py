import os
import shutil
from typing import Optional

# Import QCL Logger
try:
    from qcl.logger import Logger
except ImportError:
    import logging


    class Logger:
        def __init__(self, subsystem):
            self.logger = logging.getLogger(subsystem)

        def debug(self, msg): self.logger.debug(msg)

        def info(self, msg): self.logger.info(msg)

        def warning(self, msg): self.logger.warning(msg)

        def error(self, msg): self.logger.error(msg)

        def critical(self, msg): self.logger.critical(msg)

from .parsers import ParsingEngine
from .scheduler import Scheduler


class InSituDataUploader:
    """
    In-Situ Data Uploader sub-system is responsible for collecting
    and securely transmitting field-acquired data.
    """

    id = 'ISU'

    def __init__(self, input_dir: str = 'data/input', processed_dir: str = 'data/processed'):
        self.logger = Logger(subsystem=self.id)
        self.logger.debug('ISU Subsystem initializing...')

        self.input_dir = input_dir
        self.processed_dir = processed_dir

        # Initialize sub-components
        self.parsing_engine = ParsingEngine(logger=self.logger)
        self.scheduler = Scheduler(interval_seconds=10)

        self._ensure_directories()
        self.logger.debug('ISU Subsystem components initialized.')

    def _ensure_directories(self):
        for path in [self.input_dir, self.processed_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
                self.logger.info(f"Created directory: {path}")

    def start(self):
        self.logger.info(f"Starting ISU Subsystem (monitoring {self.input_dir})...")
        self.scheduler.start(self._scan_and_process_files)

    def stop(self):
        self.scheduler.stop()
        self.logger.info("ISU Subsystem stopped.")

    def _scan_and_process_files(self):
        self.logger.debug("Scanning for files...")
        try:
            files = [f for f in os.listdir(self.input_dir)
                     if os.path.isfile(os.path.join(self.input_dir, f))]
        except OSError as e:
            self.logger.error(f"Failed to access input directory {self.input_dir}: {e}")
            return

        if not files:
            return

        self.logger.info(f"Found {len(files)} files. Processing...")

        for filename in files:
            file_path = os.path.join(self.input_dir, filename)
            self._process_single_file(file_path, filename)

    def _process_single_file(self, file_path: str, filename: str):
        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            result = self.parsing_engine.route_and_parse(content, filename)
            status = result.get('status')

            if status == 'success':
                parser_name = result.get('parser_applied')
                row_count = result.get('row_count')
                self.logger.info(f"Successfully parsed {filename} via {parser_name} ({row_count} rows).")
                self._archive_file(filename)
            elif status == 'quarantine':
                self.logger.warning(f"Quarantine {filename}: {result.get('reason')}")
                self._archive_file(filename)
            else:
                self.logger.error(f"Failed to parse {filename}: {result.get('error')}")

        except (OSError, IOError) as e:
            self.logger.error(f"File access error on {filename}: {str(e)}")

    def _archive_file(self, filename: str):
        src_path = os.path.join(self.input_dir, filename)
        dst_path = os.path.join(self.processed_dir, filename)
        try:
            shutil.move(src_path, dst_path)
            self.logger.debug(f"Archived {filename}")
        except (OSError, shutil.Error) as e:
            self.logger.error(f"Failed to move file {filename}: {e}")