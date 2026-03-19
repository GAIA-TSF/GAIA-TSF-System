import os
import shutil

# Import QCL Logger
from subsystems.qcl.logger import Logger
from .parsers import ParsingEngine
from .scheduler import Scheduler
from lib.base import GaiaBase, SubsystemId

class InSituDataUploader(GaiaBase):
    """
    In-Situ Data Uploader sub-system is responsible for collecting
    and securely transmitting field-acquired data.
    """
    def __init__(
        self, input_dir: str = 'data/input', processed_dir: str = 'data/processed'
    ):
        """
        Initialize the InSituDataUploader subsystem.

        :param input_dir: The directory path to monitor for incoming data files.
        :type input_dir: str
        :param processed_dir: The directory path where files are moved after processing.
        :type processed_dir: str
        """
        super().__init__(SubsystemId.ISU)

        self.input_dir = input_dir
        self.processed_dir = processed_dir

        # Initialize sub-components
        self.parsing_engine = ParsingEngine(logger=self.logger)
        self.scheduler = Scheduler(interval_seconds=10)

        self._ensure_directories()
        self.logger.debug('ISU Subsystem components initialized.')

    def _ensure_directories(self):
        """
        Verify the existence of required directories and create them if they are missing.

        :raises OSError: If directory creation fails due to permissions or OS errors.
        :return: None
        """
        for path in [self.input_dir, self.processed_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
                self.logger.info(f'Created directory: {path}')

    def start(self):
        """
        Start the ISU Subsystem and begin the scheduled file monitoring.

        :return: None
        """
        self.logger.info(f'Starting ISU Subsystem (monitoring {self.input_dir})...')
        self.scheduler.start(self._scan_and_process_files)

    def stop(self):
        """
        Gracefully stop the ISU Subsystem and halt scheduled tasks.

        :return: None
        """
        self.scheduler.stop()
        self.logger.info('ISU Subsystem stopped.')

    def _scan_and_process_files(self):
        """
        Scan the input directory for existing files and trigger the processing pipeline.

        :return: None
        """
        self.logger.debug('Scanning for files...')
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

        self.logger.info(f'Found {len(files)} files. Processing...')

        for filename in files:
            file_path = os.path.join(self.input_dir, filename)
            self._process_single_file(file_path, filename)

    def _process_single_file(self, file_path: str, filename: str):
        """
        Read a single file from disk and route it through the parsing engine.

        :param file_path: The absolute or relative path to the file.
        :type file_path: str
        :param filename: The name of the file being processed.
        :type filename: str
        :return: None
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            result = self.parsing_engine.route_and_parse(content, filename)
            status = result.get('status')

            if status == 'success':
                parser_name = result.get('parser_applied')
                row_count = result.get('row_count')
                self.logger.info(
                    f'Successfully parsed {filename} via {parser_name} ({row_count} rows).'
                )
                self._archive_file(filename)
            elif status == 'quarantine':
                self.logger.warning(f'Quarantine {filename}: {result.get("reason")}')
                self._archive_file(filename)
            else:
                self.logger.error(f'Failed to parse {filename}: {result.get("error")}')

        except (OSError, IOError) as e:
            self.logger.error(f'File access error on {filename}: {str(e)}')

    def _archive_file(self, filename: str):
        """
        Move a file from the input directory to the processed directory.

        :param filename: The name of the file to archive.
        :type filename: str
        :return: None
        """
        src_path = os.path.join(self.input_dir, filename)
        dst_path = os.path.join(self.processed_dir, filename)
        try:
            shutil.move(src_path, dst_path)
            self.logger.debug(f'Archived {filename}')
        except (OSError, shutil.Error) as e:
            self.logger.error(f'Failed to move file {filename}: {e}')
