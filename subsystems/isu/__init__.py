import logging
import os
import shutil
import time
from .parsers import ParsingEngine
from .scheduler import Scheduler

logger = logging.getLogger('gaia.isu')


class InSituDataUploader:
    """
    Main entry point for the In-Situ Data Uploader (ISU) subsystem.
    Orchestrates parsing logic, task scheduling, and file lifecycle management.
    """

    def __init__(self, input_dir: str = 'data/input', processed_dir: str = 'data/processed'):
        """
        Initialize the ISU subsystem.

        Args:
            input_dir: Directory to monitor for new files.
            processed_dir: Directory to archive successfully processed files.
        """
        self.input_dir = input_dir
        self.processed_dir = processed_dir

        # 1. Initialize the Parsing Engine
        self.parsing_engine = ParsingEngine()

        # 2. Initialize the Scheduler (running every 10 seconds)
        self.scheduler = Scheduler(interval_seconds=10)

        # Ensure directories exist
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)
            logger.info(f"Created input directory: {self.input_dir}")

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            logger.info(f"Created processed directory: {self.processed_dir}")

        logger.info("ISU Subsystem initialized.")

    def start(self):
        """Start the subsystem services."""
        logger.info(f"Starting ISU Subsystem (monitoring {self.input_dir})...")
        self.scheduler.start(self._scan_and_process_files)
        print('ISU Subsystem started.')

    def stop(self):
        """Stop the subsystem services."""
        self.scheduler.stop()
        logger.info("ISU Subsystem stopped.")

    def _scan_and_process_files(self):
        """
        The periodic task: Scan -> Parse -> Archive.
        """
        logger.debug("Scanning for files...")

        # Specific exception catching for directory access
        try:
            files = [f for f in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, f))]
        except OSError as e:
            logger.error(f"Failed to access input directory {self.input_dir}: {e}")
            return

        if not files:
            return

        logger.info(f"Found {len(files)} files. Processing...")

        for filename in files:
            file_path = os.path.join(self.input_dir, filename)

            try:
                # 1. Parse
                result_df = self.parsing_engine.parse_file(file_path)

                if result_df is not None and not result_df.empty:
                    logger.info(f"✅ Successfully parsed {filename}: {len(result_df)} rows found.")

                    # 2. Archive (Move file to processed folder)
                    self._archive_file(filename)
                else:
                    logger.warning(f"⚠️ Parsed {filename} but got no data.")

            # Catch specific errors as requested by reviewer
            except (ValueError, OSError) as e:
                logger.error(f"❌ Error processing {filename}: {str(e)}")
            except Exception as e:
                # Fallback for truly unexpected errors to prevent thread death,
                # but logged as critical.
                logger.critical(f"🔥 Unexpected crash processing {filename}: {e}", exc_info=True)

    def _archive_file(self, filename: str):
        """Move the processed file to the archive directory."""
        src_path = os.path.join(self.input_dir, filename)
        dst_path = os.path.join(self.processed_dir, filename)

        try:
            shutil.move(src_path, dst_path)
            logger.info(f"📦 Archived {filename} to {self.processed_dir}")
        except (OSError, shutil.Error) as e:
            # Catch file permission errors or destination existing errors
            logger.error(f"Failed to move file {filename}: {e}")