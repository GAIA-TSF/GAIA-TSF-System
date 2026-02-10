import logging

# from typing import Optional
from .parsers import ParsingEngine
from .scheduler import Scheduler

logger = logging.getLogger('gaia.isu')


class InSituDataUploader:
    """
    Main entry point for the In-Situ Data Uploader (ISU) subsystem.
    Orchestrates parsing and scheduling.
    """

    def __init__(self):
        """Initialize the ISU subsystem components."""
        #
        self.parsing_engine = ParsingEngine()

        # Initialize the scheduler (default 60s)
        self.scheduler = Scheduler(interval_seconds=60)

        logger.info('ISU Subsystem initialized.')

    def start(self):
        """Start the subsystem services (Scheduler)."""
        logger.info('Starting ISU Subsystem...')

        # Start the scheduler with a wrapper task
        self.scheduler.start(self._scheduled_job)

        print('ISU Subsystem started.')

    def stop(self):
        """Stop the subsystem services."""
        self.scheduler.stop()
        logger.info('ISU Subsystem stopped.')

    def _scheduled_job(self):
        """The task to be executed periodically."""
        # This will later trigger the file scanning logic
        # For now, it just logs a heartbeat
        logger.debug('Scheduler heartbeat: Scanning for new files...')
