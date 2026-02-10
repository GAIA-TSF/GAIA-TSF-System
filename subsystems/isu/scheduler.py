import threading
import time
import logging
from typing import Callable, Optional

# isu logger
logger = logging.getLogger('gaia.isu.scheduler')


class Scheduler:
    """
    Thread-safe scheduler for running periodic tasks.
    """

    def __init__(self, interval_seconds: int = 60):
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_running = False

    def start(self, task_func: Callable[[], None]) -> None:
        """Start the background scheduler thread."""
        if self._is_running:
            logger.warning('Scheduler is already running.')
            return

        self._stop_event.clear()
        self._is_running = True

        # Run the loop in a separate daemon thread
        self._thread = threading.Thread(
            target=self._run_loop, args=(task_func,), daemon=True
        )
        self._thread.start()
        logger.info(f'Scheduler started with interval: {self.interval}s')

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if not self._is_running:
            return

        logger.info('Stopping scheduler...')
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._is_running = False
        logger.info('Scheduler stopped.')

    def _run_loop(self, task_func: Callable[[], None]) -> None:
        """Internal loop handling execution and sleep."""
        while not self._stop_event.is_set():
            try:
                # Execute the task
                task_func()
            except Exception as e:
                logger.error(f'Scheduled task failed: {str(e)}')

            # Sleep in short bursts to allow quick stopping
            # (Check stop_event every 0.5 seconds)
            for _ in range(int(self.interval * 2)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.5)
