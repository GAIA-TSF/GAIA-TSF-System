import threading
import time
from typing import Callable, Optional, Any


class Scheduler:
    """
    Thread-safe scheduler for running periodic tasks.
    Supports dependency injection for logging.
    """

    def __init__(self, interval_seconds: int = 60, logger: Any = None):
        """
        Initialize the Scheduler.

        :param interval_seconds: How often to run the task (in seconds).
        :type interval_seconds: int
        :param logger: Injected QCL Logger instance.
        :type logger: Any
        """
        self.interval = interval_seconds
        self.logger = logger

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_running = False

    def start(self, task_func: Callable[[], None]) -> None:
        """
        Start the background scheduler thread.

        :param task_func: The function to be executed periodically.
        :type task_func: Callable[[], None]
        :return: None
        """
        if self._is_running:
            self.logger.warning('Scheduler is already running.')
            return

        self._stop_event.clear()
        self._is_running = True

        # Run the loop in a separate daemon thread
        self._thread = threading.Thread(
            target=self._run_loop, args=(task_func,), daemon=True
        )
        self._thread.start()
        self.logger.info(f'Scheduler started with interval: {self.interval}s')

    def stop(self) -> None:
        """
        Stop the scheduler gracefully.

        :return: None
        """
        if not self._is_running:
            return

        self.logger.info('Stopping scheduler...')
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._is_running = False
        self.logger.info('Scheduler stopped.')

    def _run_loop(self, task_func: Callable[[], None]) -> None:
        """
        Internal loop handling execution and sleep.
        
        :param task_func: The function to execute.
        :type task_func: Callable[[], None]
        :return: None
        """
        while not self._stop_event.is_set():
            try:
                # Execute the task
                task_func()

            # Catch specific errors only (No bare exceptions)
            except (OSError, RuntimeError, ValueError) as e:
                self.logger.error(f'Scheduled task failed: {str(e)}', exc_info=True)

            # Responsive sleep loop
            # Check stop_event frequently (every 0.5s) to allow quick shutdown
            # Calculate steps: 10s interval / 0.5s step = 20 steps
            for _ in range(int(self.interval * 2)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.5)
