import time
import signal
import sys
from eodag import setup_logging

setup_logging(
    verbose=0
)  # Otherwise the eodag progressbar is spamming the logging output
# import os

from subsystems.eou.bulk_upload_scheduler import BulkUploadScheduler
from tests.utils import TestUtils


def main():
    # Define project path
    project_path = TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
    # os.environ['GAIA_PROJECT_PATH'] = str(project_path)  # If running locally to prevent logging KeyError (import os)

    # Instantiate the scheduler
    scheduler = BulkUploadScheduler(project_path=project_path)

    # (Optional) Override settings programmatically if needed for testing/runs:
    # scheduler.scheduler.interval = 120  # Default value 3600 seconds
    # scheduler.quicklook = True  # Default False

    # Set up signal handlers for graceful shutdown (Ctrl+C)
    def signal_handler(sig, frame):
        scheduler.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start the background scanning thread
    scheduler.start()

    # Keep the main thread alive so the daemon thread can continue running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == '__main__':
    main()
