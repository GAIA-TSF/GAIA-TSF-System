import datetime
from typing import Optional

from lib.base import GaiaBase, SubsystemId
from lib.scheduler import Scheduler
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway


class BulkUploadScheduler(GaiaBase):
    """
    GAIA-TSF EOU: Bulk Upload Scheduler for Earth Observation Data.

    Periodically searches EO data via the DataAcquisitionGateway
    and downloads missing products to a local target directory.
    """

    def __init__(self, project_path: Optional[str] = None):
        super().__init__(SubsystemId.EOU, project_path=project_path)

        self.gateway = DataAcquisitionGateway()

        eou_settings = self.settings.get('eou', {}) if self.settings else {}
        self.target_dir = eou_settings.get('data_dir', 'sentinel2')
        self.interval = eou_settings.get('scan_interval_seconds', 3600)

        self.base_search_filter = eou_settings.get(
            'search_filter',
            {
                'provider': 'cop_dataspace',
                'productType': 'S2_MSI_L2A',
            },
        )

        # Allow the lookback window size to be configurable (defaulting to a safe 3-day window)
        self.lookback_days = eou_settings.get('lookback_days', 3)
        self.quicklook = False

        self.scheduler = Scheduler(
            interval_seconds=self.interval, logger=self.logger
        )

    def start(self) -> None:
        """Starts the periodic background scanning thread."""
        self.logger.info(
            'Initializing background EO polling cycle (Download-Only mode)...'
        )
        self.scheduler.start(self._download_eo_products)

    def _download_eo_products(self) -> None:
        """The zero-argument wrapper executed periodically by BaseScheduler."""
        try:
            self.logger.info('Starting periodic EO catalog search...')

            if not self.project_config:
                self.logger.error(
                    'Cannot scan catalog: No project configuration loaded.'
                )
                return

            aoi_geom = self.project_config.aoi()
            if not aoi_geom:
                self.logger.error(
                    'No valid AOI geometry found in project configuration. Skipping search.'
                )
                return

            # --- DYNAMIC DATE GENERATION ---
            # Calculated inside the execution loop so it advances every day the daemon runs.
            today = datetime.date.today()
            start_date = today - datetime.timedelta(days=self.lookback_days)

            # Clone base filters and inject the moving time window
            current_filter = dict(self.base_search_filter)
            current_filter['end'] = today.isoformat()
            current_filter['start'] = start_date.isoformat()

            self.logger.info(
                f'Scanning catalog window: {current_filter["start"]} to {current_filter["end"]}'
            )
            # -------------------------------

            # Query remote catalog metadata using the fresh, live dates
            results = self.gateway.backend.search(geom=aoi_geom, **current_filter)

            if len(results) == 0:
                self.logger.info(
                    'No satellite imagery matching filters found for this interval.'
                )
                return

            self.logger.info(
                f'Found {len(results)} potential products. Reconciling with disk...'
            )

            downloaded_paths = self.gateway.backend.download_all(
                results, target_dir=self.target_dir, quicklook=self.quicklook
            )

            if downloaded_paths:
                self.logger.info(
                    f"Successfully synchronized {len(downloaded_paths)} product(s) to '{self.target_dir}'."
                )
            else:
                self.logger.info(
                    'All matching catalog products already exist locally. No product downloaded.'
                )

        except Exception as e:
            self.logger.error(
                f'Error encountered during background EO sync: {str(e)}', exc_info=True
            )

    def stop(self) -> None:
        """Safely terminates the background thread execution cycle."""
        self.logger.info('Stopping EO Bulk Upload Scheduler...')
        self.scheduler.stop()
