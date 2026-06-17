from __future__ import annotations
from pathlib import Path

from lib.base import GaiaBase, SubsystemId


class DataAcquisitionGateway(GaiaBase):
    """Data Acquisition Gateway module serves as the automated
    ingestion engine for the sub-system.
    """

    def __init__(self, backend: str = 'eodag'):
        """Initialize Data Acquisition Gateway.

        Currently supported backends:
        - EODAG
        - ASF

        :param str backend: backend to be used for searching and downloading data
        """
        super().__init__(SubsystemId.EOU)

        if backend == 'eodag':
            from subsystems.eou.data_acquisition_gateway.eodag_backend import (
                EODAGDataAcquisitionBackend as DataAcquisitionBackend,
            )
        elif backend == 'asf':
            from subsystems.eou.data_acquisition_gateway.asf_backend import (
                ASFDataAcquisitionBackend as DataAcquisitionBackend,
            )
        else:
            raise RuntimeError(f'Unsupported data acquisition backend: {backend}')

        self.backend = DataAcquisitionBackend(
            data_dir=Path(self.settings['storage']['data_dir']), logger=self.logger
        )
        self.logger.debug(f'DataAcquisitionGateway initialized with {backend} backend')
        self.backend.set_config(self.settings['eou'][backend])