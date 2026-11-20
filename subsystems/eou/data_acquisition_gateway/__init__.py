from __future__ import annotations
from pathlib import Path

from lib.exceptions import GaiaConfigError
from lib.base import GaiaBase, SubsystemId


class DataAcquisitionGateway(GaiaBase):
    """Data Acquisition Gateway module serves as the automated
    ingestion engine for the sub-system.
    """

    def __init__(self, backend: str = 'eodag', project_path: Path = None):
        """Initialize Data Acquisition Gateway.

        Currently supported backends:
        - EODAG
        - ASF

        :param str backend: backend to be used for searching and downloading data
        :param Path project_path: path to project confiuguration file or None
        """
        super().__init__(SubsystemId.EOU, project_path)

        if backend == 'eodag':
            from subsystems.eou.data_acquisition_gateway.eodag_backend import (
                EODAGDataAcquisitionBackend as DataAcquisitionBackend,
            )
        elif backend == 'asf':
            from subsystems.eou.data_acquisition_gateway.asf_backend import (
                ASFDataAcquisitionBackend as DataAcquisitionBackend,
            )
        else:
            raise GaiaConfigError(
                f'Unsupported data acquisition backend: {backend}', self.logger
            )

        data_dir = (
            Path(self.settings['storage']['data_dir'])
            if self.project_config is None
            else self.project_config.data_dir()
        )
        self.backend = DataAcquisitionBackend(data_dir=data_dir, logger=self.logger)
        self.backend.set_config(self.settings['eou'][backend])
