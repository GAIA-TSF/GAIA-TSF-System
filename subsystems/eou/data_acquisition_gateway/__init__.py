from __future__ import annotations
import os

from lib.base import GaiaBase, SubsystemId


class DataAcquisitionGateway(GaiaBase):
    """Data Acquisition Gateway module serves as the automated
    ingestion engine for the sub-system.
    """

    def __init__(self, target_dir: str, backend: str = 'eodag'):
        """Initialize Data Acquisition Gateway.

        Currently supported backends:
        - EODAG
        - ASF

        :param str target_dir: target directory (absolute or relative) to store downloaded product
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

        self.backend = DataAcquisitionBackend()

        self.backend.set_config(self.settings['eou'][backend])
        if os.path.isabs(target_dir):
            self.backend.output_dir = target_dir
        else:
            self.backend.output_dir = os.path.join(
                self.settings['storage']['data_dir'], target_dir
            )
