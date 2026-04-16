from __future__ import annotations

from lib.base import GaiaBase, SubsystemId


class DataAcquisitionGateway(GaiaBase):
    """Data Acquisition Gateway module serves as the automated
    ingestion engine for the sub-system.
    """

    def __init__(self, backend: str = 'eodag'):
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