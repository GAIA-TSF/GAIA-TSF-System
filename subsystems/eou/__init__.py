from pathlib import Path

from subsystems.eou.manual_file_loader import ManualFileLoader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway

from lib.base import GaiaBase, SubsystemId
from lib.dispatcher import EouDprDispatcher, EouQcDispatcher


class EarthObservationDataUploader(GaiaBase):
    """Earth Observation Data Uploader sub-system is designed to
    manage the acquisition of satellite imagery from both public and
    restricted repositories."""

    def __init__(self):
        super().__init__(SubsystemId.EOU)

        self.manual_file_loader = ManualFileLoader()
        self.data_acquisition_gateway = DataAcquisitionGateway()

        # Active dispatchers for EOU interfaces
        self._dpr_dispatcher = EouDprDispatcher(logger=self.logger)
        self._qcl_dispatcher = EouQcDispatcher(logger=self.logger)

        # Downstream services may be injected later (use set_* methods)
        self.dpr_service = None
        self.qcl_service = None

    def set_dpr_service(self, service):
        """Inject DPR downstream service instance used by EOU_I_1."""
        self.dpr_service = service

    def set_qcl_service(self, service):
        """Inject QCL downstream service instance used by EOU_I_2."""
        self.qcl_service = service

    def forward_to_dpr(self, payload: dict) -> None:
        """Forward newly acquired payload to DPR (EOU_I_1).

        Payload expected keys: data_type, file_path, metadata, dataset_id
        """
        self._dpr_dispatcher.dispatch(payload, self.dpr_service)

    def forward_to_qcl(self, data_type: str, data, metadata: dict, dataset_id: str) -> None:
        """Forward raw data or pointer to QCL for validation (EOU_I_2)."""
        self._qcl_dispatcher.dispatch(data_type, data, metadata, dataset_id, self.qcl_service)

    def ingest_manual_file(self, file_path: str, dataset_id: str = None) -> dict:
        """Helper that performs local file validity check and automatically
        forwards valid EO payloads to DPR and QCL (auto-forward behaviour).

        EOU is EO-only: all files are treated as EO rasters.

        :returns: result dict from ManualFileLoader.check_file_validity()
        """
        result = self.manual_file_loader.check_file_validity(file_path)
        # build a sensible dataset id
        dataset_id = dataset_id or Path(file_path).name

        data_type = 'eo_raster'

        payload = {
            'data_type': data_type,
            'file_path': file_path,
            'metadata': {'validity': result},
            'dataset_id': dataset_id,
        }

        # Auto-forward when file is valid (but still forward metadata when invalid)
        self.forward_to_dpr(payload)
        self.forward_to_qcl(data_type, file_path, payload['metadata'], dataset_id)

        return result
