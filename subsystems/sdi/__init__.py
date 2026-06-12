from typing import Any, Dict

from lib.base import GaiaBase, SubsystemId


class SpatialDataInfrastructure(GaiaBase):
    """SpatialDataInfrastructure sub-system"""

    def __init__(self):
        super().__init__(SubsystemId.SDI)

    def store_qc_result(self, qc_result: Dict[str, Any]) -> None:
        """
        QCL_I_1: Receive and persist a QC result record from QCL.

        :param dict qc_result: QC result containing dataset_id, final_status, metrics, errors.
        """
        dataset_id = qc_result.get('dataset_id', 'unknown')
        status = qc_result.get('final_status', 'unknown')
        self.logger.info(f'[SDI] QC result stored: {dataset_id} → {status}')
