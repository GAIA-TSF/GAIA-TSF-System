from typing import Any, Dict
from .logger import Logger as Logger

from .layer import QualityControlLoggingLayer


class QCLayer:
    """
    GAIA-TSF Unified Quality Control Gateway.

    This class serves as the public-facing interface (Facade) for the
    Quality Control subsystem. It encapsulates rule routing, lineage logging,
    and all active output dispatching.

    Active output interfaces:
    - **QCL_I_1 → SDI**: Quality metrics and results are pushed to SDI after
      every validation cycle via ``sdi_service.store_qc_result()``.
    - **QC_IR_04 → NTF**: Failures and warnings actively trigger the Notification
      Service via ``notification_service.send_alert()``.
    - **QCL_I_2 → VID**: System health status is pushed to the Visualisation
      Dashboard via ``vid_service.push_status()`` after every validation cycle.

    :param sdi_service: Optional SDI service instance (QCL_I_1).
    :type sdi_service: Any
    :param notification_service: Optional Notification Service instance (QC_IR_04).
    :type notification_service: Any
    :param vid_service: Optional VID dashboard service instance (QCL_I_2).
    :type vid_service: Any
    """

    def __init__(
        self,
        sdi_service: Any = None,
        notification_service: Any = None,
        vid_service: Any = None,
    ):
        """
        Initializes the QC Layer and wires active output services.
        """
        self._engine = QualityControlLoggingLayer(
            sdi_service=sdi_service,
            notification_service=notification_service,
            vid_service=vid_service,
        )

    def check(
        self, data_type: str, data: Any, metadata: Dict[str, Any], dataset_id: str
    ) -> Dict[str, Any]:
        """
        Intercepts and validates incoming data against established QC rules.
        Data must pass this validation before being ingested into the SDI.

        :param data_type: The category of the data (e.g., ``'in_situ'``, ``'eo_raster'``).
        :type data_type: str
        :param data: The actual data entity (e.g., a pandas DataFrame or Numpy Array).
        :type data: typing.Any
        :param metadata: A dictionary containing metadata and necessary metrics.
        :type metadata: typing.Dict[str, typing.Any]
        :param dataset_id: The unique identifier for the dataset being processed.
        :type dataset_id: str

        :returns: A dictionary containing the validation results, including ``final_status``
                  (Pass/Warn/Fail), ``metrics``, and ``errors``.
        :rtype: typing.Dict[str, typing.Any]
        """

        return self._engine.process_incoming_data(
            data_type=data_type, data=data, metadata=metadata, dataset_id=dataset_id
        )
