from typing import Any, Dict

from .logger import Logger as Logger
from .layer import QualityControlLoggingLayer

from lib.base import GaiaBase, SubsystemId


class QCLayer(GaiaBase):
    """
    GAIA-TSF Unified Quality Control Gateway.

    This class serves as the public-facing interface (Facade) for the
    Quality Control subsystem. It safely encapsulates the complex rule
    routing and lineage logging mechanisms.
    """

    def __init__(self):
        """
        Initializes the QC Layer by setting up the underlying rule repositories,
        catalogs, and loggers invisibly to the end user.
        """
        super().__init__(SubsystemId.QCL)
        self._engine = QualityControlLoggingLayer()

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