import threading
from typing import Any, Callable, Dict, List, Optional

from .stream_handler import (
    KafkaStreamHandler,
    KinesisStreamHandler,
    SensorThingsAPIHandler
)
from lib.base import GaiaBase, SubsystemId

class StreamingDataHandler(GaiaBase):
    """
    Streaming Data Handler can subscribe to live datastreams (Kafka, Kinesis, OGC),
    validate timestamps and units, and apply initial QA/QC checks
    before routing the data to the central ETL Engine.
    """
    def __init__(
        self,
        source_type: str,
        qc_layer: Any,
        etl_callback: Callable,
        config: Optional[Dict[str, Any]] = None,
        kafka_broker: Optional[str] = None,
        kafka_topics: Optional[List[str]] = None,
        kinesis_stream: Optional[str] = None,
        kinesis_region: Optional[str] = None,
        ogc_broker: Optional[str] = None,
        ogc_datastream_id: Optional[str] = None,
    ):
        """
        Initialize the StreamingDataHandler facade / factory.

        :param source_type: The stream protocol to use ('kafka', 'kinesis', or 'sensorthings').
        :type source_type: str
        :param qc_layer: The Quality Control Layer instance for data validation.
        :type qc_layer: Any
        :param etl_callback: The callback function to route validated data to the ETL engine.
        :type etl_callback: Callable
        """
        super().__init__(SubsystemId.ISU)
        self.qc_layer = qc_layer
        self.etl_callback = etl_callback
        self.source_type = source_type.lower()
        self.config = config or {}
        self._thread = None

        # ---------------------------------------------------------
        # Factory Pattern: Instantiate the corresponding underlying stream processor
        # based on source_type. Includes parameter validation to prevent 
        # missing critical configurations.
        # ---------------------------------------------------------
        if self.source_type == 'kafka':
            if not kafka_broker or not kafka_topics:
                raise ValueError("kafka_broker and kafka_topics must be provided for Kafka.")
            self._stream_processor = KafkaStreamHandler(
                broker=kafka_broker,
                topics=kafka_topics,
                logger=self.logger,
                qc_layer=self.qc_layer,
                etl_callback=self.etl_callback,
                config=self.config,
            )

        elif self.source_type == 'kinesis':
            if not kinesis_stream or not kinesis_region:
                raise ValueError("kinesis_stream and kinesis_region must be provided for Kinesis.")
            self._stream_processor = KinesisStreamHandler(
                stream_name=kinesis_stream,
                region_name=kinesis_region,
                logger=self.logger,
                qc_layer=self.qc_layer,
                etl_callback=self.etl_callback,
                config=self.config,
            )

        elif self.source_type == 'sensorthings':
            if not ogc_broker or not ogc_datastream_id:
                raise ValueError("ogc_broker and ogc_datastream_id must be provided for SensorThings.")
            self._stream_processor = SensorThingsAPIHandler(
                broker_url=ogc_broker,
                datastream_id=ogc_datastream_id,
                logger=self.logger,
                qc_layer=self.qc_layer,
                etl_callback=self.etl_callback,
                config=self.config,
            )

        else:
            raise ValueError(f"Unsupported source_type: {self.source_type}. Valid options are 'kafka', 'kinesis', 'sensorthings'.")

    def start(self) -> None:
        """
        Start the streaming listener in a background daemon thread.

        :raises RuntimeError: If the thread fails to start.
        :return: None
        :rtype: None
        """
        if self._thread and self._thread.is_alive():
            self.logger.warning(f'StreamingDataHandler ({self.source_type}) is already running.')
            return

        self.logger.info(f'Starting StreamingDataHandler thread for protocol: {self.source_type}...')
        self._thread = threading.Thread(
            target=self._stream_processor.start_consuming,
            daemon=True,
            name=f'ISU-{self.source_type.capitalize()}-Thread',
        )
        self._thread.start()

    def stop(self) -> None:
        """
        Safely stop the streaming listener and clean up thread resources.

        :return: None
        :rtype: None
        """
        self.logger.info(f'Stopping StreamingDataHandler ({self.source_type})...')
        self._stream_processor.stop()

        if self._thread:
            self._thread.join(timeout=3.0)

        self.logger.info(f'StreamingDataHandler ({self.source_type}) successfully stopped.')

