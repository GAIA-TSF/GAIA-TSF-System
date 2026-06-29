import threading
from typing import Any, Callable, Optional

from .stream_handler import (
    KafkaStreamHandler,
    KinesisStreamHandler,
    SensorThingsAPIHandler,
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
        etl_callback: Callable,
        qc_layer: Any = None,
        project_path: Optional[str] = None,
    ):
        """
        Initialize the StreamingDataHandler facade / factory.
        """
        super().__init__(SubsystemId.ISU, project_path=project_path)

        self.qc_layer = qc_layer
        self.etl_callback = etl_callback
        self._thread = None
        self._stream_processor = None

        self.config = self.settings.get('isu', {}) if self.settings else {}
        self.source_type = self.config.get('stream_source_type', 'kafka').lower()

        if self.source_type == 'kafka':
            kafka_broker = self.config.get('kafka_broker')
            kafka_topics = self.config.get('kafka_topics')
            if not kafka_broker or not kafka_topics:
                self.logger.error(
                    'Missing kafka_broker or kafka_topics in settings. Kafka streaming will be DISABLED.'
                )
                return

            self._stream_processor = KafkaStreamHandler(
                broker=kafka_broker,
                topics=kafka_topics,
                logger=self.logger,
                qc_layer=self.qc_layer,
                etl_callback=self.etl_callback,
                config=self.config,
            )

        elif self.source_type == 'kinesis':
            kinesis_stream = self.config.get('kinesis_stream')
            kinesis_region = self.config.get('kinesis_region')
            if not kinesis_stream or not kinesis_region:
                self.logger.error(
                    'Missing kinesis_stream or kinesis_region in settings. Kinesis streaming will be DISABLED.'
                )
                return

            self._stream_processor = KinesisStreamHandler(
                stream_name=kinesis_stream,
                region_name=kinesis_region,
                logger=self.logger,
                qc_layer=self.qc_layer,
                etl_callback=self.etl_callback,
                config=self.config,
            )

        elif self.source_type == 'sensorthings':
            ogc_broker = self.config.get('ogc_broker')
            ogc_datastream_id = self.config.get('ogc_datastream_id')
            if not ogc_broker or not ogc_datastream_id:
                self.logger.error(
                    'Missing ogc_broker or ogc_datastream_id in settings. SensorThings streaming will be DISABLED.'
                )
                return

            self._stream_processor = SensorThingsAPIHandler(
                broker_url=ogc_broker,
                datastream_id=ogc_datastream_id,
                logger=self.logger,
                qc_layer=self.qc_layer,
                etl_callback=self.etl_callback,
                config=self.config,
            )

        else:
            self.logger.error(
                f'Unsupported source_type: {self.source_type}. Streaming will be DISABLED.'
            )
            return

    def start(self) -> None:
        """
        Start the streaming listener in a background daemon thread.

        :raises RuntimeError: If the thread fails to start.
        :return: None
        :rtype: None
        """
        if not self._stream_processor:
            self.logger.warning(
                f'StreamingDataHandler ({self.source_type}) is disabled. Skipping start.'
            )
            return

        if self._thread and self._thread.is_alive():
            self.logger.warning(
                f'StreamingDataHandler ({self.source_type}) is already running.'
            )
            return

        self.logger.info(
            f'Starting StreamingDataHandler thread for protocol: {self.source_type}...'
        )
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
        if not self._stream_processor:
            return

        self.logger.info(f'Stopping StreamingDataHandler ({self.source_type})...')
        self._stream_processor.stop()

        if self._thread:
            self._thread.join(timeout=3.0)

        self.logger.info(
            f'StreamingDataHandler ({self.source_type}) successfully stopped.'
        )
