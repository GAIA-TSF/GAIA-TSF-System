import threading
from typing import Any, Callable, List

from .stream_handler import KafkaStreamHandler


class StreamingDataHandler:
    """
    Streaming Data Handler can subscribe to live datastreams,
    validate timestamps and units, and apply initial QA/QC checks
    before routing the data to the central ETL Engine.
    """

    def __init__(
        self,
        broker: str,
        topics: List[str],
        logger: Any,
        qc_layer: Any,
        etl_callback: Callable,
    ):
        """
        Initialize the StreamingDataHandler facade.

        :param broker: The Kafka broker address.
        :type broker: str
        :param topics: A list of Kafka topics to subscribe to.
        :type topics: List[str]
        :param logger: The injected unified logger instance.
        :type logger: Any
        :param qc_layer: The Quality Control Layer instance for data validation.
        :type qc_layer: Any
        :param etl_callback: The callback function to route validated data to the ETL engine.
        :type etl_callback: Callable
        """
        self.logger = logger
        self.qc_layer = qc_layer
        self.etl_callback = etl_callback
        
        self._stream_processor = KafkaStreamHandler(
            broker=broker,
            topics=topics,
            logger=self.logger,
            qc_layer=self.qc_layer,
            etl_callback=self.etl_callback,
        )
        self._thread = None

    def start(self) -> None:
        """
        Start the streaming listener in a background daemon thread.

        :raises RuntimeError: If the thread fails to start.
        :return: None
        :rtype: None
        """
        if self._thread and self._thread.is_alive():
            self.logger.warning('StreamingDataHandler is already running.')
            return

        self.logger.info('Starting StreamingDataHandler thread...')
        self._thread = threading.Thread(
            target=self._stream_processor.start_consuming,
            daemon=True,
            name='ISU-Streaming-Thread',
        )
        self._thread.start()

    def stop(self) -> None:
        """
        Safely stop the streaming listener and clean up thread resources.

        :return: None
        :rtype: None
        """
        self.logger.info('Stopping StreamingDataHandler...')
        self._stream_processor.stop()
        
        if self._thread:
            self._thread.join(timeout=3.0)
            
        self.logger.info('StreamingDataHandler successfully stopped.')