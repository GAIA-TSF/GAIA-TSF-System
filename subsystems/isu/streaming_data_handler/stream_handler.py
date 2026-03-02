import json
import time
import uuid
from typing import Callable, Any, Dict, List
import pandas as pd

try:
    from kafka import KafkaConsumer
except ImportError:
    KafkaConsumer = None


class KafkaStreamHandler:
    """
    Handles high-frequency, real-time sensor feeds from Kafka.
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
        Initialize the Kafka stream handler.

        :param broker: The Kafka broker address (e.g., 'localhost:9092').
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
        self.broker = broker
        self.topics = topics
        self.logger = logger
        self.qc_layer = qc_layer
        self.etl_callback = etl_callback
        self._is_running = False
        self._consumer = None

        self._initialize_consumer()

    def _initialize_consumer(self) -> None:
        """
        Initialize the Kafka consumer with standard configurations.

        :raises ImportError: Logged as an error if the kafka-python package is missing.
        :return: None
        :rtype: None
        """
        if KafkaConsumer is None:
            self.logger.error('kafka-python is not installed.')
            return

        try:
            self._consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.broker,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id='gaia_tsf_isu_streaming_group',
            )
        except Exception as e:
            self.logger.error(f'Failed to initialize Kafka Consumer: {str(e)}')

    def start_consuming(self) -> None:
        """
        Continuously poll the Kafka topics for new messages.

        Supports the required 1-10 messages per second throughput by
        utilizing a non-blocking timeout.

        :return: None
        :rtype: None
        """
        if not self._consumer:
            self.logger.error('Kafka Consumer not initialized. Cannot stream.')
            return

        self._is_running = True
        self.logger.info(f'Started streaming ingestion on topics: {self.topics}')

        while self._is_running:
            try:
                msg_pack = self._consumer.poll(timeout_ms=1000)
                for topic_partition, messages in msg_pack.items():
                    for message in messages:
                        self._process_message(
                            payload=message.value,
                            topic=message.topic,
                        )
            except Exception as e:
                self.logger.error(
                    f'Error during streaming consumption: {str(e)}', exc_info=True
                )
                time.sleep(1)

    def stop(self) -> None:
        """
        Gracefully shut down the stream consumer and close the connection.

        :return: None
        :rtype: None
        """
        self._is_running = False
        if self._consumer:
            self._consumer.close()
        self.logger.info('Streaming ingestion stopped.')

    def _process_message(
        self,
        payload: Dict[str, Any],
        topic: str,
    ) -> None:
        """
        Process a single message by validating, checking via QC, and routing to ETL.

        :param payload: The decoded JSON payload received from the message broker.
        :type payload: Dict[str, Any]
        :param topic: The Kafka topic from which the message was consumed.
        :type topic: str
        :return: None
        :rtype: None
        """
        dataset_id = f'stream_{uuid.uuid4().hex[:8]}'

        self.logger.info(
            f'Ingesting stream data | Source: {topic} | Mode: Streaming | ID: {dataset_id}'
        )

        try:
            df = pd.DataFrame([payload])

            sensor_type = payload.get('sensor_type', 'unknown')
            data_type = 'in_situ'

            metadata = {
                'sensor_type': sensor_type,
                'topic': topic,
                'ingestion_mode': 'streaming',
                'timestamp': payload.get('timestamp'),
            }

            qc_result = self.qc_layer.check(
                data_type=data_type,
                data=df,
                metadata=metadata,
                dataset_id=dataset_id,
            )

            if qc_result['final_status'] == 'Fail':
                self.logger.warning(
                    f'Data {dataset_id} failed QC. Errors: {qc_result["errors"]}. Dropping payload.'
                )
                return

            self.etl_callback(
                df=df,
                metadata=metadata,
                qc_result=qc_result,
            )
            self.logger.info(f'Data {dataset_id} passed QC and routed to ETL.')

        except Exception as e:
            self.logger.error(
                f'Failed to process stream payload {dataset_id}: {str(e)}'
            )
