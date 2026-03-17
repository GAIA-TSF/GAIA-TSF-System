import json
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

# --- Optional Dependencies & Specific Exceptions ---
try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError, NoBrokersAvailable
except ImportError:
    KafkaConsumer = None
    # Dummy classes for safe exception catching if kafka is not installed
    class KafkaError(Exception): 
        pass
    class NoBrokersAvailable(Exception): # noqa: N818
        pass

try:
    import boto3
    import botocore.exceptions
except ImportError:
    boto3 = None
    botocore = None

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


class BaseStreamHandler(ABC):
    """
    Abstract base class for all streaming handlers.
    Provides shared initialization, state management, and the central ETL routing pipeline.
    """

    def __init__(self, logger: Any, qc_layer: Any, etl_callback: Callable, config: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.qc_layer = qc_layer
        self.etl_callback = etl_callback
        self.config = config
        self._is_running = False

    @abstractmethod
    def start_consuming(self) -> None:
        """Starts the streaming loop. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops the streaming loop and performs cleanup."""
        self._is_running = False

    def _execute_pipeline(self, payload: Dict[str, Any], source: str, protocol: str, dataset_id: str) -> None:
        """
        Unified processing pipeline: Converts payload to DataFrame, applies QC checks, 
        and routes to the downstream ETL engine.
        """
        try:
            # 1. Structural Validation (Fast Fail)
            if not isinstance(payload, dict):
                raise TypeError(f"Payload must be a dictionary, got {type(payload)}")

            # 2. DataFrame Conversion
            df = pd.DataFrame([payload])

            # 3. Metadata Extraction
            metadata = {
                'sensor_type': payload.get('sensor_type', 'unknown'),
                'source_topic_or_stream': source,
                'ingestion_mode': 'streaming',
                'protocol': protocol,
                'timestamp': payload.get('timestamp'),
            }
            target_data_type = self.config.get('data_type', 'in_situ')

            # 4. Quality Control
            qc_result = self.qc_layer.check(
                data_type=target_data_type,
                data=df,
                metadata=metadata,
                dataset_id=dataset_id,
            )

            if qc_result.get('final_status') == 'Fail':
                self.logger.warning(
                    f'[{dataset_id}] Data failed QC. Errors: {qc_result.get("errors")}. Dropping payload.'
                )
                return

            # 5. Route to ETL
            self.etl_callback(
                df=df,
                metadata=metadata,
                qc_result=qc_result,
            )
            self.logger.info(f'[{dataset_id}] Passed QC and routed to ETL.')

        except TypeError as e:
            self.logger.error(f'[{dataset_id}] Type Error: {e}. Dropping invalid payload.')
        except ValueError as e:
            self.logger.error(f'[{dataset_id}] Value Error (e.g., Pandas conversion failed): {e}')
        except Exception as e:
            # Safety net for unexpected pipeline crashes
            self.logger.critical(f'[{dataset_id}] UNEXPECTED error in processing pipeline: {e}', exc_info=True)


class KafkaStreamHandler(BaseStreamHandler):
    """Handles high-frequency real-time sensor feeds from Apache Kafka."""

    def __init__(self, broker: str, topics: List[str], logger: Any, qc_layer: Any, etl_callback: Callable, config: Optional[Dict[str, Any]] = None):
        super().__init__(logger, qc_layer, etl_callback, config)
        self.broker = broker
        self.topics = topics
        self._consumer = None
        self._initialize_consumer()

    def _initialize_consumer(self) -> None:
        if KafkaConsumer is None:
            self.logger.error('kafka-python is not installed. Aborting Kafka init.')
            return
        
        group_id = self.config.get('kafka_group_id', 'gaia_tsf_isu_streaming_group')
        offset_reset = self.config.get('kafka_auto_offset_reset', 'latest')

        try:
            self._consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.broker,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset=offset_reset,
                enable_auto_commit=True,
                group_id=group_id,
            )
        except NoBrokersAvailable:
            self.logger.critical(f'Kafka cluster at {self.broker} is unreachable.')
        except KafkaError as e:
            self.logger.error(f'Kafka configuration or initialization error: {e}')
        except Exception as e:
            self.logger.critical(f'Unexpected error initializing Kafka: {e}', exc_info=True)

    def start_consuming(self) -> None:
        if not self._consumer:
            self.logger.error('Kafka Consumer not ready. Cannot start consuming.')
            return

        self._is_running = True
        poll_timeout = self.config.get('kafka_poll_timeout_ms', 1000)
        error_backoff = self.config.get('kafka_error_backoff_sec', 1)
        self.logger.info(f'Kafka ingestion started on topics: {self.topics}')

        while self._is_running:
            try:
                msg_pack = self._consumer.poll(poll_timeout)
                for _, messages in msg_pack.items():
                    for message in messages:
                        dataset_id = f'kafka_{uuid.uuid4().hex[:8]}'
                        self._execute_pipeline(
                            payload=message.value, 
                            source=message.topic, 
                            protocol='kafka', 
                            dataset_id=dataset_id
                        )
            # Catching deserialization errors (JSON decode fails inside value_deserializer)
            except json.JSONDecodeError as e:
                self.logger.error(f'Kafka Deserialization Error - Malformed JSON: {e}')
            except KafkaError as e:
                self.logger.error(f'Kafka Polling Error: {e}')
                time.sleep(error_backoff) # Backoff before retrying
            except Exception as e:
                self.logger.critical(f'Unexpected error during Kafka consumption: {e}', exc_info=True)
                time.sleep(error_backoff)

    def stop(self) -> None:
        super().stop()
        if self._consumer:
            self._consumer.close()
        self.logger.info('Kafka streaming stopped.')


class KinesisStreamHandler(BaseStreamHandler):
    """Handles real-time sensor feeds from AWS Kinesis Streams."""

    def __init__(self, stream_name: str, region_name: str, logger: Any, qc_layer: Any, etl_callback: Callable, config: Optional[Dict[str, Any]] = None):
        super().__init__(logger, qc_layer, etl_callback, config)
        self.stream_name = stream_name
        self.region_name = region_name
        self._kinesis_client = None
        self._shard_iterator = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        if boto3 is None:
            self.logger.error('boto3 is not installed. Aborting Kinesis init.')
            return
            
        try:
            self._kinesis_client = boto3.client('kinesis', region_name=self.region_name)
            response = self._kinesis_client.describe_stream(StreamName=self.stream_name)
            shard_id = response['StreamDescription']['Shards'][0]['ShardId']
            iterator_type = self.config.get('kinesis_iterator_type', 'LATEST')
            iter_response = self._kinesis_client.get_shard_iterator(
                StreamName=self.stream_name, 
                ShardId=shard_id, 
                ShardIteratorType=iterator_type
            )
            self._shard_iterator = iter_response['ShardIterator']
            
        except botocore.exceptions.ClientError as e:
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            self.logger.error(f'AWS ClientError during Kinesis init: {error_msg}')
        except botocore.exceptions.BotoCoreError as e:
            self.logger.error(f'AWS BotoCore Error (e.g., missing credentials): {e}')
        except Exception as e:
            self.logger.critical(f'Unexpected error initializing Kinesis: {e}', exc_info=True)

    def start_consuming(self) -> None:
        if not self._kinesis_client or not self._shard_iterator:
            return

        self._is_running = True
        record_limit = self.config.get('kinesis_record_limit', 100)
        loop_sleep = self.config.get('kinesis_loop_sleep_sec', 1)
        throttle_backoff = self.config.get('kinesis_throttle_backoff_sec', 5)
        error_backoff = self.config.get('kinesis_error_backoff_sec', 2)
        self.logger.info(f'Kinesis ingestion started: {self.stream_name}')

        while self._is_running:
            try:
                record_response = self._kinesis_client.get_records(
                    ShardIterator=self._shard_iterator, Limit=record_limit
                )
                self._shard_iterator = record_response['NextShardIterator']

                for record in record_response['Records']:
                    # Explicit try-except for JSON decoding per record
                    try:
                        payload = json.loads(record['Data'].decode('utf-8'))
                        dataset_id = f'kinesis_{uuid.uuid4().hex[:8]}'
                        self._execute_pipeline(
                            payload=payload, 
                            source=self.stream_name, 
                            protocol='kinesis', 
                            dataset_id=dataset_id
                        )
                    except json.JSONDecodeError as e:
                        self.logger.error(f'Kinesis record dropped - Invalid JSON format: {e}')
                    except UnicodeDecodeError as e:
                        self.logger.error(f'Kinesis record dropped - Invalid UTF-8 encoding: {e}')
                
                time.sleep(loop_sleep)  # Respect API rate limits
                
            except botocore.exceptions.ClientError as e:
                error_code = e.response.get('Error', {}).get('Code')
                if error_code == 'ProvisionedThroughputExceededException':
                    self.logger.warning('Kinesis throughput exceeded. Throttling back for 5 seconds...')
                    time.sleep(throttle_backoff)  # Exponential backoff recommended here
                elif error_code == 'ExpiredIteratorException':
                    self.logger.error('Kinesis Shard Iterator expired. Re-initialization needed.')
                    # Note: In production, trigger a re-init of the iterator here
                    self._is_running = False 
                else:
                    self.logger.error(f'Kinesis ClientError: {e}')
                    time.sleep(error_backoff)
            except Exception as e:
                self.logger.critical(f'Unexpected error during Kinesis consumption: {e}', exc_info=True)
                time.sleep(error_backoff)

    def stop(self) -> None:
        super().stop()
        self.logger.info('Kinesis streaming stopped.')


class SensorThingsAPIHandler(BaseStreamHandler):
    """Handles OGC SensorThings API data via MQTT extension."""

    def __init__(self, broker_url: str, datastream_id: str, logger: Any, qc_layer: Any, etl_callback: Callable, config: Optional[Dict[str, Any]] = None):
        super().__init__(logger, qc_layer, etl_callback, config)
        self.broker_url = broker_url
        self.datastream_id = datastream_id
        self._client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        if mqtt is None:
            self.logger.error('paho-mqtt is not installed. Aborting MQTT init.')
            return
            
        try:
            self._client = mqtt.Client()
            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message
        except Exception as e:
            self.logger.error(f'Failed to initialize MQTT Client object: {e}')

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            topic = f"v1.0/Datastreams({self.datastream_id})/Observations"
            client.subscribe(topic)
            self.logger.info(f'MQTT Connected successfully. Subscribed to {topic}')
        else:
            # Standard MQTT connection return codes
            reasons = {
                1: "Incorrect protocol version",
                2: "Invalid client identifier",
                3: "Server unavailable",
                4: "Bad username or password",
                5: "Not authorized"
            }
            reason = reasons.get(rc, "Unknown reason")
            self.logger.error(f'MQTT Connection failed with code {rc}: {reason}')

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            dataset_id = f'ogc_{uuid.uuid4().hex[:8]}'
            self._execute_pipeline(
                payload=payload, 
                source=f'Datastream_{self.datastream_id}', 
                protocol='sensorthings', 
                dataset_id=dataset_id
            )
        except json.JSONDecodeError as e:
            self.logger.error(f'MQTT message dropped - Invalid JSON: {e}')
        except UnicodeDecodeError as e:
            self.logger.error(f'MQTT message dropped - Invalid UTF-8: {e}')
        except Exception as e:
            self.logger.critical(f'Unexpected error processing MQTT message: {e}', exc_info=True)

    def start_consuming(self) -> None:
        if not self._client:
            return
            
        self._is_running = True
        default_port = self.config.get('mqtt_default_port', 1883)
        keepalive = self.config.get('mqtt_keepalive_sec', 60)
        loop_sleep = self.config.get('mqtt_loop_sleep_sec', 1)

        try:
            parts = self.broker_url.split(':')
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else default_port
        except ValueError as e:
            self.logger.error(f'Invalid MQTT broker URL format "{self.broker_url}": {e}')
            return
            
        try:
            self._client.connect(host, port, keepalive)
            self._client.loop_start()  # Run network loop in a background thread
            
            while self._is_running:
                time.sleep(loop_sleep) # Keep main thread alive
                
        except ConnectionRefusedError:
            self.logger.error(f'Connection refused by MQTT broker at {host}:{port}')
        except Exception as e:
            self.logger.critical(f'Failed to start MQTT loop: {e}', exc_info=True)

    def stop(self) -> None:
        super().stop()
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
        self.logger.info('SensorThings MQTT stopped.')