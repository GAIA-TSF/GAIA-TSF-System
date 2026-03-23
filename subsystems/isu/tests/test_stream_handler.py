import pytest
import pandas as pd
import time
from unittest.mock import MagicMock, patch

from subsystems.isu.streaming_data_handler.stream_handler import (
    KafkaStreamHandler,
    KinesisStreamHandler,
    SensorThingsAPIHandler,
)
from subsystems.isu.streaming_data_handler import StreamingDataHandler


@pytest.fixture
def mock_logger() -> MagicMock:
    """Fixture to provide a mocked logger instance."""
    return MagicMock()

@pytest.fixture
def mock_qc_layer() -> MagicMock:
    """Fixture to provide a mocked Quality Control (QC) layer."""
    return MagicMock()

@pytest.fixture
def mock_etl_callback() -> MagicMock:
    """Fixture to provide a mocked ETL callback function."""
    return MagicMock()

@pytest.fixture
def sample_payload() -> dict:
    """Fixture providing a standard sensor data payload for testing."""
    return {
        'sensor_type': 'piezometer',
        'timestamp': '2026-02-27T12:00:00Z',
        'pressure_kpa': 120.5,
        'temperature_c': 15.2,
    }


class TestKafkaStreamHandler:
    """Tests for the Kafka-specific streaming data handler."""

    @patch('subsystems.isu.streaming_data_handler.stream_handler.KafkaConsumer')
    def test_STR_001_initialization(
        self,
        mock_kafka_consumer: MagicMock,
        mock_logger: MagicMock,
        mock_qc_layer: MagicMock,
        mock_etl_callback: MagicMock,
    ) -> None:
        """
        Verify that the Kafka handler initializes the consumer with correct parameters.
        """
        test_config = {'kafka_group_id': 'test_group'}
        handler = KafkaStreamHandler(
            broker='localhost:9092',
            topics=['slope_stability'],
            logger=mock_logger,
            qc_layer=mock_qc_layer,
            etl_callback=mock_etl_callback,
            config=test_config,
        )

        assert handler.broker == 'localhost:9092'
        mock_kafka_consumer.assert_called_once()

    @patch('subsystems.isu.streaming_data_handler.stream_handler.KafkaConsumer')
    def test_STR_002_process_message_pass_qc(
        self, mock_kafka_consumer, mock_logger, mock_qc_layer, mock_etl_callback, sample_payload
    ) -> None:
        """
        Ensure messages that pass Quality Control are forwarded to the ETL callback.
        """
        mock_qc_layer.check.return_value = {'final_status': 'Pass', 'metrics': {}, 'errors': []}
        handler = KafkaStreamHandler(
            broker='localhost', topics=['t1'], logger=mock_logger,
            qc_layer=mock_qc_layer, etl_callback=mock_etl_callback, config={}
        )
        handler._execute_pipeline(sample_payload, 't1', 'kafka', 'test_pass')
        mock_qc_layer.check.assert_called_once()
        mock_etl_callback.assert_called_once()

    @patch('subsystems.isu.streaming_data_handler.stream_handler.KafkaConsumer')
    def test_STR_003_process_message_fail_qc(
        self, mock_kafka_consumer, mock_logger, mock_qc_layer, mock_etl_callback, sample_payload
    ) -> None:
        """
        Ensure messages that fail Quality Control are dropped and logged.
        """
        mock_qc_layer.check.return_value = {'final_status': 'Fail', 'metrics': {}, 'errors': ['Error']}
        handler = KafkaStreamHandler(
            broker='localhost', topics=['t1'], logger=mock_logger,
            qc_layer=mock_qc_layer, etl_callback=mock_etl_callback, config={}
        )
        handler._execute_pipeline(sample_payload, 't1', 'kafka', 'test_fail')
        mock_etl_callback.assert_not_called()
        mock_logger.warning.assert_called()


class TestKinesisStreamHandler:
    """Tests for the AWS Kinesis-specific streaming data handler."""

    @patch('subsystems.isu.streaming_data_handler.stream_handler.boto3')
    def test_STR_004_initialization(
        self, mock_boto3, mock_logger, mock_qc_layer, mock_etl_callback
    ) -> None:
        """
        Verify Kinesis handler correctly initializes the Boto3 client and shard iterators.
        """
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.describe_stream.return_value = {
            'StreamDescription': {'Shards': [{'ShardId': 'shardId-0000'}]}
        }
        mock_client.get_shard_iterator.return_value = {'ShardIterator': 'test_iterator_string'}

        handler = KinesisStreamHandler(
            stream_name='tsf_stream', region_name='eu-central-1', logger=mock_logger,
            qc_layer=mock_qc_layer, etl_callback=mock_etl_callback, config={'kinesis_iterator_type': 'LATEST'}
        )
        assert handler._shard_iterator == 'test_iterator_string'
        mock_boto3.client.assert_called_once_with('kinesis', region_name='eu-central-1')

    @patch('subsystems.isu.streaming_data_handler.stream_handler.boto3')
    def test_STR_005_process_message_fail_qc(
        self, mock_boto3, mock_logger, mock_qc_layer, mock_etl_callback, sample_payload
    ) -> None:
        """
        Verify Kinesis handler drops data upon QC failure.
        """
        mock_qc_layer.check.return_value = {'final_status': 'Fail', 'errors': ['Anomaly']}
        handler = KinesisStreamHandler(
            stream_name='tsf', region_name='eu', logger=mock_logger,
            qc_layer=mock_qc_layer, etl_callback=mock_etl_callback, config={}
        )
        handler._execute_pipeline(sample_payload, 'tsf', 'kinesis', 'fail_001')
        mock_etl_callback.assert_not_called()


class TestSensorThingsAPIHandler:
    """Tests for the OGC SensorThings API (MQTT-based) data handler."""

    @patch('subsystems.isu.streaming_data_handler.stream_handler.mqtt')
    def test_STR_006_initialization(
        self, mock_mqtt, mock_logger, mock_qc_layer, mock_etl_callback
    ) -> None:
        """
        Verify MQTT client setup for SensorThings API ingestion.
        """
        _ = SensorThingsAPIHandler(
            broker_url='mqtt', datastream_id='99', logger=mock_logger,
            qc_layer=mock_qc_layer, etl_callback=mock_etl_callback, config={}
        )
        mock_mqtt.Client.assert_called_once()

    @patch('subsystems.isu.streaming_data_handler.stream_handler.mqtt')
    def test_STR_007_process_message_pass_qc(
        self, mock_mqtt, mock_logger, mock_qc_layer, mock_etl_callback, sample_payload
    ) -> None:
        """
        Verify successful pipeline execution for SensorThings MQTT messages.
        """
        mock_qc_layer.check.return_value = {'final_status': 'Pass', 'errors': []}
        handler = SensorThingsAPIHandler(
            broker_url='mqtt', datastream_id='99', logger=mock_logger,
            qc_layer=mock_qc_layer, etl_callback=mock_etl_callback, config={}
        )
        handler._execute_pipeline(sample_payload, '99', 'sensorthings', 'pass')
        mock_etl_callback.assert_called_once()


class TestStreamingDataHandler:
    """Integration tests for the main StreamingDataHandler subsystem Facade."""

    @patch('subsystems.isu.streaming_data_handler.stream_handler.KafkaStreamHandler.start_consuming')
    @patch('subsystems.isu.streaming_data_handler.stream_handler.KafkaConsumer')
    def test_STR_008_start_and_stop(
        self, mock_kafka_consumer, mock_start_consuming, mock_logger, mock_qc_layer, mock_etl_callback
    ) -> None:
        """
        Verify that start() correctly spawns a background thread and stop() terminates it.
        """
        mock_start_consuming.side_effect = lambda: time.sleep(0.5)

        # 1. 
        handler = StreamingDataHandler(
            qc_layer=mock_qc_layer,
            etl_callback=mock_etl_callback,
        )

        # 2. 
        handler.source_type = 'kafka'
        mock_processor = MagicMock()
        mock_processor.start_consuming = mock_start_consuming
        handler._stream_processor = mock_processor

        # 3. 
        assert handler._thread is None
        handler.start()
        
        assert handler._thread is not None
        assert handler._thread.is_alive() is True
        assert handler._thread.name == 'ISU-Kafka-Thread'

        # 4. 
        handler.stop()
        mock_processor.stop.assert_called_once()
