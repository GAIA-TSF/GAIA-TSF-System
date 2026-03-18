import pytest
import pandas as pd
import time
from unittest.mock import MagicMock, patch

from isu.streaming_data_handler.stream_handler import (
    KafkaStreamHandler,
    KinesisStreamHandler,
    SensorThingsAPIHandler,
)
from isu.streaming_data_handler import StreamingDataHandler


@pytest.fixture
def mock_logger() -> MagicMock:
    """
    Provide a mocked logger for testing.

    :return: A MagicMock instance acting as the logger.
    :rtype: MagicMock
    """
    return MagicMock()


@pytest.fixture
def mock_qc_layer() -> MagicMock:
    """
    Provide a mocked Quality Control Layer.

    :return: A MagicMock instance acting as the QC layer.
    :rtype: MagicMock
    """
    return MagicMock()


@pytest.fixture
def mock_etl_callback() -> MagicMock:
    """
    Provide a mocked ETL callback function.

    :return: A MagicMock instance acting as the ETL callback.
    :rtype: MagicMock
    """
    return MagicMock()


@pytest.fixture
def sample_payload() -> dict:
    """
    Provide a standard JSON payload simulating a sensor reading.

    :return: A dictionary representing sensor data.
    :rtype: dict
    """
    return {
        'sensor_type': 'piezometer',
        'timestamp': '2026-02-27T12:00:00Z',
        'pressure_kpa': 120.5,
        'temperature_c': 15.2,
    }


class TestKafkaStreamHandler:
    """
    Test suite for the KafkaStreamHandler class.
    """

    @patch('isu.streaming_data_handler.stream_handler.KafkaConsumer')
    def test_initialization(
        self,
        mock_kafka_consumer: MagicMock,
        mock_logger: MagicMock,
        mock_qc_layer: MagicMock,
        mock_etl_callback: MagicMock,
    ) -> None:
        """
        Test that the handler initializes correctly with the provided dependencies.

        :param mock_kafka_consumer: Mocked KafkaConsumer class.
        :type mock_kafka_consumer: MagicMock
        :param mock_logger: Mocked logger fixture.
        :type mock_logger: MagicMock
        :param mock_qc_layer: Mocked QC layer fixture.
        :type mock_qc_layer: MagicMock
        :param mock_etl_callback: Mocked ETL callback fixture.
        :type mock_etl_callback: MagicMock
        :return: None
        :rtype: None
        """
        # config
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
        assert handler.topics == ['slope_stability']
        assert handler.logger == mock_logger
        assert handler.config == test_config
        mock_kafka_consumer.assert_called_once()

    @patch('isu.streaming_data_handler.stream_handler.KafkaConsumer')
    def test_process_message_pass_qc(
        self,
        mock_kafka_consumer: MagicMock,
        mock_logger: MagicMock,
        mock_qc_layer: MagicMock,
        mock_etl_callback: MagicMock,
        sample_payload: dict,
    ) -> None:
        """
        Test that a valid message passing QC is properly routed to the ETL callback.

        :param mock_kafka_consumer: Mocked KafkaConsumer class.
        :type mock_kafka_consumer: MagicMock
        :param mock_logger: Mocked logger fixture.
        :type mock_logger: MagicMock
        :param mock_qc_layer: Mocked QC layer fixture.
        :type mock_qc_layer: MagicMock
        :param mock_etl_callback: Mocked ETL callback fixture.
        :type mock_etl_callback: MagicMock
        :param sample_payload: Sample sensor data fixture.
        :type sample_payload: dict
        :return: None
        :rtype: None
        """
        # Configure QC layer to simulate a successful validation
        mock_qc_layer.check.return_value = {
            'final_status': 'Pass',
            'metrics': {},
            'errors': [],
        }

        handler = KafkaStreamHandler(
            broker='localhost:9092',
            topics=['slope_stability'],
            logger=mock_logger,
            qc_layer=mock_qc_layer,
            etl_callback=mock_etl_callback,
            config={},
        )

        handler._execute_pipeline(
            payload=sample_payload,
            source='slope_stability',
            protocol='kafka',
            dataset_id='test_kafka_pass_123',
        )

        # Assert QC check was called
        mock_qc_layer.check.assert_called_once()

        # Assert ETL callback was executed since QC passed
        mock_etl_callback.assert_called_once()

        # Verify the DataFrame argument passed to the ETL callback
        args, kwargs = mock_etl_callback.call_args
        passed_df = kwargs['df'] if 'df' in kwargs else args[0]
        assert isinstance(passed_df, pd.DataFrame)
        assert passed_df.iloc[0]['pressure_kpa'] == 120.5

    @patch('isu.streaming_data_handler.stream_handler.KafkaConsumer')
    def test_process_message_fail_qc(
        self,
        mock_kafka_consumer: MagicMock,
        mock_logger: MagicMock,
        mock_qc_layer: MagicMock,
        mock_etl_callback: MagicMock,
        sample_payload: dict,
    ) -> None:
        """
        Test that a message failing QC is dropped and NOT routed to ETL.

        :param mock_kafka_consumer: Mocked KafkaConsumer class.
        :type mock_kafka_consumer: MagicMock
        :param mock_logger: Mocked logger fixture.
        :type mock_logger: MagicMock
        :param mock_qc_layer: Mocked QC layer fixture.
        :type mock_qc_layer: MagicMock
        :param mock_etl_callback: Mocked ETL callback fixture.
        :type mock_etl_callback: MagicMock
        :param sample_payload: Sample sensor data fixture.
        :type sample_payload: dict
        :return: None
        :rtype: None
        """
        # Configure QC layer to simulate a failed validation (Gatekeeping feature)
        mock_qc_layer.check.return_value = {
            'final_status': 'Fail',
            'metrics': {},
            'errors': ['Physical range validation failed'],
        }

        handler = KafkaStreamHandler(
            broker='localhost:9092',
            topics=['slope_stability'],
            logger=mock_logger,
            qc_layer=mock_qc_layer,
            etl_callback=mock_etl_callback,
            config={},
        )

        handler._execute_pipeline(
            payload=sample_payload,
            source='slope_stability',
            protocol='kafka',
            dataset_id='test_kafka_fail_456',
        )

        # Assert QC check was called
        mock_qc_layer.check.assert_called_once()

        # Assert ETL callback was NEVER called because QC failed
        mock_etl_callback.assert_not_called()
        mock_logger.warning.assert_called()


class TestKinesisStreamHandler:
    """
    Test suite for the KinesisStreamHandler class.
    """

    @patch('isu.streaming_data_handler.stream_handler.boto3')
    def test_initialization(
        self,
        mock_boto3: MagicMock,
        mock_logger: MagicMock,
        mock_qc_layer: MagicMock,
        mock_etl_callback: MagicMock,
    ) -> None:
        """
        Test that the Kinesis handler initializes correctly with boto3.

        :param mock_boto3: Mocked boto3 library.
        :type mock_boto3: MagicMock
        :param mock_logger: Mocked logger fixture.
        :type mock_logger: MagicMock
        :param mock_qc_layer: Mocked QC layer fixture.
        :type mock_qc_layer: MagicMock
        :param mock_etl_callback: Mocked ETL callback fixture.
        :type mock_etl_callback: MagicMock
        :return: None
        :rtype: None
        """
        # Simulate boto3 client and response
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.describe_stream.return_value = {
            'StreamDescription': {'Shards': [{'ShardId': 'shardId-0000'}]}
        }
        mock_client.get_shard_iterator.return_value = {
            'ShardIterator': 'test_iterator_string'
        }

        # config
        test_config = {'kinesis_iterator_type': 'LATEST'}
        handler = KinesisStreamHandler(
            stream_name='tsf_stream',
            region_name='eu-central-1',
            logger=mock_logger,
            qc_layer=mock_qc_layer,
            etl_callback=mock_etl_callback,
            config=test_config,
        )

        assert handler.stream_name == 'tsf_stream'
        assert handler.region_name == 'eu-central-1'
        assert handler.config == test_config
        assert handler._shard_iterator == 'test_iterator_string'
        mock_boto3.client.assert_called_once_with('kinesis', region_name='eu-central-1')

    @patch('isu.streaming_data_handler.stream_handler.boto3')
    def test_process_message_fail_qc(
        self,
        mock_boto3: MagicMock,
        mock_logger: MagicMock,
        mock_qc_layer: MagicMock,
        mock_etl_callback: MagicMock,
        sample_payload: dict,
    ) -> None:
        """
        Test that a Kinesis message failing QC is dropped and NOT routed to ETL.

        :param mock_boto3: Mocked boto3 library.
        :type mock_boto3: MagicMock
        :param mock_logger: Mocked logger fixture.
        :type mock_logger: MagicMock
        :param mock_qc_layer: Mocked QC layer fixture.
        :type mock_qc_layer: MagicMock
        :param mock_etl_callback: Mocked ETL callback fixture.
        :type mock_etl_callback: MagicMock
        :param sample_payload: Sample sensor data fixture.
        :type sample_payload: dict
        :return: None
        :rtype: None
        """
        # Configure QC layer to simulate a failed validation
        mock_qc_layer.check.return_value = {
            'final_status': 'Fail',
            'metrics': {},
            'errors': ['Kinesis physical range anomaly'],
        }

        handler = KinesisStreamHandler(
            stream_name='tsf_stream',
            region_name='eu-central-1',
            logger=mock_logger,
            qc_layer=mock_qc_layer,
            etl_callback=mock_etl_callback,
            config={},
        )

        handler._execute_pipeline(
            payload=sample_payload,
            source='tsf_stream',
            protocol='kinesis',
            dataset_id='test_kinesis_fail_001',
        )

        # Assert QC check was called
        mock_qc_layer.check.assert_called_once()

        # Assert ETL callback was NEVER called because QC failed
        mock_etl_callback.assert_not_called()
        mock_logger.warning.assert_called()


class TestSensorThingsAPIHandler:
    """
    Test suite for the SensorThingsAPIHandler class.
    """

    @patch('isu.streaming_data_handler.stream_handler.mqtt')
    def test_initialization(
        self,
        mock_mqtt: MagicMock,
        mock_logger: MagicMock,
        mock_qc_layer: MagicMock,
        mock_etl_callback: MagicMock,
    ) -> None:
        """
        Test that the SensorThings API handler initializes correctly.

        :param mock_mqtt: Mocked mqtt module inside stream_handler.
        :type mock_mqtt: MagicMock
        :param mock_logger: Mocked logger fixture.
        :type mock_logger: MagicMock
        :param mock_qc_layer: Mocked QC layer fixture.
        :type mock_qc_layer: MagicMock
        :param mock_etl_callback: Mocked ETL callback fixture.
        :type mock_etl_callback: MagicMock
        :return: None
        :rtype: None
        """
        # config
        test_config = {'mqtt_keepalive_sec': 120}
        handler = SensorThingsAPIHandler(
            broker_url='mqtt.sensorthings.org:1883',
            datastream_id='99',
            logger=mock_logger,
            qc_layer=mock_qc_layer,
            etl_callback=mock_etl_callback,
            config=test_config,
        )

        assert handler.broker_url == 'mqtt.sensorthings.org:1883'
        assert handler.datastream_id == '99'
        assert handler.config == test_config
        mock_mqtt.Client.assert_called_once()

    @patch('isu.streaming_data_handler.stream_handler.mqtt')
    def test_process_message_pass_qc(
        self,
        mock_mqtt: MagicMock,
        mock_logger: MagicMock,
        mock_qc_layer: MagicMock,
        mock_etl_callback: MagicMock,
        sample_payload: dict,
    ) -> None:
        """
        Test that a valid OGC SensorThings message passing QC is properly routed to ETL.

        :param mock_mqtt: Mocked mqtt module inside stream_handler.
        :type mock_mqtt: MagicMock
        :param mock_logger: Mocked logger fixture.
        :type mock_logger: MagicMock
        :param mock_qc_layer: Mocked QC layer fixture.
        :type mock_qc_layer: MagicMock
        :param mock_etl_callback: Mocked ETL callback fixture.
        :type mock_etl_callback: MagicMock
        :param sample_payload: Sample sensor data fixture.
        :type sample_payload: dict
        :return: None
        :rtype: None
        """
        # Configure QC layer to simulate a successful validation
        mock_qc_layer.check.return_value = {
            'final_status': 'Pass',
            'metrics': {},
            'errors': [],
        }

        handler = SensorThingsAPIHandler(
            broker_url='mqtt.sensorthings.org:1883',
            datastream_id='99',
            logger=mock_logger,
            qc_layer=mock_qc_layer,
            etl_callback=mock_etl_callback,
            config={},
        )

        handler._execute_pipeline(
            payload=sample_payload,
            source='Datastream_99',
            protocol='sensorthings',
            dataset_id='test_ogc_pass_777',
        )

        # Assert QC check was called
        mock_qc_layer.check.assert_called_once()

        # Assert ETL callback was executed since QC passed
        mock_etl_callback.assert_called_once()


class TestStreamingDataHandler:
    """
    Test suite for the StreamingDataHandler facade class.
    """

    @patch(
        'isu.streaming_data_handler.stream_handler.KafkaStreamHandler.start_consuming'
    )
    @patch('isu.streaming_data_handler.stream_handler.KafkaConsumer')
    def test_start_and_stop(
        self,
        mock_kafka_consumer: MagicMock,
        mock_start_consuming: MagicMock,
        mock_logger: MagicMock,
        mock_qc_layer: MagicMock,
        mock_etl_callback: MagicMock,
    ) -> None:
        """
        Test that the background thread starts and stops gracefully.

        :param mock_kafka_consumer: Mocked KafkaConsumer class.
        :type mock_kafka_consumer: MagicMock
        :param mock_start_consuming: Mocked start_consuming method.
        :type mock_start_consuming: MagicMock
        :param mock_logger: Mocked logger fixture.
        :type mock_logger: MagicMock
        :param mock_qc_layer: Mocked QC layer fixture.
        :type mock_qc_layer: MagicMock
        :param mock_etl_callback: Mocked ETL callback fixture.
        :type mock_etl_callback: MagicMock
        :return: None
        :rtype: None
        """
        mock_start_consuming.side_effect = lambda: time.sleep(0.5)

        handler = StreamingDataHandler(
            source_type='kafka',
            logger=mock_logger,
            qc_layer=mock_qc_layer,
            etl_callback=mock_etl_callback,
            kafka_broker='localhost:9092',
            kafka_topics=['water_quality'],
            config={},
        )

        # Ensure no thread is active initially
        assert handler._thread is None

        # Start the facade
        handler.start()

        # Verify thread was created and started
        assert handler._thread is not None
        assert handler._thread.is_alive() is True

        assert handler._thread.name == 'ISU-Kafka-Thread'

        # Stop the facade
        handler.stop()

        mock_logger.info.assert_any_call('Stopping StreamingDataHandler (kafka)...')
