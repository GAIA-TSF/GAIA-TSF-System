"""
Streaming throughput tests for ISU StreamingDataHandler.

Validates that the pipeline can sustain 1–10 msg/s under realistic sensor
data rates without dropping messages or stalling on QC failures.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from subsystems.isu.streaming_data_handler.stream_handler import KafkaStreamHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_logger() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_qc_layer() -> MagicMock:
    qc = MagicMock()
    qc.check.return_value = {'final_status': 'Pass', 'metrics': {}, 'errors': []}
    return qc


@pytest.fixture
def mock_etl_callback() -> MagicMock:
    return MagicMock()


@pytest.fixture
def sample_payload() -> dict:
    return {
        'sensor_type': 'piezometer',
        'timestamp': '2026-01-01T00:00:00Z',
        'pressure_kpa': 120.5,
        'temperature_c': 15.2,
    }


@pytest.fixture
def kafka_handler(mock_logger, mock_qc_layer, mock_etl_callback) -> KafkaStreamHandler:
    with patch('subsystems.isu.streaming_data_handler.stream_handler.KafkaConsumer'):
        return KafkaStreamHandler(
            broker='localhost:9092',
            topics=['test_topic'],
            logger=mock_logger,
            qc_layer=mock_qc_layer,
            etl_callback=mock_etl_callback,
            config={'data_type': 'in_situ'},
        )


# ---------------------------------------------------------------------------
# Throughput tests
# ---------------------------------------------------------------------------


class TestStreamingThroughput:
    """
    Validates the streaming pipeline meets the 1–10 msg/s throughput requirement.

    All tests drive `_execute_pipeline` directly so they are independent of
    Kafka/Kinesis brokers and focus purely on pipeline processing capacity.
    """

    @pytest.mark.parametrize('msg_count,min_rate', [
        (20,  1),   # lower bound: pipeline must handle at least  1 msg/s
        (100, 5),   # mid range: pipeline must handle at least    5 msg/s
        (200, 10),  # upper bound: pipeline must handle at least 10 msg/s
    ])
    def test_STR_THR_001_pipeline_meets_throughput_requirement(
        self,
        kafka_handler: KafkaStreamHandler,
        mock_etl_callback: MagicMock,
        sample_payload: dict,
        msg_count: int,
        min_rate: int,
    ) -> None:
        """
        Pipeline sustains >= min_rate msg/s over a burst of msg_count messages.
        Every message that passes QC must reach the ETL callback.
        """
        start = time.perf_counter()
        for i in range(msg_count):
            kafka_handler._execute_pipeline(
                payload=sample_payload,
                source='test_topic',
                protocol='kafka',
                dataset_id=f'thr_{i:04d}',
            )
        elapsed = time.perf_counter() - start

        actual_rate = msg_count / elapsed
        assert actual_rate >= min_rate, (
            f'Throughput {actual_rate:.1f} msg/s is below the required {min_rate} msg/s'
        )
        assert mock_etl_callback.call_count == msg_count

    def test_STR_THR_002_no_messages_dropped_under_burst(
        self,
        kafka_handler: KafkaStreamHandler,
        mock_etl_callback: MagicMock,
        sample_payload: dict,
    ) -> None:
        """All messages in a burst of 200 are forwarded to the ETL callback — no silent drops."""
        n = 200
        for i in range(n):
            kafka_handler._execute_pipeline(
                payload=sample_payload,
                source='test_topic',
                protocol='kafka',
                dataset_id=f'burst_{i:04d}',
            )
        assert mock_etl_callback.call_count == n

    def test_STR_THR_003_qc_failures_do_not_stall_pipeline(
        self,
        kafka_handler: KafkaStreamHandler,
        mock_qc_layer: MagicMock,
        mock_etl_callback: MagicMock,
        sample_payload: dict,
    ) -> None:
        """
        Interleaved QC failures don't slow down the pipeline: passing messages
        still flow through at >= 10 msg/s and failing messages are cleanly dropped.
        """
        n_total = 100
        # Alternate pass / fail
        mock_qc_layer.check.side_effect = [
            {
                'final_status': 'Pass' if i % 2 == 0 else 'Fail',
                'metrics': {},
                'errors': [] if i % 2 == 0 else ['threshold_exceeded'],
            }
            for i in range(n_total)
        ]

        start = time.perf_counter()
        for i in range(n_total):
            kafka_handler._execute_pipeline(
                payload=sample_payload,
                source='test_topic',
                protocol='kafka',
                dataset_id=f'mixed_{i:04d}',
            )
        elapsed = time.perf_counter() - start

        n_passed = n_total // 2
        assert mock_etl_callback.call_count == n_passed
        assert (n_total / elapsed) >= 10, (
            'Pipeline throughput dropped below 10 msg/s when handling mixed QC results'
        )

    @pytest.mark.slow
    @pytest.mark.parametrize('rate', [1, 5, 10])
    def test_STR_THR_004_rate_controlled_delivery(
        self,
        kafka_handler: KafkaStreamHandler,
        mock_etl_callback: MagicMock,
        sample_payload: dict,
        rate: int,
    ) -> None:
        """
        Messages arriving at exactly `rate` msg/s are all processed with no drops.
        Marked slow because it uses real wall-clock sleep to simulate sensor data rates.
        """
        n_messages = rate * 3  # run for 3 simulated seconds
        interval = 1.0 / rate

        for i in range(n_messages):
            kafka_handler._execute_pipeline(
                payload=sample_payload,
                source='test_topic',
                protocol='kafka',
                dataset_id=f'rate{rate}_{i:04d}',
            )
            time.sleep(interval)

        assert mock_etl_callback.call_count == n_messages
