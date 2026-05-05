"""Unit tests for the Parsing Engine module."""

import pytest
from unittest.mock import MagicMock

from subsystems.isu.etl_engine.parsers import ParsingEngine
from tests.utils import get_data_path

TEST_DATA_DIR = get_data_path('isu')


@pytest.fixture
def mock_qcl_logger():
    """Create a mock logger to simulate the system logger."""
    return MagicMock()


@pytest.fixture
def engine(mock_qcl_logger):
    """Initialize the ParsingEngine with the mock logger injected."""
    return ParsingEngine(logger=mock_qcl_logger)


class TestParsingEngine:
    def test_MOD_001_csv_parsing(self, engine):
        """Test if CSV files are correctly identified and parsed."""
        p = TEST_DATA_DIR / 'slope_sensor_data.csv'

        content = p.read_bytes()
        result = engine.route_and_parse(content, p.name)

        assert result['status'] == 'success'
        assert 'Slope' in result['parser_applied']
        assert result['row_count'] > 0
        engine.logger.info.assert_called()

    def test_MOD_002_excel_parsing(self, engine):
        """Test if Excel (.xlsx) files are correctly identified and parsed."""
        p = TEST_DATA_DIR / 'water_quality_data.xlsx'

        content = p.read_bytes()
        result = engine.route_and_parse(content, p.name)

        assert result['status'] == 'success'
        assert result['row_count'] > 0

    def test_MOD_003_unknown_extension(self, engine):
        """Test that unsupported file formats are handled gracefully."""
        p = TEST_DATA_DIR / 'unsupported_data.txt'
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('random text')

        content = p.read_bytes()
        result = engine.route_and_parse(content, p.name)

        assert result['status'] == 'quarantine'
        assert 'No parser matched' in result['reason']
