"""Unit tests for the Parsing Engine module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock
from subsystems.isu.parsers import ParsingEngine

TEST_DATA_DIR = Path(__file__).parent / 'test_data'


@pytest.fixture
def mock_qcl_logger():
    """Create a mock logger to simulate the system logger."""
    return MagicMock()


@pytest.fixture
def engine(mock_qcl_logger):
    """Initialize the ParsingEngine with the mock logger injected."""
    return ParsingEngine(logger=mock_qcl_logger)


class TestParsingEngine:
    def test_csv_parsing(self, engine):
        """Test if CSV files are correctly identified and parsed."""
        # 1. Setup
        p = TEST_DATA_DIR / 'slope_sensor_data.csv'

        # 2. Action
        content = p.read_bytes()
        result = engine.route_and_parse(content, p.name)

        # 3. Assertion
        assert result['status'] == 'success'
        assert 'Slope' in result['parser_applied']
        assert result['row_count'] > 0

        # Verify logger
        engine.logger.info.assert_called()

    def test_excel_parsing(self, engine):
        """Test if Excel (.xlsx) files are correctly identified and parsed."""
        # 1. Setup
        p = TEST_DATA_DIR / 'water_quality_data.xlsx'

        # 2. Action
        content = p.read_bytes()
        result = engine.route_and_parse(content, p.name)

        # 3. Assertion
        assert result['status'] == 'success'
        assert result['row_count'] > 0

    def test_unknown_extension(self, engine):
        """Test that unsupported file formats are handled gracefully."""
        # 1. Setup
        p = TEST_DATA_DIR / 'unsupported_data.txt'

        # 2. Action
        content = p.read_bytes()
        result = engine.route_and_parse(content, p.name)

        # 3. Assertion
        assert result['status'] == 'quarantine'
        assert 'No parser matched' in result['reason']
