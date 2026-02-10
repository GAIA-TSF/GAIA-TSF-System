"""Unit tests for the Parsing Engine module."""

import pytest
import pandas as pd
from unittest.mock import MagicMock
from subsystems.isu.parsers import ParsingEngine


@pytest.fixture
def mock_logger():
    """Create a mock logger to simulate the system logger."""
    return MagicMock()


@pytest.fixture
def engine(mock_logger):
    """Initialize the ParsingEngine with the mock logger injected."""
    return ParsingEngine(logger=mock_logger)


def test_csv_parsing(engine, tmp_path):
    """Test if CSV files are correctly identified and parsed."""
    # 1. Setup: Create a dummy CSV file in a temp directory
    d = tmp_path / 'input'
    d.mkdir()
    p = d / 'test_data.csv'
    p.write_text('timestamp,displacement_x,displacement_y\n2026-01-01,0.1,0.2')

    # 2. Action: Read bytes and pass to route_and_parse
    content = p.read_bytes()
    result = engine.route_and_parse(content, p.name)

    # 3. Assertion: Verify the result
    assert result['status'] == 'success'
    assert 'Slope' in result['parser_applied']
    assert result['row_count'] == 1

    # Verify logger
    engine.logger.info.assert_called()


def test_excel_parsing(engine, tmp_path):
    """Test if Excel (.xlsx) files are correctly identified and parsed."""
    # 1. Setup
    d = tmp_path / 'input'
    d.mkdir()
    p = d / 'test_data.xlsx'

    # Create a real Excel file using pandas
    df_source = pd.DataFrame({'timestamp': ['2026-01-01'], 'ph': [7.0]})
    df_source.to_excel(p, index=False)

    # 2. Action
    content = p.read_bytes()
    result = engine.route_and_parse(content, p.name)

    # 3. Assertion
    assert result['status'] == 'success'
    assert result['row_count'] == 1


def test_unknown_extension(engine, tmp_path):
    """Test that unsupported file formats are handled gracefully."""
    # 1. Setup: Create a file with an unsupported extension
    d = tmp_path / 'input'
    d.mkdir()
    p = d / 'funny_image.jpg'
    p.write_bytes(b'\xff\xd8\xff')

    # 2. Action
    content = p.read_bytes()
    result = engine.route_and_parse(content, p.name)

    # 3. Assertion
    assert result['status'] == 'quarantine'
    assert 'No parser matched' in result['reason']
