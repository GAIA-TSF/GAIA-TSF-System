"""Unit tests for the Parsing Engine module."""
import os
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
    d = tmp_path / "input"
    d.mkdir()
    p = d / "test_data.csv"
    p.write_text("col1,col2\n1,2")

    # 2. Action: Parse the file using the engine
    # Note: We must pass the file path as a string
    df = engine.parse_file(str(p))

    # 3. Assertion: Verify the result
    assert df is not None, "Resulting DataFrame should not be None"
    assert not df.empty, "DataFrame should not be empty"
    # Verify that the logger recorded the success info
    engine.logger.info.assert_called()


def test_excel_parsing(engine, tmp_path):
    """Test if Excel (.xlsx) files are correctly identified and parsed."""
    # 1. Setup
    d = tmp_path / "input"
    d.mkdir()
    p = d / "test_data.xlsx"

    # Create a real Excel file using pandas
    df_source = pd.DataFrame({'a': [10, 20], 'b': [30, 40]})
    df_source.to_excel(p, index=False)

    # 2. Action
    df = engine.parse_file(str(p))

    # 3. Assertion
    assert df is not None
    assert len(df) == 2, "Should read exactly 2 rows"


def test_unknown_extension(engine, tmp_path):
    """Test that unsupported file formats are handled gracefully."""
    # 1. Setup: Create a file with an unsupported extension
    d = tmp_path / "input"
    d.mkdir()
    p = d / "funny_image.jpg"
    p.write_text("not data")

    # 2. Action
    df = engine.parse_file(str(p))

    # 3. Assertion
    # The engine should return None for unsupported files, not crash
    assert df is None
    # Verify that a warning was logged
    engine.logger.warning.assert_called()
