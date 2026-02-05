"""Unit tests for the Parsing Engine module."""

import pytest
from subsystems.isu.parsers import ParsingEngine


@pytest.fixture
def engine():
    """Fixture to initialize the engine."""
    return ParsingEngine()


def test_water_quality_detection(engine):
    """Test if water quality CSV is correctly identified."""
    # Mock content: header has pH and Conductivity
    csv_content = b'timestamp,ph,conductivity,temp\n2023-01-01,7.0,500,20'
    filename = 'test_water.csv'

    result = engine.route_and_parse(csv_content, filename)

    assert result['status'] == 'success'
    assert 'Water_Quality_Parser' in result['parser_applied']
    assert result['confidence'] > 0.8


def test_slope_detection(engine):
    """Test if slope CSV is correctly identified."""
    # Mock content: header has displacement
    csv_content = (
        b'timestamp,sensor_id,displacement_x,displacement_y\n2023-01-01,S1,0.1,0.2'
    )
    filename = 'test_slope_sensor.csv'

    result = engine.route_and_parse(csv_content, filename)

    assert result['status'] == 'success'
    assert 'Slope_Stability_Parser' in result['parser_applied']


def test_quarantine_garbage(engine):
    """Test if garbage file goes to quarantine."""
    csv_content = b'item,price,shop\nApple,1.2,Supermarket'
    filename = 'shopping_list.csv'

    result = engine.route_and_parse(csv_content, filename)

    assert result['status'] == 'quarantine'
