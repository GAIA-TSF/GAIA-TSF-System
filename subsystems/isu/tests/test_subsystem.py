"""Tests for the ISU subsystem main class."""

from subsystems.isu import InSituDataUploader


def test_initialization():
    """Test that the subsystem initializes correctly."""
    isu = InSituDataUploader()
    assert isu is not None


def test_start_method(capsys):
    """Test the start method output."""
    isu = InSituDataUploader()
    isu.start()

    # Capture stdout
    captured = capsys.readouterr()
    assert 'ISU Subsystem started' in captured.out
