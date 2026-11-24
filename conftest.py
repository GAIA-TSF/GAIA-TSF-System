import pytest

from lib.config import ProjectConfigReader


@pytest.fixture(autouse=True)
def reset_default_project_path():
    """Prevent the default project path from leaking between tests."""
    ProjectConfigReader.reset_default_path()
    yield
    ProjectConfigReader.reset_default_path()
