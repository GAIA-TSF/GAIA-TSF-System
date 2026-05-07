import pytest

from lib.config import ProjectConfigReader
from tests.utils import TestUtils.get_project_config_path


@pytest.fixture(scope='class')
def project_config():
    return ProjectConfigReader(TestUtils.get_project_config_path('amd_monitoring_yxsjoberg'))


class TestConfig:
    def test_config_001(self, project_config):
        """Process sample project file by ConfigReader and check project/name option."""
        assert project_config['project']['name'] == 'amd_baseline'

    def test_config_002(self, project_config):
        """Process sample project config file by ProjectConfigReader."""
        assert project_config.is_valid() is True

        # delete required option
        config = ProjectConfigReader(
            TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
        )
        del config['project']['name']
        config.validate(dict(config))  # re-validate config after modification
        assert config.is_valid() is False

        # make WKT invalid
        config = ProjectConfigReader(
            TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
        )
        config['project']['aoi'] = 'X'
        config.validate(dict(config))  # re-validate config after modification
        assert config.is_valid() is False
