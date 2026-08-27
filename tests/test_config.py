import pytest

from lib.config import ProjectConfigReader, SettingsReader
from tests.utils import TestUtils


@pytest.fixture(scope='class')
def project_config():
    return ProjectConfigReader(
        TestUtils.get_project_config_path('amd_monitoring_yxsjoberg')
    )


@pytest.fixture(scope='class')
def settings():
    return SettingsReader()


class TestProjectConfig:
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

    def test_config_003_aoi_geom(self, project_config):
        """Test AOI definitions provided by ProjectConfigReader."""
        assert project_config.aoi().wkt.startswith('POLYGON')

    def test_config_004_data_dir(self, project_config):
        """Check if project directory was created."""
        data_dir = project_config.data_dir()
        assert data_dir.exists() and data_dir.is_dir()


class TestSettings:
    def test_settings_001(self, settings):
        assert settings.is_valid() is True
