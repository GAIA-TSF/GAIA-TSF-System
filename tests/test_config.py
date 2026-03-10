import yaml
import tempfile
import pytest
from pathlib import Path

from lib.config import ConfigReader, ProjectConfigReader
from lib.exceptions import GaiaConfigError


class TestConfig:
    project_file = Path(__file__).parent / 'projects' / 'jagersfontein.yml'

    def test_config_001(self):
        """Process sample project file by ConfigReader and check project/name option."""
        config = ConfigReader(Path(__file__).parent / 'projects' / 'jagersfontein.yml')

        assert config['project']['name'] == 'Jagersfontein'

    def test_config_002(self):
        """Process sample project config file by ProjectConfigReader."""
        config = ProjectConfigReader(self.project_file)

        assert config.is_valid() is True

        # delete required option
        config = ProjectConfigReader(self.project_file)
        del config['project']['name']
        config.validate(dict(config))  # re-validate config after modification
        assert config.is_valid() is False

        # make WKT invalid
        config = ProjectConfigReader(self.project_file)
        config['project']['aoi']['geom'] = 'X'
        config.validate(dict(config))  # re-validate config after modification
        assert config.is_valid() is False

    def test_config_003(self):
        """Check invalid project config file by ProjectConfigReader."""
        with open(self.project_file) as fd:
            config = yaml.safe_load(fd)
        del config['project']['name']

        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', delete=False
        ) as fd:
            yaml.dump(config, fd)

        with pytest.raises(GaiaConfigError):
            config = ProjectConfigReader(fd.name)
