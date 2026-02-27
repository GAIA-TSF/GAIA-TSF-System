from pathlib import Path

from lib.config import ConfigReader, ProjectConfigReader


class TestConfig:
    def test_config_001(self):
        """Process sample project file by ConfigReader and check project/name option."""
        config = ConfigReader(Path(__file__).parent / 'projects' / 'jagersfontein.yml')

        assert config['project']['name'] == 'Jagersfontein'

    def test_config_002(self):
        """Process sample project config file by ProjectConfigReader."""
        config = ProjectConfigReader(
            Path(__file__).parent / 'projects' / 'jagersfontein.yml'
        )

        assert config.is_valid() is True

        # delete required option
        config = ProjectConfigReader(
            Path(__file__).parent / 'projects' / 'jagersfontein.yml'
        )
        del config['project']['name']
        config.validate(
            dict(config)
        )  # re-validate config after modification
        assert config.is_valid() is False

        # make WKT invalid
        config = ProjectConfigReader(
            Path(__file__).parent / 'projects' / 'jagersfontein.yml'
        )
        config['project']['aoi']['geom'] = 'X'
        config.validate(
            dict(config)
        )  # re-validate config after modification
        assert config.is_valid() is False
