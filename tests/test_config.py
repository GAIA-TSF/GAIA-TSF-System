from pathlib import Path

from lib.config import ConfigReader


class TestConfig:
    def test_config_001(self):
        config = ConfigReader(Path(__file__).parent / 'projects' / 'jagersfontein.yml')

        assert config['project']['name'] == 'Jagersfontein'
