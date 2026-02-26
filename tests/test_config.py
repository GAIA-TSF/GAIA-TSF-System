from pathlib import Path


class TestConfig:
    def test_config_001(self):
        from lib.config import ConfigReader

        config = ConfigReader(
            Path(__file__).parent.parent / 'projects' / 'jagersfontein.yml'
        )

        assert config['project']['name'] == 'Jagersfontein'
