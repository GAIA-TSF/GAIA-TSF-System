import yaml


class ConfigReader(dict):
    def __init__(self, config_path: str):
        self.config_path = config_path
        super().__init__()

        with open(self.config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
            self.update(_config)
