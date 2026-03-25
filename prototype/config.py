from pathlib import Path
import yaml


class Config:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        self.root_dir = Path(self._cfg["project"]["root_dir"])

        # Resolve paths
        self.shapefiles_dir = self.root_dir / self._cfg["data"]["raw"]["shapefiles"]
        self.rasters_dir = self.root_dir / self._cfg["data"]["raw"]["rasters"]

        self.features_dir = self.root_dir / self._cfg["data"]["processed"]["features"]
        self.models_dir = self.root_dir / self._cfg["data"]["processed"]["models"]
        self.outputs_dir = self.root_dir / self._cfg["data"]["processed"]["outputs"]

        # GitHub
        self.github_repo = self._cfg["github"]["repo"]
        self.github_commit = self._cfg["github"]["commit"]
        self.github_subdir = self._cfg["github"]["subdir"]

        # Fields
        self.class_field = self._cfg["fields"]["class_name"]

    def ensure_directories(self):
        """Create all required directories."""
        for path in [
            self.shapefiles_dir,
            self.rasters_dir,
            self.features_dir,
            self.models_dir,
            self.outputs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)