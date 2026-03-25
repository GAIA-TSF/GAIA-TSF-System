from pathlib import Path
from typing import List
import requests
import zipfile
import io
import shutil

from config import Config


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def list_scenes(folder: Path) -> List[Path]:
    """Return sorted list of Sentinel-2 .tif scenes."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    return sorted(folder.glob("*.tif"))


def download_sentinel2_scenes(cfg: Config) -> None:
    """
    Download Sentinel-2 scenes from GitHub archive.

    Uses config:
        - repo
        - commit
        - subdir (adapted to sentinel2_scenes)
    """
    save_dir = cfg.rasters_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://github.com/{cfg.github_repo}/archive/{cfg.github_commit}.zip"
    print(f"[INFO] Downloading: {url}")

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Download failed: {e}")

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        prefix = f"amd-{cfg.github_commit}/sentinel2_scenes/"

        files_extracted = 0

        for member in z.namelist():
            if not member.startswith(prefix) or member.endswith("/"):
                continue

            filename = Path(member).name
            target_path = save_dir / filename

            with z.open(member) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            files_extracted += 1

        print(f"[INFO] Extracted {files_extracted} scenes to {save_dir}")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def main(config_path: Path = Path("config.yaml")) -> None:
    cfg = Config(config_path)

    # ensure directories exist
    cfg.ensure_directories()

    # download data
    download_sentinel2_scenes(cfg)

    # list scenes
    scenes = list_scenes(cfg.rasters_dir)

    print(f"\n[INFO] Sentinel-2 scenes: {len(scenes)}")

    print("\n[INFO] First 10 scenes:\n")
    for s in scenes[:10]:
        print(f"  - {s.name}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main() 