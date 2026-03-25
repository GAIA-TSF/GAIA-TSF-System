from pathlib import Path
import geopandas as gpd
import requests
import zipfile
import io
import shutil

from config import Config


# ---------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------
def list_shapefiles(folder: Path):
    return sorted(folder.glob("*.shp"))


def extract_class_names(shp_paths, field_name):
    values = set()

    for path in shp_paths:
        try:
            gdf = gpd.read_file(path)
            if field_name in gdf.columns:
                values.update(gdf[field_name].dropna().unique())
        except Exception as e:
            print(f"[WARNING] {path.name}: {e}")

    return values


def download_shapefiles(cfg: Config):
    url = f"https://github.com/{cfg.github_repo}/archive/{cfg.github_commit}.zip"

    print(f"[INFO] Downloading {url}")

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        prefix = f"amd-{cfg.github_commit}/{cfg.github_subdir}/"

        for member in z.namelist():
            if member.startswith(prefix) and not member.endswith("/"):
                filename = Path(member).name
                target = cfg.shapefiles_dir / filename

                with z.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        print(f"[INFO] Shapefiles downloaded to {cfg.shapefiles_dir}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    cfg = Config(Path("config.yaml"))

    # ensure folder structure
    cfg.ensure_directories()

    # download data
    download_shapefiles(cfg)

    # list shapefiles
    shp_paths = list_shapefiles(cfg.shapefiles_dir)

    print("\n[INFO] Shapefiles:")
    for shp in shp_paths:
        print(f"  - {shp.name}")

    # extract classes
    classes = extract_class_names(shp_paths, cfg.class_field)

    print("\n[INFO] Classes:")
    for c in sorted(classes):
        print(f"  - {c}")


if __name__ == "__main__":
    main()