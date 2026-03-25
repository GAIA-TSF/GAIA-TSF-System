
from pathlib import Path
from config import Config


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_directory(path: Path, create: bool = True) -> None:
    """
    Ensure directory exists.

    Args:
        path (Path): Directory path
        create (bool): Whether to create if missing
    """
    if path.exists():
        print(f"[OK] {path}")
    else:
        if create:
            path.mkdir(parents=True, exist_ok=True)
            print(f"[CREATED] {path}")
        else:
            raise FileNotFoundError(f"[ERROR] Missing directory: {path}")


def validate_file(path: Path) -> None:
    """Check if file exists."""
    if path.exists():
        print(f"[OK] File found: {path}")
    else:
        print(f"[WARNING] File missing: {path}")


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
def setup_project_structure(cfg: Config) -> None:
    """
    Create required project directories from config.
    """
    print("[INFO] Setting up project structure...\n")

    for path in [
        cfg.shapefiles_dir,
        cfg.rasters_dir,
        cfg.features_dir,
        cfg.models_dir,
        cfg.outputs_dir,
    ]:
        ensure_directory(path)


def validate_inputs(cfg: Config) -> None:
    """
    Validate key input files (optional, extendable).
    """
    print("\n[INFO] Validating inputs...\n")

    # Example: check one reference image
    example_image = cfg.rasters_dir / "20180701T102021.tif"
    validate_file(example_image)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
# def main(config_path: Path = Path("config.yaml")) -> None:
#     cfg = Config(config_path)

def main():
    config_path = Path(__file__).parent / "config.yaml"
    cfg = Config(config_path)

    # 1. Create directory structure
    setup_project_structure(cfg)

    # 2. Validate inputs
    validate_inputs(cfg)

    print("\n[INFO] Setup complete.")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
