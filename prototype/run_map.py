from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump, load

from config import Config
# from data.inventory import list_scenes


def list_scenes(folder: Path, recursive: bool = False) -> List[Path]:
    """
    Return sorted list of Sentinel-2 scenes (.tif).

    Parameters
    ----------
    folder : Path
        Directory containing raster scenes
    recursive : bool
        If True, search subdirectories

    Returns
    -------
    List[Path]
        Sorted list of raster paths
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    pattern = "**/*.tif" if recursive else "*.tif"
    scenes = sorted(folder.glob(pattern))

    if not scenes:
        print(f"[WARNING] No scenes found in {folder}")

    return scenes


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_spectral_library(csv_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load spectral samples (X, y)."""
    df = pd.read_csv(csv_path)

    y = df.iloc[:, 0].tolist()
    X = df.iloc[:, 1:].values

    return X, y


def group_by_class(X: np.ndarray, y: List[str]) -> Dict[str, np.ndarray]:
    """Group spectra by class."""
    classes = np.unique(y)
    return {c: X[np.array(y) == c] for c in classes}


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
def compute_sam_features(X: np.ndarray, reference_spectra: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute SAM features."""
    sam_features = []

    for cls, spectra in reference_spectra.items():
        ref = np.nanmedian(spectra, axis=0)

        dot = np.dot(X, ref)
        norm_X = np.linalg.norm(X, axis=1)
        norm_ref = np.linalg.norm(ref)

        sam = np.arccos(dot / (norm_X * norm_ref + 1e-10))
        sam_features.append(sam.reshape(-1, 1))

    return np.hstack(sam_features)


def compute_ndvi(X: np.ndarray) -> np.ndarray:
    """Compute NDVI (S2: B8=7, B4=3)."""
    ndvi = (X[:, 7] - X[:, 3]) / (X[:, 7] + X[:, 3] + 1e-10)
    return ndvi.reshape(-1, 1)


def build_features(X: np.ndarray, reference_spectra: Dict[str, np.ndarray]) -> np.ndarray:
    """Combine spectral + engineered features."""
    sam = compute_sam_features(X, reference_spectra)
    ndvi = compute_ndvi(X)

    return np.hstack((X, sam, ndvi))


# ---------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------
def train_model(X: np.ndarray, y: List[str]) -> RandomForestClassifier:
    """Train RF with grid search."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y
    )

    params = {
        "n_estimators": [50, 100],
        "max_features": ["sqrt", None],
        "min_samples_leaf": [10, 25],
        "class_weight": ["balanced"],
    }

    clf = GridSearchCV(
        RandomForestClassifier(n_jobs=-1),
        param_grid=params,
        cv=3,
        scoring="accuracy",
        verbose=1,
    )

    clf.fit(X_train, y_train)

    print("[INFO] Best params:", clf.best_params_)

    best_model = clf.best_estimator_

    # evaluation
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[INFO] Accuracy: {acc*100:.2f}%")
    print("[INFO] Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    return best_model


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------
def predict_scene(
    scene_path: Path,
    model: RandomForestClassifier,
    reference_spectra: Dict[str, np.ndarray],
    label_map: Dict[str, int],
) -> np.ndarray:
    """Predict one scene."""
    with rasterio.open(scene_path) as src:
        data = src.read().astype("float32")
        profile = src.profile

    bands, rows, cols = data.shape
    pixels = data.reshape(bands, -1).T

    # features
    X = build_features(pixels, reference_spectra)

    y_pred = model.predict(X)
    y_num = np.array([label_map[c] for c in y_pred])

    return y_num.reshape(rows, cols), profile


def save_prediction(
    output_path: Path,
    prediction: np.ndarray,
    profile: dict,
):
    """Save raster prediction."""
    profile.update(dtype=rasterio.uint8, count=1, compress="lzw")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction.astype(rasterio.uint8), 1)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def main(config_path: Path = Path("config.yaml")):
    cfg = Config(config_path)

    # -----------------------------------------------------------------
    # Load training data
    # -----------------------------------------------------------------
    csv_path = cfg.outputs_dir / "spectral_samples.csv"
    X_raw, y = load_spectral_library(csv_path)

    reference_spectra = group_by_class(X_raw, y)

    # build features
    X = build_features(X_raw, reference_spectra)

    # -----------------------------------------------------------------
    # Train model
    # -----------------------------------------------------------------
    model = train_model(X, y)

    model_path = cfg.models_dir / "rf_model.joblib"
    dump(model, model_path)
    print(f"[INFO] Model saved: {model_path}")

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------
    scenes = list_scenes(cfg.rasters_dir)

    label_map = {c: i for i, c in enumerate(reference_spectra.keys())}

    for scene_path in scenes:
        print(f"[INFO] Predicting {scene_path.name}")

        pred, profile = predict_scene(
            scene_path, model, reference_spectra, label_map
        )

        out_path = cfg.outputs_dir / "predictions" / f"{scene_path.stem}_pred.tif"
        save_prediction(out_path, pred, profile)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
