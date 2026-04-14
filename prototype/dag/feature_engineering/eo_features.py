
from pathlib import Path
import numpy as np
import rasterio

# subsystems.
from dag.core.data_model import DataContainer


class EOFeatureExtractor:
    def run(self, data: DataContainer) -> DataContainer:
        print("[EOFeatureExtractor] Running EO feature extraction step")
        return data


class AMDFeatureExtractor:
    def __init__(self, var_cfg, feature_cfg, output_cfg):
        self.var_cfg = var_cfg
        self.feature_cfg = feature_cfg
        self.output_cfg = output_cfg

    def _compute_indices(self, bands):
        outputs = []
        names = []

        # spec_cfg = self.feature_cfg.spectral_indices
        spec_cfg = getattr(self.feature_cfg, "spectral_indices", None)

        if spec_cfg is None:
            raise ValueError("Missing 'spectral_indices' in feature config")

        # AMD difference
        if hasattr(spec_cfg, "amd_diff") and spec_cfg.amd_diff.enabled:
            outputs.append(bands["B04"] - bands["B02"])
            names.append("amd_diff")

        # AMD ratio (optional)
        if hasattr(spec_cfg, "amd_ratio") and spec_cfg.amd_ratio.enabled:
            ratio = bands["B04"] / (bands["B02"] + 1e-10)
            outputs.append(ratio)
            names.append("amd_ratio")

        # NDWI
        if hasattr(spec_cfg, "ndwi") and spec_cfg.ndwi:
            ndwi = (bands["B02"] - bands["B04"]) / (
                bands["B02"] + bands["B04"] + 1e-10
            )
            outputs.append(ndwi)
            names.append("ndwi")

        if not outputs:
            raise ValueError("No spectral indices enabled in config")

        return np.stack(outputs, axis=0), names

    def run(self, data: DataContainer) -> DataContainer:

        print("[AMDFeatureExtractor] Computing AMD features")

        s2_path = Path(self.var_cfg.sentinel2.data_path)
        out_dir = Path(self.output_cfg.paths.features)
        out_dir.mkdir(parents=True, exist_ok=True)

        rasters = sorted(s2_path.glob("*.tif"))

        feature_stack = []
        feature_paths = []        

        for path in rasters:
            print(f"[AMD] Processing {path.name}")
            feature_paths.append(path)

            with rasterio.open(path) as src:
                bands = {
                    "B02": src.read(1).astype("float32"),
                    "B04": src.read(2).astype("float32"),
                }

                features, names = self._compute_indices(bands)
                feature_stack.append(features)

                # Save intermediate features
                if self.output_cfg.save_intermediate:
                    out_path = out_dir / f"{path.stem}_features.tif"

                    profile = src.profile.copy()
                    profile.update(count=features.shape[0], dtype="float32")

                    with rasterio.open(out_path, "w", **profile) as dst:
                        dst.write(features)

        # shape: (T, C, H, W)
        if not hasattr(data, "data"):
            raise TypeError("Expected DataContainer")
        
        data.data = np.array(feature_stack)
        data.metadata["paths"] = feature_paths   # FIX
        data.metadata["feature_names"] = names

        print(f"[AMD] Feature stack shape: {data.data.shape}")

        return data
