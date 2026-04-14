
from pathlib import Path
import numpy as np
import rasterio

# subsystems. 
from dag.core.data_model import DataContainer

def extract_sentinel2_timestamp(path):
    """
    Extract timestamp from filename.
    Temporal alighnment layer helper. 

    Example:
        20180701T102021.tif -> 20180701T102021
    """
    return path.stem.split("_")[0] 


class AMDCloudMasking:
    def __init__(self, config):
        self.cfg = config

        self.s2_cfg = config.inputs.amd.sentinel2

        self.cloud_dir = Path(self.s2_cfg.cloud_mask_path)
        self.cloud_suffix = self.s2_cfg.cloud_sufix

        self.cloud_code = int(self.s2_cfg.cloud_code)
        self.water_code = int(self.s2_cfg.water_code)

    def _index_cloud_masks(self): 
        """
        Build dictionary: timestamp -> mask array
        """
        
        mask_files = list(self.cloud_dir.glob(self.cloud_suffix + ".tif"))

        mask_index = {}

        for path in mask_files:
            ts = extract_sentinel2_timestamp(path)

            with rasterio.open(path) as src:
                mask_index[ts] = src.read(1)

        return mask_index

    def run(self, data: DataContainer) -> DataContainer:
        print("[AMDCloudMasking] Applying cloud + water mask")

        features = data.data  # (T, C, H, W)
        feature_paths = data.metadata.get("paths", [])

        if not feature_paths:
            raise ValueError("Missing feature paths in metadata")

        mask_index = self._index_cloud_masks()
        print(f"[AMDCloudMasking] Indexed {len(mask_index)} masks")

        T, C, H, W = features.shape
        masked = features.copy()

        for t, fpath in enumerate(feature_paths):
            ts = extract_sentinel2_timestamp(fpath)
            print(f"[MATCH] {fpath.name} -> {ts}") 

            if ts not in mask_index:
                raise ValueError(f"No cloud mask for timestamp: {ts}")

            cm = mask_index[ts]

            # --- masks ---
            cloud_mask = cm == self.cloud_code
            water_mask = cm == self.water_code

            valid_mask = (~cloud_mask) & (water_mask)

            valid_mask = np.expand_dims(valid_mask, axis=0)
            valid_mask = np.repeat(valid_mask, C, axis=0)

            masked[t][~valid_mask] = np.nan

        data.data = masked

        print("[AMDCloudMasking] Masking complete")

        return data
