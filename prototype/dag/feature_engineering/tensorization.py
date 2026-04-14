
from pathlib import Path
import numpy as np

# subsystems. 
from dag.core.data_model import DataContainer

try:
    import torch
except ImportError:
    torch = None


class Tensorizer:
    def __init__(self, config):
        """
        Tensorization module.

        Converts feature stack into desired layout and format
        based on config.

        Expected input:
            data.data → numpy array (T, C, H, W)
        """
        self.cfg = config

        self.tensor_cfg = config.tensorization
        self.output_cfg = config.output

    def _convert_layout(self, arr):
        """
        Convert array layout.

        Input:
            (T, C, H, W)

        Output:
            THWC → (T, H, W, C)
            TCHW → (T, C, H, W)
        """
        layout = self.tensor_cfg.layout.upper()

        if layout == "THWC":
            return np.transpose(arr, (0, 2, 3, 1))

        elif layout == "TCHW":
            return arr  # already in correct format

        else:
            raise ValueError(f"Unsupported layout: {layout}")

    def _to_output_format(self, arr):
        """
        Convert to desired output format.
        """
        fmt = self.tensor_cfg.output_format.lower()

        if fmt == "numpy":
            return arr

        elif fmt == "torch":
            if torch is None:
                raise ImportError("PyTorch not installed")
            return torch.from_numpy(arr)

        else:
            raise ValueError(f"Unsupported output_format: {fmt}")

    def _save(self, tensor):
        """
        Save tensor to disk.
        """
        out_dir = Path(self.output_cfg.paths.tensors)
        out_dir.mkdir(parents=True, exist_ok=True)

        fmt = self.output_cfg.formats.tensor.lower()

        if fmt == "npy":
            path = out_dir / "tensor.npy"
            np.save(path, tensor)

        elif fmt == "pt":
            if torch is None:
                raise ImportError("PyTorch not installed")
            path = out_dir / "tensor.pt"
            torch.save(tensor, path)

        else:
            raise ValueError(f"Unsupported tensor format: {fmt}")

        print(f"[Tensorizer] Saved tensor to: {path}")

    def run(self, data):
        """
        Execute tensorization step.
        """
        print("[Tensorizer] Running tensorization")

        arr = data.data  

        # --- Layout conversion ---
        arr = self._convert_layout(arr)
        print(f"[Tensorizer] Layout converted → {arr.shape}")

        # --- Format conversion ---
        tensor = self._to_output_format(arr)

        # --- Save ---
        if self.output_cfg.save_intermediate:
            self._save(tensor)

        # --- Update container ---
        data.data = tensor

        return data