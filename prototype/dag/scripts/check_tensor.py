from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
except ImportError:
    torch = None


def load_tensor(path: Path):
    """
    Load tensor from file (npy or pt).
    """
    if path.suffix == ".npy":
        tensor = np.load(path)

    elif path.suffix == ".pt":
        if torch is None:
            raise ImportError("PyTorch not installed")
        tensor = torch.load(path)
        # tensor = tensor.cpu().numpy()

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return tensor



def inspect_tensor(tensor):
    """
    Print basic tensor info.
    """
    print("\n[TENSOR INSPECTION]")
    print("--------------------")

    print(f"Shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")

    # Interpret dimensions
    if len(tensor.shape) == 4:
        T, H, W, C = tensor.shape
        print(f"→ Time steps (T): {T}")
        print(f"→ Height (H): {H}")
        print(f"→ Width (W): {W}")
        print(f"→ Channels (C): {C}")

    elif len(tensor.shape) == 3:
        print("→ Likely flattened tensor (N, T, C)")

    # NaN statistics (important for masking)
    nan_count = np.isnan(tensor).sum()
    total = tensor.size

    for c in range(tensor.shape[-1]):
        channel = tensor[..., c]
        print(f"Channel {c}: min={np.nanmin(channel)}, max={np.nanmax(channel)}")

    print(f"\nNaN values: {nan_count} / {total} ({100 * nan_count / total:.2f}%)")

    print("--------------------\n")

    
    # Quick visualization of the first feature/channel at the first time step
    plt.imshow(tensor[2, :, :, 0])
    plt.title("Feature 0, timestep 0")
    plt.colorbar()
    plt.show()


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    # PATH
    tensor_path = Path(
        "/Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/AMD_monitoring_Yxsjoberg/results/tensors/tensor.pt"
    )

    tensor = load_tensor(tensor_path)

    inspect_tensor(tensor) 
