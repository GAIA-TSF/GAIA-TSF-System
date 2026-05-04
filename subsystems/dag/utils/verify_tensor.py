import argparse
from pathlib import Path
import torch
import numpy as np

""" 
Usage: 
python3 subsystems/dag/utils/verify_tensor.py \
    --input /Users/lukas/Work/prfuk/ownCloud/Projects/GAIA_TSF/tsf_experiments/AMD_monitoring_Yxsjoberg/results/tensors/tensor.pt \
    --pixel_y 142 \
    --pixel_x 81
"""

def load_tensor(path: str):
    print(f'[VERIFY] Loading tensor: {path}')

    data = torch.load(path)
    print(f'[INFO] Tensor data type: {type(data)}')

    # Case 1: already torch tensor
    if isinstance(data, torch.Tensor):
        print('[INFO] Detected PyTorch tensor')
        return data

    # Case 2: numpy array
    elif isinstance(data, np.ndarray):
        print('[INFO] Detected NumPy array → converting to torch.Tensor')
        return torch.from_numpy(data)

    # Case 3: dict (common in pipelines)
    elif isinstance(data, dict):
        print('[INFO] Detected dict, trying to extract tensor...')

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f'[INFO] Found tensor under key: {key}')
                return value
            elif isinstance(value, np.ndarray):
                print(f'[INFO] Found numpy array under key: {key}')
                return torch.from_numpy(value)

        raise TypeError('No tensor-like object found in dict')

    else:
        raise TypeError(f'Unsupported type: {type(data)}')

def check_shape(tensor):
    print('\n[CHECK] Shape')

    print('Shape:', tuple(tensor.shape))

    if tensor.ndim != 4:
        print('WARNING: Expected 4D tensor (T, H, W, C)')
    else:
        t, h, w, c = tensor.shape
        print(f'T={t}, H={h}, W={w}, C={c}')


def check_nan_inf(tensor):
    print('\n[CHECK] NaN / Inf')

    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()

    print(f'NaNs: {nan_count}')
    print(f'Infs: {inf_count}')


def check_stats(tensor):
    print('\n[CHECK] Global statistics')

    print('Min:', torch.min(tensor).item())
    print('Max:', torch.max(tensor).item())
    print('Mean:', torch.mean(tensor).item())
    print('Std:', torch.std(tensor).item())


def check_per_channel(tensor):
    print('\n[CHECK] Per-channel statistics')

    if tensor.ndim != 4:
        print('Skipping per-channel stats (not 4D)')
        return

    t, h, w, c = tensor.shape

    for i in range(c):
        channel = tensor[..., i]

        print(f'\nChannel {i}:')
        print('  Min:', torch.min(channel).item())
        print('  Max:', torch.max(channel).item())
        print('  Mean:', torch.mean(channel).item())
        print('  Std:', torch.std(channel).item())


def check_temporal_profile(tensor, pixel=(0, 0)):
    print('\n[CHECK] Temporal profile at pixel', pixel)

    if tensor.ndim != 4:
        print('Skipping temporal check')
        return

    t, h, w, c = tensor.shape

    y, x = pixel

    if y >= h or x >= w:
        print('Invalid pixel index')
        return

    profile = tensor[:, y, x, :]

    print('Profile shape:', profile.shape)
    print(profile)


def main():
    parser = argparse.ArgumentParser(description='Verify DAG tensor')

    parser.add_argument('--input', required=True, help='Path to .pt tensor file')
    parser.add_argument('--pixel_y', type=int, default=0)
    parser.add_argument('--pixel_x', type=int, default=0)

    args = parser.parse_args()

    tensor = load_tensor(args.input)

    check_shape(tensor)
    check_nan_inf(tensor)
    check_stats(tensor)
    check_per_channel(tensor)
    check_temporal_profile(tensor, pixel=(args.pixel_y, args.pixel_x))


if __name__ == '__main__':
    main()
