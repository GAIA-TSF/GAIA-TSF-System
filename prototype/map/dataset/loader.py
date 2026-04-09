
"""
Dataset loading module

In real system:
- reads Sentinel-2 data
- constructs time series
"""

def load_dataset(config):
    """Load training dataset"""
    print("[Dataset] Loading AMD dataset")
    return "raw_data"


def load_new_data(config):
    """Load inference dataset"""
    print("[Dataset] Loading NEW AMD data")
    return "new_data" 
