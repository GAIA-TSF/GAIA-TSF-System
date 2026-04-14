"""
Central module registration for DAG system.

This file maps operation names (from config.yaml)
to actual Python classes.
"""

from dag.core.registry import registry

# Import modules
from dag.feature_engineering.eo_features import AMDFeatureExtractor
from dag.data_processing.masking import AMDCloudMasking
from dag.feature_engineering.tensorization import Tensorizer


def register_all():
    """
    Register all available modules into the global registry.
    """
    
    print("[Registry] Registering modules...")

    registry.register("feature_engineering", AMDFeatureExtractor)
    # registry.register("masking", AMDMasking)
    registry.register("cloud_masking", AMDCloudMasking)
    registry.register("tensorization", Tensorizer)

    print("[Registry] Done.")

