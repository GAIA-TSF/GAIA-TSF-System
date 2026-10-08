from subsystems.dag.core.registry import PLUGIN_REGISTRY
from subsystems.dag.plugins.eda.slope_eda import SlopeEDA
from subsystems.dag.plugins.features.slope_features import SlopeFeatureExtractor
from subsystems.dag.plugins.features.temporal_features import TemporalFeatureExtractor
from subsystems.dag.plugins.ingestion.sentinel1_loader import Sentinel1LOSLoader


PLUGIN_REGISTRY.register('sentinel1_los_loader', Sentinel1LOSLoader)
PLUGIN_REGISTRY.register('slope_eda', SlopeEDA)
PLUGIN_REGISTRY.register('slope_feature_extractor', SlopeFeatureExtractor)
PLUGIN_REGISTRY.register('temporal_feature_extractor', TemporalFeatureExtractor)

__all__ = [
    'PLUGIN_REGISTRY',
    'Sentinel1LOSLoader',
    'SlopeEDA',
    'SlopeFeatureExtractor',
    'TemporalFeatureExtractor',
]
