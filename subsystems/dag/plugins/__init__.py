"""Import and register the built-in DAG plugin implementations."""

from subsystems.dag.core.registry import PLUGIN_REGISTRY
from subsystems.dag.plugins.eda.slope_eda import SlopeEDA
from subsystems.dag.plugins.features.meteo_features import MeteoFeatureExtractor
from subsystems.dag.plugins.features.slope_features import SlopeFeatureExtractor
from subsystems.dag.plugins.features.temporal_features import TemporalFeatureExtractor
from subsystems.dag.plugins.features.topographic_features import (
    TopographicFeatureExtractor,
)
from subsystems.dag.plugins.ingestion.meteo_loader import MeteoRasterLoader
from subsystems.dag.plugins.ingestion.sentinel1_loader import Sentinel1LOSLoader

PLUGIN_REGISTRY.register('sentinel1_los_loader', Sentinel1LOSLoader)
PLUGIN_REGISTRY.register('slope_eda', SlopeEDA)
PLUGIN_REGISTRY.register('slope_feature_extractor', SlopeFeatureExtractor)
PLUGIN_REGISTRY.register('meteo_feature_extractor', MeteoFeatureExtractor)
PLUGIN_REGISTRY.register('temporal_feature_extractor', TemporalFeatureExtractor)
PLUGIN_REGISTRY.register('meteo_raster_loader', MeteoRasterLoader)
PLUGIN_REGISTRY.register('topographic_feature_extractor', TopographicFeatureExtractor)

__all__ = [
    'PLUGIN_REGISTRY',
    'MeteoFeatureExtractor',
    'MeteoRasterLoader',
    'Sentinel1LOSLoader',
    'SlopeEDA',
    'SlopeFeatureExtractor',
    'TemporalFeatureExtractor',
    'TopographicFeatureExtractor',
]
