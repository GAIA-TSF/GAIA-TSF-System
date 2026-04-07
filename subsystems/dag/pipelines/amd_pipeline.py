
from subsystems.dag.pipelines.base_pipeline import BasePipeline

from subsystems.dag.data_import.eo_loader import EOLoader
from subsystems.dag.data_import.insitu import InSituLoader, TemporalAligner

from subsystems.dag.data_processing.harmonization import SpatialHarmonizer
from subsystems.dag.data_processing.alignment import TemporalAlignerEO

from subsystems.dag.feature_engineering.eo_features import EOFeatureExtractor
from subsystems.dag.feature_engineering.aggregation import MultiModalAggregator
from subsystems.dag.feature_engineering.tensorization import Tensorizer


class AMDPipeline(BasePipeline):
    def __init__(self):
        steps = [
            EOLoader(),
            InSituLoader(),
            TemporalAligner(),
            SpatialHarmonizer(),
            TemporalAlignerEO(),
            EOFeatureExtractor(),
            MultiModalAggregator(),
            Tensorizer(),
        ]
        super().__init__(steps)

