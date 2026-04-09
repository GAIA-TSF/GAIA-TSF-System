
from subsystems.dag.pipelines.base_pipeline import BasePipeline

# Data Import
from subsystems.dag.data_import.eo_loader import EOLoader
from subsystems.dag.data_import.insitu import InSituLoader, TemporalAligner

# Data Processing
from subsystems.dag.data_processing.harmonization import SpatialHarmonizer
from subsystems.dag.data_processing.alignment import TemporalAlignerEO
from subsystems.dag.data_processing.preprocessing import Preprocessor
from subsystems.dag.data_processing.masking import Masking

# Feature Engineering
from subsystems.dag.feature_engineering.eo_features import EOFeatureExtractor
from subsystems.dag.feature_engineering.aggregation import MultiModalAggregator
from subsystems.dag.feature_engineering.tensorization import Tensorizer


class SlopePipeline(BasePipeline):
    """
    Pipeline for slope stability analysis.

    Conceptually:
    EO (InSAR, S2) + InSitu → aligned → features → tensor
    """

    def __init__(self):
        steps = [
            # -------------------------
            # Data Import
            # -------------------------
            EOLoader(),
            InSituLoader(),
            TemporalAligner(),

            # -------------------------
            # Data Processing
            # -------------------------
            SpatialHarmonizer(),
            TemporalAlignerEO(),
            Preprocessor(),
            Masking(),

            # -------------------------
            # Feature Engineering
            # -------------------------
            EOFeatureExtractor(),     # later: slope-specific features (e.g. displacement trends)
            MultiModalAggregator(),
            Tensorizer(),
        ]

        super().__init__(steps) 
        