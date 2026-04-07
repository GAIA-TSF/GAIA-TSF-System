
from subsystems.dag.core.data_model import DataContainer


class EOFeatureExtractor:
    def run(self, data: DataContainer) -> DataContainer:
        # data.metadata["features_extracted"] = True    
        print("[EOFeatureExtractor] Running EO feature extraction step") 
        return data

"""    
class DisplacementFeatureExtractor:
    def compute_velocity(self, cube):
        print('[DisplacementFeatureExtractor] Computing velocity')
        return FeatureCube()  # noqa: F821

    def compute_acceleration(self, cube):
        print('[DisplacementFeatureExtractor] Computing acceleration')
        return FeatureCube()  # noqa: F821


class AMDFeatureExtractor:
    def compute_amd_index(self, cube):
        print('[AMDFeatureExtractor] Computing AMD index')
        return FeatureCube()  # noqa: F821
"""