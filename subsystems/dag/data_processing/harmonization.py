
from subsystems.dag.core.data_model import DataContainer

class SpatialHarmonizer:
    def run(self, data: DataContainer) -> DataContainer:
        # data.metadata["harmonized"] = True
        print("[SpatialHarmonizer] Running spatial harmonization step")
        return data
    
class TemporalHarmonizer:
    def run(self, data: DataContainer) -> DataContainer:
        # data.metadata["harmonized"] = True
        print("[TemporalHarmonizer] Running temporal harmonization step")   
        return data
