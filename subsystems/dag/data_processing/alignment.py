
from subsystems.dag.core.data_model import DataContainer


class TemporalAlignerEO:
    def run(self, data: DataContainer) -> DataContainer:
        # data.metadata["eo_temporal_aligned"] = True
        print("[TemporalAlignerEO] Running EO temporal alignment step") 
        return data
