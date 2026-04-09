
from subsystems.dag.core.data_model import DataContainer


class EOLoader:
    def run(self, data: DataContainer) -> DataContainer:
        # placeholder
        # data.metadata = data.metadata or {}
        # data.metadata["eo_loaded"] = True
        print("[EOLoader] Running EO data loading step") 
        
        return data
    