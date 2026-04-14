
from subsystems.dag.core.data_model import DataContainer


class EOLoader:
    def run(self, data: DataContainer) -> DataContainer:
        """Run EO data loading step.
        Args:
            data (DataContainer): Input data container (can be empty or contain metadata)
        Returns:            
            DataContainer: Output data container with loaded EO data
        """
        # placeholder
        # data.metadata = data.metadata or {}
        # data.metadata["eo_loaded"] = True
        # TODO: implement actual EO data loading logic here 
        # (e.g. read from disk, query API, etc.)
        print("[EOLoader] Running EO data loading step") 
        
        return data
    