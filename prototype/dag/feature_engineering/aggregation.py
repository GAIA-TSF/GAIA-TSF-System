# subsystems. 
from dag.core.data_model import DataContainer


class MultiModalAggregator:
    def run(self, data: DataContainer) -> DataContainer:
        # data.metadata["aggregated"] = True
        print("[MultiModalAggregator] Running multi-modal aggregation step") 
        return data
