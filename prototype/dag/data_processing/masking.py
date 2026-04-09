
from subsystems.dag.core.data_model import DataContainer


class Masking:
    def run(self, data: DataContainer) -> DataContainer:
        # data.metadata["masked"] = True
        print("[Masking] Running masking step") 
        return data
