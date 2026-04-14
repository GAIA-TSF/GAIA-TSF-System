# subsystems. 
from dag.core.data_model import DataContainer


class Validator:
    def run(self, data: DataContainer) -> DataContainer:
        data.metadata["validated"] = True
        return data
