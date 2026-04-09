
from subsystems.dag.core.data_model import DataContainer


class Sampler:
    def run(self, data: DataContainer) -> DataContainer:
        data.metadata["sampled"] = True
        return data
 