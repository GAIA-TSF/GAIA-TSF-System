
from subsystems.dag.core.data_model import DataContainer


class Tensorizer:
    def run(self, data: DataContainer) -> DataContainer:
        # data.metadata["tensor_ready"] = True
        print("[Tensorizer] Running tensorization step") 
        return data
    """ 
    def to_numpy(self, cube):
        print('[Tensorizer] Converting to numpy array')
        return 'numpy_array_mock'
    """ 

