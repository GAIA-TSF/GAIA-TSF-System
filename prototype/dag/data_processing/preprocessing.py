# subsystems. 
from dag.core.data_model import DataContainer


class Preprocessor:
    def run(self, data: DataContainer) -> DataContainer:
        # data.metadata["preprocessed"] = True
        print("[Preprocessor] Running preprocessing step") 
        return data
 
""" 
class Normalizer:
    def fit(self, cube):
        print('[Normalizer] Fitting')

    def transform(self, cube):
        print('[Normalizer] Transforming')
        return cube

class MissingValueHandler:
    def impute(self, cube, strategy='mean'):
        print('[MissingValueHandler] Imputing with', strategy)
        return cube

class Transformer:
    def log_transform(self, cube):
        print('[Transformer] Log transform')
        return cube
"""