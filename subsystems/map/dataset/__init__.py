import abc
from torch.utils.data import Dataset

"""
Base dataset module. 
It is planned to have: 
- dataset for InSAR time series (Mirmazloumi et al. 2023, simulated, and real)  
- dataset for AMD (Sweden, real)
"""

class DatasetModule(Dataset, metaclass=abc.ABCMeta):
    """Abstract base class for MAP datasets."""

    @abc.abstractmethod
    def build(self):
        """Build internal tensors."""
        raise NotImplementedError

