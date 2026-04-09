from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DataContainer:
    """Generic container passed between DAG nodes."""
    data: Any
    metadata: Dict = None


"""
class DataCube:
    def __init__(
        self, data=None, timestamps=None, transform=None, crs=None, metadata=None
    ):
        print('[DataCube] Initialized')
        self.data = data
        self.timestamps = timestamps or []
        self.transform = transform
        self.crs = crs
        self.metadata = metadata or {}


class FeatureCube(DataCube):
    def __init__(self, *args, feature_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        print('[FeatureCube] Initialized')
        self.feature_names = feature_names or []


class InSituSeries:
    def __init__(self, timestamps, values, x, y, point_id):
        print('[InSituSeries] Initialized')

        self.timestamps = timestamps
        self.values = values
        self.x = x
        self.y = y
        self.point_id = point_id

"""
