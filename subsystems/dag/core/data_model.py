
class DataCube:
    def __init__(self, data=None, timestamps=None, transform=None, crs=None, metadata=None):
        print('[DataCube] Initialized')
        self.data = data
        self.timestamps = timestamps or []
        self.transform = transform
        self.crs = crs
        self.metadata = metadata or {}
