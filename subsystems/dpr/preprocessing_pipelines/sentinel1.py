from .base import BasePipeline

class Sentinel1Pipeline(BasePipeline):
    metadata = {
        'title': 'Sentinel-1',
        'abstract': 'Anomaly detection for slope stability: preprocess Sentinel-1 data'
    }
    def _build_sbas_stack(self):
        raise NotImplementedError()

    def _reframe_sbas(self):
        raise NotImplementedError()

    def run(self):
        self._build_sbas_stack()
        self._reframe_sbas
        # ...
    
    
