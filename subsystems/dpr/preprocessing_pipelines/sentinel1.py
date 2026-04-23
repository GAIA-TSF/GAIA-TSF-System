from .base import PreprocessingBasePipeline


class Sentinel1Pipeline(PreprocessingBasePipeline):
    metadata = {
        'title': 'Sentinel-1',
        'abstract': 'Anomaly detection for slope stability: preprocess Sentinel-1 data',
        'params': {},
    }

    def _build_sbas_stack(self):
        raise NotImplementedError()

    def _reframe_sbas(self):
        raise NotImplementedError()

    def _run(self):
        self._build_sbas_stack()
        self._reframe_sbas
        # ...
