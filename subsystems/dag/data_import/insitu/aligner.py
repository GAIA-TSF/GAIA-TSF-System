import numpy as np

from subsystems.dag.core.data_model import DataContainer


class TemporalAligner:
    def run(self, data: DataContainer) -> DataContainer:
        # data.metadata = data.metadata or {}
        # data.metadata["insitu_temporal_aligned"] = True
        print("[TemporalAligner] Running temporal alignment step")
        return data

    def align_to_eo(self, insitu, eo_timestamps, method='nearest'):
        print('[TemporalAligner] Aligning in-situ to EO timestamps')

        aligned_values = []

        for eo_time in eo_timestamps:
            closest_idx = self._find_closest(insitu['timestamps'], eo_time)
            aligned_values.append(insitu['values'][closest_idx])

        return np.array(aligned_values)

    def _find_closest(self, timestamps, target):
        diffs = [abs((t - target).total_seconds()) for t in timestamps]
        return diffs.index(min(diffs))
