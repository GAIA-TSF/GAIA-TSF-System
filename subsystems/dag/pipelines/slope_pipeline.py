from subsystems.dag.pipelines.base_pipeline import BasePipeline

from subsystems.dag.modules.ingestion import EODataLoader
from subsystems.dag.modules.harmonization import SpatialHarmonizer
from subsystems.dag.modules.feature_engineering import DisplacementFeatureExtractor
from subsystems.dag.modules.preprocessing import Normalizer
from subsystems.dag.modules.tensorization import Tensorizer
from subsystems.dag.modules.masking import MaskApplier


class SlopeStabilityPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        print('[SlopeStabilityPipeline] Initialized')

        self.loader = EODataLoader()
        self.mask = MaskApplier()
        self.harmonizer = SpatialHarmonizer()
        self.features = DisplacementFeatureExtractor()
        self.normalizer = Normalizer()
        self.tensorizer = Tensorizer()

    def run(self, inputs: dict):
        print('[SlopeStabilityPipeline] Running')

        # cube = self.loader.load_sentinel1(inputs.get('s1'))

        # cube = self.mask.apply_aoi_mask(cube, inputs.get('aoi'))

        # cube = self.harmonizer.resample_to_common_grid([cube])[0]

        # velocity = self.features.compute_velocity(cube)
        #  acceleration = self.features.compute_acceleration(cube)

        # self.normalizer.fit(velocity)
        # velocity = self.normalizer.transform(velocity)

        # output = self.tensorizer.to_numpy(velocity)

        print('[SlopeStabilityPipeline] Finished')

        # return output
        return 0


# Optional standalone run
if __name__ == '__main__':
    pipeline = SlopeStabilityPipeline()

    inputs = {
        's1': ['dummy_s1.tif'],
        'aoi': 'dummy_aoi.geojson',
    }

    result = pipeline.run(inputs)
    print('Output:', result)
