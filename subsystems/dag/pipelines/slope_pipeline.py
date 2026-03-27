
class SlopeStabilityPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        self.loader = EODataLoader()
        self.mask = MaskApplier()
        self.harmonizer = SpatialHarmonizer()
        self.features = DisplacementFeatureExtractor()
        self.normalizer = Normalizer()
        self.tensorizer = Tensorizer()

    def run(self, inputs):
        print('[SlopePipeline] Running')

        cube = self.loader.load_sentinel1(inputs.get('s1'))
        cube = self.mask.apply_aoi_mask(cube, inputs.get('aoi'))
        cube = self.harmonizer.resample_to_common_grid([cube])[0]

        vel = self.features.compute_velocity(cube)
        acc = self.features.compute_acceleration(cube)

        self.normalizer.fit(vel)
        vel = self.normalizer.transform(vel)

        return self.tensorizer.to_numpy(vel)

