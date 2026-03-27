
class AMDPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        self.loader = EODataLoader()
        self.mask = MaskApplier()
        self.harmonizer = SpatialHarmonizer()
        self.features = AMDFeatureExtractor()
        self.normalizer = Normalizer()
        self.tensorizer = Tensorizer()

    def run(self, inputs):
        print('[AMDPipeline] Running')

        cube = self.loader.load_sentinel2(inputs.get('s2'))
        cube = self.mask.apply_aoi_mask(cube, inputs.get('aoi'))
        cube = self.mask.apply_water_mask(cube, inputs.get('water_mask'))
        cube = self.harmonizer.resample_to_common_grid([cube])[0]

        amd = self.features.compute_amd_index(cube)

        self.normalizer.fit(amd)
        amd = self.normalizer.transform(amd)

        return self.tensorizer.to_numpy(amd) 
    