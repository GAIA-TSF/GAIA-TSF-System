from subsystems.dag.pipelines.base_pipeline import BasePipeline 

from subsystems.dag.modules.ingestion import EODataLoader
from subsystems.dag.modules.harmonization import SpatialHarmonizer
from subsystems.dag.modules.feature_engineering import AMDFeatureExtractor
from subsystems.dag.modules.preprocessing import Normalizer
from subsystems.dag.modules.tensorization import Tensorizer
from subsystems.dag.modules.masking import MaskApplier

from subsystems.dag.modules.insitu import InSituLoader, TemporalAligner 


class AMDPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        self.loader = EODataLoader()
        self.mask = MaskApplier()
        self.harmonizer = SpatialHarmonizer()
        self.features = AMDFeatureExtractor()
        self.normalizer = Normalizer()
        self.tensorizer = Tensorizer()

        self.insitu_loader = InSituLoader()
        self.temporal_aligner = TemporalAligner()

    def run(self, inputs):
        print('[AMDPipeline] Running')

        # cube = self.loader.load_sentinel2(inputs.get('s2'))
        # cube = self.mask.apply_aoi_mask(cube, inputs.get('aoi'))
        # cube = self.mask.apply_water_mask(cube, inputs.get('water_mask'))
        # cube = self.harmonizer.resample_to_common_grid([cube])[0]

        # amd = self.features.compute_amd_index(cube)

        # -----------------------------
        # NEW: In-situ integration
        # -----------------------------
        if inputs.get('insitu'):
            insitu = self.insitu_loader.load_csv(inputs.get('insitu'))

            # aligned = self.temporal_aligner.align_to_eo(
            #     insitu,
            #     cube.timestamps if hasattr(cube, 'timestamps') else []
            # ) # noqa: F821  

            # print('[AMDPipeline] In-situ aligned shape:', len(aligned))

        # self.normalizer.fit(amd)
        # amd = self.normalizer.transform(amd)

        # return self.tensorizer.to_numpy(amd) 


        return 0 

if __name__ == '__main__':
    print('[AMDPipeline] Standalone run')

    pipeline = AMDPipeline()

    inputs = {
        's2': ['dummy_s2_file.tif'],
        'aoi': 'dummy_aoi.geojson',
        'water_mask': 'dummy_mask.tif',
    }

    output = pipeline.run(inputs)

    print('Output:', output) 