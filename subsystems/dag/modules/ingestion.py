
class EODataLoader:
    def load_sentinel1(self, paths):
        print('[EODataLoader] Loading Sentinel-1:', paths)
        return DataCube()

    def load_sentinel2(self, paths):
        print('[EODataLoader] Loading Sentinel-2:', paths)
        return DataCube() 
    