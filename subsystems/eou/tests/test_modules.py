from pathlib import Path
import shutil

from osgeo import gdal

gdal.UseExceptions()

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway


def load_geom(file_path):
    ds = gdal.OpenEx(file_path, gdal.OF_VECTOR)
    layer = ds.GetLayer(0)

    srs = layer.GetSpatialRef()
    srs.AutoIdentifyEPSG()
    auth = srs.GetAuthorityName(None)
    code = srs.GetAuthorityCode(None)
    if auth != 'EPSG' or code != '4326':
        raise RuntimeError(f'Unsupported CRS: {auth}:{code}')

    feature = layer.GetNextFeature()
    if feature is None:
        ds = None
        raise RuntimeError('No features found')

    wkt = feature.GetGeometryRef().ExportToWkt()
    ds = None

    return wkt


class TestModules:
    search_filter = {
        'provider': 'cop_dataspace',
        'start': '2026-01-01',
        'end': '2026-01-10',
        'productType': 'S2_MSI_L2A',
    }

    @staticmethod
    def _get_data_path(filename):
        return str(Path(__file__).parent / 'sample_data' / filename)

    def test_ManualFileLoader_001(self):
        """Test ManualFileLoader module.

        Performs file validity test.
        """
        from subsystems.eou.manual_file_loader import ManualFileLoader

        module = ManualFileLoader()
        result = module.check_file_validity(self._get_data_path('ENMAP01_sample.tif'))

        assert result['valid'] is True and result['driver'] == 'GTiff'
        assert len(result['errors']) < 1
        assert len(result['warnings']) < 1

    def test_DataAcquisitionGateway_001_eodag_search(self):
        """Test DataAcquisitionGateway module.

        Test search capability using default backend (eodag).
        """
        from eodag.api.search_result import SearchResult

        module = DataAcquisitionGateway()

        result = module.backend.search(
            geom=load_geom(self._get_data_path('area_intervencao.kmz')),
            **self.search_filter,
        )

        assert isinstance(result, SearchResult)
        assert len(result) > 0
        assert result[0].product_type == self.search_filter['productType']

    def test_DataAcquisitionGateway_001_asf_search(self):
        """Test DataAcquisitionGateway module.

        Test search capability using ASF backend.
        """
        from geopandas import GeoDataFrame

        module = DataAcquisitionGateway(backend='asf')
        aoi_geom = load_geom(self._get_data_path('area_intervencao.kmz'))
        result = module.backend.search(
            aoi=aoi_geom,
            start=self.search_filter['start'],
            end=self.search_filter['end'],
            direction='A',
        )

        assert isinstance(result, GeoDataFrame)
        assert result is not None
        assert len(result) > 0

    def test_DataAcquisitionGateway_002_eodag_download(self):
        """Test DataAcquisitionGateway module.

        Test download capability using default backend (eodag).
        """
        module = DataAcquisitionGateway()

        results = module.backend.search(
            geom=load_geom(self._get_data_path('area_intervencao.kmz')),
            **self.search_filter,
        )

        assert len(results) > 0

        ql_path = None
        try:
            ql_path = module.download(
                results[0], target_dir='sentinel2', quicklook=True
            )

            assert isinstance(ql_path, str)
            assert Path(ql_path).exists()
            assert Path(ql_path).stat().st_size > 0
        finally:
            if ql_path and Path(ql_path).exists():
                Path(ql_path).unlink()

    def test_DataAcquisitionGateway_002_asf_download(self):
        """Test DataAcquisitionGateway module.

        Test download capability using ASF backend.
        """
        download_path = Path(__file__).parent / 'sample_data' / 'asf_download'
        module = DataAcquisitionGateway(backend='asf')
        aoi_geom = load_geom(self._get_data_path('area_intervencao.kmz'))
        result = module.backend.search(
            aoi=aoi_geom,
            start=self.search_filter['start'],
            end=self.search_filter['end'],
            direction='A',
        )

        assert len(result) > 0

        try:
            download_path.mkdir(parents=True, exist_ok=True)
            datadir = module.backend.download(download_path, result)
            assert any(datadir.iterdir())
        finally:
            if download_path.exists() and download_path.is_dir():
                shutil.rmtree(download_path)

    def test_DataExtraction_001(self):
        """Test DataExtraction module.

        Example of unit test.
        """
        pass
