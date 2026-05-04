import shutil
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()

from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway
from lib.config import SettingsReader


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

        target_dir = 'sentinel2'
        try:
            ql_path = Path(
                module.backend.download(
                    results[0], target_dir=target_dir, quicklook=True
                )
            )

            assert ql_path.exists()
            assert ql_path.stat().st_size > 0
            assert (
                ql_path.parent.resolve()
                == Path(SettingsReader()['storage']['data_dir'], target_dir).resolve()
            )
        finally:
            if ql_path and Path(ql_path).exists():
                Path(ql_path).unlink()

    def test_DataAcquisitionGateway_002_asf_download(self):
        """Test DataAcquisitionGateway module.

        Test download capability using ASF backend.
        """
        module = DataAcquisitionGateway(backend='asf')
        aoi_geom = load_geom(self._get_data_path('area_intervencao.kmz'))
        result = module.backend.search(
            aoi=aoi_geom,
            start=self.search_filter['start'],
            end=self.search_filter['end'],
            direction='A',
        )

        assert len(result) > 0

        target_dir = 'sentinel1'
        try:
            datadir = Path(module.backend.download(result, target_dir=target_dir))
            assert any(datadir.iterdir())
            assert (
                datadir.resolve()
                == Path(SettingsReader()['storage']['data_dir'], target_dir).resolve()
            )
        finally:
            if datadir.exists() and datadir.is_dir():
                shutil.rmtree(datadir)

    def test_DataExtraction_001(self):
        """Test DataExtraction module.

        Example of unit test.
        """
        pass
