import json
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()

from eou.data_acquisition_gateway import DataAcquisitionGateway
from eou.data_extraction import DataExtraction


def load_geom(file_path):
    ds = gdal.OpenEx(file_path, gdal.OF_VECTOR)
    layer = ds.GetLayer(0)

    srs = layer.GetSpatialRef()
    srs.AutoIdentifyEPSG()
    auth = srs.GetAuthorityName(None)
    code = srs.GetAuthorityCode(None)
    if auth != 'EPSG' or code != '4326':
        raise RuntimeError(f'Unsupported CRS: {auth}:{code}')
    lonmin, lonmax, latmin, latmax = layer.GetExtent()
    ds = None

    return [lonmin, latmin, lonmax, latmax]


class TestModules:
    search_filter = {
        'provider': 'cop_dataspace',
        'start': '2026-01-01',
        'end': '2026-01-29',
        'productType': 'S2_MSI_L2A',
    }

    @staticmethod
    def _get_data_path(filename):
        return str(Path(__file__).parent / 'sample_data' / filename)

    def test_ManualFileLoader_001(self):
        """Test ManualFileLoader module.

        Performs file validity test.
        """
        from eou.manual_file_loader import ManualFileLoader

        module = ManualFileLoader()
        result = module.check_file_validity(self._get_data_path('ENMAP01_sample.tif'))

        assert result['valid'] is True and result['driver'] == 'GTiff'
        assert len(result['errors']) < 1
        assert len(result['warnings']) < 1

    def test_DataAcquisitionGateway_001(self):
        """Test DataAcquisitionGateway module.

        Test search capability using default backend (eodag).
        """
        from eodag.api.search_result import SearchResult

        module = DataAcquisitionGateway()

        result = module.search(
            geom=load_geom(self._get_data_path('area_intervencao.kmz')),
            **self.search_filter,
        )

        assert isinstance(result, SearchResult)
        assert len(result) > 0
        assert result[0].product_type == self.search_filter['productType']

    def test_DataAcquisitionGateway_002(self):
        """Test DataAcquisitionGateway module.

        Test download capability using default backend (eodag).
        """
        config_file = 'eou/tests/eodag_config.yml'

        module = DataAcquisitionGateway()

        results = module.search(
            geom=load_geom(self._get_data_path('area_intervencao.kmz')),
            **self.search_filter,
        )

        assert len(results) > 0

        module.set_config(config_file)
        ql_path = module.download(results[0], quicklook=True)

        assert isinstance(ql_path, str)
        assert Path(ql_path).exists()
        assert Path(ql_path).stat().st_size > 0

    def test_DataExtraction_001(self):
        """Test DataExtraction module.

        Test metadata creation on single GeoTIFF file.
        """

        def item_dict_no_datetime(item_dict):
            if 'properties' in item_dict and 'datetime' in item_dict['properties']:
                del item_dict['properties']['datetime']
            return item_dict

        module = DataExtraction(self._get_data_path('ENMAP01_sample.tif'))
        item_dict = module.stac_factory().create_item()

        with open(self._get_data_path('ENMAP01_sample.json'), 'r') as f:
            json_dict = json.load(f)
        assert item_dict_no_datetime(item_dict) == item_dict_no_datetime(json_dict)
