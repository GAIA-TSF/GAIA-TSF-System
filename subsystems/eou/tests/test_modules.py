from osgeo import gdal

gdal.UseExceptions()

from eou.data_acquisition_gateway import DataAcquisitionGateway


def load_geom(file_path):
    with gdal.OpenEx(file_path, gdal.OF_VECTOR) as ds:
        layer = ds.GetLayer(0)

        srs = layer.GetSpatialRef()
        srs.AutoIdentifyEPSG()
        auth = srs.GetAuthorityName(None)
        code = srs.GetAuthorityCode(None)
        if auth != 'EPSG' or code != '4326':
            raise RuntimeError(f'Unsupported CRS: {auth}:{code}')

        lonmin, lonmax, latmin, latmax = layer.GetExtent()

        return [lonmin, latmin, lonmax, latmax]


class TestModules:
    def test_ManualFileLoader_001(self):
        """Test ManualFileLoader module.

        Example of unit test.
        """
        pass

    def test_DataAcquisitionGateway_001(self):
        """Test DataAcquisitionGateway module.

        Test search capability using default backend (eodag).
        """
        from eodag.api.search_result import SearchResult

        module = DataAcquisitionGateway()

        # Search parameters
        provider = 'cop_dataspace'
        start = '2026-01-01'
        end = '2026-01-29'
        geom = load_geom('eou/tests/sample_data/area_intervencao.kmz')
        product_type = 'S2_MSI_L2A'

        result = module.search(
            provider=provider, start=start, end=end, geom=geom, productType=product_type
        )

        assert isinstance(result, SearchResult)
        assert len(result) > 0
        assert result[0].product_type == product_type

    def test_DataAcquisitionGateway_002(self):
        """Test DataAcquisitionGateway module.

        Test download capability using default backend (eodag).
        """
        import os
        import yaml

        config_file = 'eou/tests/eodag_config.yml'

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        added_envs = []
        for provider, details in config_data.items():
            creds = details.get('auth', {}).get('credentials', {})
            for key, value in creds.items():
                env_var = f'EODAG__{provider.upper()}__AUTH__CREDENTIALS__{key.upper()}'
                os.environ[env_var] = str(value)
                added_envs.append(env_var)

        try:
            module = DataAcquisitionGateway()

            provider = 'cop_dataspace'
            start = '2026-01-01'
            end = '2026-01-29'
            geom = load_geom('eou/tests/sample_data/area_intervencao.kmz')
            product_type = 'S2_MSI_L2A'

            results = module.search(
                provider=provider,
                start=start,
                end=end,
                geom=geom,
                productType=product_type,
            )

            assert len(results) > 0

            ql_path = module.download(results[0], quicklook=True)

            assert isinstance(ql_path, str)
            assert os.path.exists(ql_path)
            assert os.path.getsize(ql_path) > 0

        finally:
            for env_var in added_envs:
                os.environ.pop(env_var, None)

    def test_DataExtraction_001(self):
        """Test DataExtraction module.

        Example of unit test.
        """
        pass
