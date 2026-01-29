from earth_observation_data_uploader.data_acquisition_gateway import DataAcquisitionGateway

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
        # geom: tests/sample_data/area_intervencao.kmz
        # start: 2026-01-01
        # end: 2026-01-29
        # productType: "S2_MSI_L2A"
        # count=True ???
        # )
        result = module.search()

        assert isinstance(result, SearchResult)
        # assert result.number_matched > 0
        # assert result[0] check if data product is S2 L2A

    def test_DataAcquisitionGateway_002(self):
        """Test DataAcquisitionGateway module.

        Another example of unit test.
        """
        pass

    def test_DataExtraction_001(self):
        """Test DataExtraction module.

        Example of unit test.
        """
        pass
