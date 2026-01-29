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
        result = module.search()

        assert isinstance(result, SearchResult)

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
