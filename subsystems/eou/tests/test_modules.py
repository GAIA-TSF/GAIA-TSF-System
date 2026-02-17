from pathlib import Path


class TestModules:
    def test_ManualFileLoader_001(self):
        """Test ManualFileLoader module.

        Performs file validity test.
        """
        from eou.manual_file_loader import ManualFileLoader

        module = ManualFileLoader()
        result = module.check_file_validity(
            str(Path(__file__).parent / 'sample_data' / 'ENMAP01_sample.tif')
        )

        assert result['valid'] is True and result['driver'] == 'GTiff'
        assert len(result['errors']) < 1
        assert len(result['warnings']) < 1

    def test_DataAcquisitionGateway_001(self):
        """Test DataAcquisitionGateway module.

        Example of unit test.
        """
        pass

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
