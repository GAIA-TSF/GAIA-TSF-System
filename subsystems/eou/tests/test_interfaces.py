class TestInterfaces:
    def test_IF1_001(self):
        """Test IF1 DataProcessor.

        Example of integration test.
        """
        pass

    def test_IF1_002(self):
        """Test IF1 DataProcessor.

        Another example of integration test.
        """
        pass

    def test_IF2_001(self):
        """Test IF2 QualityControlLogging Layer.

        Example of integration test.
        """
        pass


# Added dispatcher integration tests for EOU interfaces
from subsystems.eou import EarthObservationDataUploader


class MockDprIngestEo:
    def __init__(self):
        self.called = False
        self.payload = None

    def ingest_eo(self, payload):
        self.called = True
        self.payload = payload


class MockDprIngestRaw:
    def __init__(self):
        self.called = False
        self.payload = None

    def ingest_raw(self, payload):
        self.called = True
        self.payload = payload


class MockQcl:
    def __init__(self):
        self.called = False
        self.args = None

    def process_incoming_data(self, data_type, data, metadata, dataset_id):
        self.called = True
        self.args = (data_type, data, metadata, dataset_id)


def test_forward_to_dpr_prefers_ingest_eo():
    eou = EarthObservationDataUploader()
    mock = MockDprIngestEo()
    eou.set_dpr_service(mock)
    payload = {"data_type": "eo_raster", "file_path": "/tmp/data.jp2", "metadata": {"a": 1}, "dataset_id": "ds1"}
    eou.forward_to_dpr(payload)
    assert mock.called
    assert mock.payload == payload


def test_forward_to_dpr_fallback_ingest_raw():
    eou = EarthObservationDataUploader()
    mock = MockDprIngestRaw()
    eou.set_dpr_service(mock)
    payload = {"data_type": "eo_raster", "file_path": "/tmp/data.jp2", "metadata": {}, "dataset_id": "ds2"}
    eou.forward_to_dpr(payload)
    assert mock.called
    assert mock.payload == payload


def test_forward_to_qcl_calls_qcl():
    eou = EarthObservationDataUploader()
    mock = MockQcl()
    eou.set_qcl_service(mock)
    eou.forward_to_qcl("eo_raster", "/tmp/data.jp2", {"a": 1}, "ds3")
    assert mock.called
    assert mock.args == ("eo_raster", "/tmp/data.jp2", {"a": 1}, "ds3")
