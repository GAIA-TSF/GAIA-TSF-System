from earth_observation_data_uploader.data_acquisition_gateway import DataAcquisitionGateway
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon

def load_geom_from_kmz(kmz_path):
    with ZipFile(kmz_path, 'r') as kmz:
        kml_filename = [f for f in kmz.namelist() if f.endswith('.kml')][0]
        with kmz.open(kml_filename) as kml_file:
            tree = ET.parse(kml_file)
            root = tree.getroot()

            ns = {'kml': 'http://www.opengis.net/kml/2.2'}

            polygons = []
            for polygon in root.findall('.//kml:Polygon', ns):
                coords_text = polygon.find('.//kml:coordinates', ns).text.strip()
                coords = []
                for line in coords_text.split():
                    lon, lat, *_ = map(float, line.split(','))
                    coords.append((lon, lat))
                polygons.append(Polygon(coords))

            if len(polygons) == 1:
                return polygons[0]
            else:
                return MultiPolygon(polygons)

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
        provider = "cop_dataspace"
        start = "2026-01-01"
        end = "2026-01-29"
        geom = load_geom_from_kmz("earth_observation_data_uploader/tests/sample_data/area_intervencao.kmz")
        product_type = "S2_MSI_L2A"

        result = module.search(
            provider=provider,
            start=start,
            end=end,
            geom=geom,
            productType=product_type
        )

        # count=True ???

        assert isinstance(result, SearchResult)
        assert len(result) > 0
        assert result[0].product_type == product_type

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
