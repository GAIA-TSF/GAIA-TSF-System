import sys
from pathlib import Path

subsystems_path = str(Path(__file__).parent.parent / "subsystems")

if subsystems_path not in sys.path:
    sys.path.insert(0, subsystems_path)

from osgeo import ogr, osr

from lib.config import ProjectConfigReader
from subsystems.eou.data_acquisition_gateway import DataAcquisitionGateway


def load_geom_from_wkt(wkt_string: str) -> list[float]:
    geom = ogr.CreateGeometryFromWkt(wkt_string)
    if geom is None:
        raise RuntimeError("Invalid WKT geometry")

    if not geom.IsValid():
        raise RuntimeError("Geometry is not valid")

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    geom.AssignSpatialReference(srs)
    srs.AutoIdentifyEPSG()
    auth = srs.GetAuthorityName(None)
    code = srs.GetAuthorityCode(None)

    if auth != "EPSG" or code != "4326":
        raise RuntimeError(f"Unsupported CRS: {auth}:{code}")

    lonmin, lonmax, latmin, latmax = geom.GetEnvelope()

    return [lonmin, latmin, lonmax, latmax]


class TestSentinel1Workflow:
    search_filter = {
        'provider': 'cop_dataspace',
        'start': '2026-01-01',
        'end': '2026-01-29',
        'productType': 'S1_SAR_SLC',
    }

    def test_download(self):
        config = ProjectConfigReader(
            Path(__file__).parent / 'projects' / 'jagersfontein.yml'
        )

        assert "POLYGON" in config["project"]["aoi"]["geom"]

        module = DataAcquisitionGateway()
        results = module.search(
            geom=load_geom_from_wkt(config["project"]["aoi"]["geom"]),
            **self.search_filter,
        )

        assert len(results) > 0

        config_eodag = 'subsystems/eou/tests/eodag_config.yml'

        module.set_config(config_eodag)
        ql_path = module.download(results[0], quicklook=True)

        assert Path(ql_path).exists()
