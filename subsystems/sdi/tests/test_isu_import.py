from pathlib import Path
from sdi.isu_import import IsuDataZipImporter


class TestIsuDataZipImporter:
    def test_import(self):
        base_dir = Path(__file__).parent
        zip_path = base_dir / 'assets' / 'isu_sample_data.zip'

        assert zip_path.exists()

        importer = IsuDataZipImporter(zip_path=zip_path)

        importer.import_zip()
