import os
import zipfile


class DataExporter:
    """Data Extraction module acts as the central logic module for
    ingestion. It receives inputs from individual subsystems and performing the
    necessary extraction and preparation steps before handing the data off to
    the spatial data infrastructure.
    """

    def __init__(self, input_file, input_metadata):
        self.input_file = input_file
        self.input_metadata = input_metadata

    def create_sdi_package(self, output_file):
        with zipfile.ZipFile(output_file, 'w') as zip:
            zip.write(self.input_file, arcname=os.path.basename(self.input_file))
            zip.write(
                self.input_metadata, arcname=os.path.basename(self.input_metadata)
            )
