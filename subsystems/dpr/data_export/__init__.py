import os
import zipfile
from pathlib import Path

from lib.base import GaiaBase, SubsystemId
from lib.config import SettingsReader

class DataExporter(GaiaBase):
    """Data Extraction module acts as the central logic module for
    ingestion. It receives inputs from individual subsystems and performing the
    necessary extraction and preparation steps before handing the data off to
    the spatial data infrastructure.
    """

    def __init__(self, input_file: Path, input_metadata: Path):
        """
        Initialize the DPR Data Exporter.

        :param input_file: Path to the input file to be processed.
        :type input_file: Path

        
        :param input_metadata: Metadata associated with the input file.
        :type input_metadata: Path
        """
        super().__init__(SubsystemId.DPR)
        self.input_file = input_file
        self.input_metadata = input_metadata

    def create_sdi_package(self, output_file: Path | None = None) -> Path:
        """
        Create an SDI package as a ZIP archive containing the input data file
        and its metadata file.

        If ``output_file`` is not provided, a temporary ZIP file is created in
        the configured temporary directory.

        :param output_file: Path to the output ZIP archive. If ``None``, a temporary file is created.
        :type output_file: Path | None
        
        :returns: Path to the created ZIP archive.
        :rtype: Path
        """
        if output_file is None:
            output_file = SettingsReader().temp_file('.zip')

        with zipfile.ZipFile(output_file, 'w') as zip:
            zip.write(self.input_file, arcname=self.input_file.name)
            zip.write(self.input_metadata, arcname=self.input_metadata.name)

        self.logger.info(f"ZIP package created for {self.input_file} and {self.input_metadata}: {output_file}")
        return output_file
