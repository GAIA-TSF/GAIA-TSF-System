import pystac

from lib.base import GaiaBase, SubsystemId


class MetadataValidator(GaiaBase):
    """
    The automatic validation of generated metadata during ingestion.
    """

    def __init__(self):
        """Initialize metadata validator."""
        super().__init__(SubsystemId.DPR)

    def validate(self, metadata_path):
        """
        Validate provided metadata.

        :param str metadata_path: metadata to be validated
        """
        catalog = pystac.read_file(metadata_path)
        catalog.validate()
        # TODO: STACValidationError -> raise GaiaMetadataError
