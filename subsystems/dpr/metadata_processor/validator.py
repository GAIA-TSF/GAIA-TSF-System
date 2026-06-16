import pystac
import requests

from lib.base import GaiaBase, SubsystemId
from lib.config import SettingsReader


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
        # get collections
        settings = SettingsReader()
        stac_url = settings['sdi']['stac']['url']
        r = requests.get(f'{stac_url}/collections')
        r.raise_for_status()
        data = r.json()

        # read input metadata file
        catalog = pystac.read_file(metadata_path)

        # add collections
        for col_json in data['collections']:
            collection = pystac.Collection.from_dict(col_json)
            catalog.add_child(collection)

        catalog.validate()
        # TODO: STACValidationError -> raise GaiaMetadataError
