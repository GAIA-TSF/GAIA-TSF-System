import hashlib
import requests

from lib.base import GaiaBase, SubsystemId

class SdiUtils(GaiaBase):
    def __init__(self):
        """Simple SDI client for searching and downloading assets."""
        GaiaBase.__init__(self, SubsystemId.SDI)

    def file_md5(self, path):
        """
        Compute MD5 hash of a file.
        """
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def delete_item_and_collection(self, stac_url, collection_id, item_id):
        requests.delete(
            f'{stac_url}/collections/{collection_id}/items/{item_id}',
        )

        requests.delete(
            f'{stac_url}/collections/{collection_id}',
        )
