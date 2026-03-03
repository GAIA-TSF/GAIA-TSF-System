import hashlib
from qcl.logger import Logger


class SdiUtils:
    def __init__(self):
        """Simple SDI client for searching and downloading assets."""
        self.id = 'SDI'
        self.logger = Logger(subsystem=self.id)

    def file_md5(self, path):
        """
        Compute MD5 hash of a file.
        """
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
