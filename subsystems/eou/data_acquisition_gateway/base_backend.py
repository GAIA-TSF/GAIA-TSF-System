from __future__ import annotations
from abc import ABC, abstractmethod
import os
import time
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subsystems.qcl.logger import Logger


class DataAcquisitionBackend(ABC):
    def __init__(self, data_dir: Path, logger: Logger):
        """
        Abstract base class for EO data acquisition backends.

        Implementations provide a common interface for searching and
        downloading Earth Observation products from different providers.

        :param data_dir: Directory used to store downloaded products.
        :type data_dir: pathlib.Path
        :param logger: Logger instance used for reporting backend activity.
        :type logger: logging.Logger
        """
        self.config: dict = {}
        self.data_dir = data_dir
        self.logger = logger

    def set_config(self, config: dict) -> None:
        """
        Set backend configuration.

        The configuration is stored internally and authentication
        credentials are validated. If the password is not provided
        in the configuration, it is read from the
        ``GAIA_EOU_AUTH_CREDENTIALS_PASSWORD`` environment variable,
        when available.

        :param config: Backend configuration dictionary.
        :type config: dict
        """
        self.config = config

        # check password
        if 'cop_dataspace' in self.config:  # eodag
            credentials = self.config['cop_dataspace']['auth']['credentials']
        else:
            credentials = self.config['auth']['credentials']
        if not credentials['password'] and os.environ.get(
            'GAIA_EOU_AUTH_CREDENTIALS_PASSWORD'
        ):
            credentials['password'] = os.environ['GAIA_EOU_AUTH_CREDENTIALS_PASSWORD']

    @abstractmethod
    def _search(self, *args, **kwargs):
        """Generic search interface"""
        pass

    def search(self, *args, **kwargs):
        """Generic search interface"""
        self.logger.info(f'Search filters: {args=}, {kwargs=}')
        result = self._search(*args, **kwargs)
        self.logger.info(f'Number of found products: {len(result)}')

        return result

    @abstractmethod
    def _download(self, *args, **kwargs) -> str:
        """Generic download interface"""
        pass

    def download(self, *args, **kwargs) -> str:
        """Generic download interface"""
        self.__set_target_dir(kwargs)
        self.logger.info(f'Downloading {args=}, {kwargs=}')
        start = time.time()
        data_path = self._download(*args, **kwargs)
        elapsed_minutes = (time.time() - start) / 60
        self.logger.info(f'Downloaded product path: {data_path}')
        self.logger.debug(f'Download completed in {elapsed_minutes:.2f} minutes')

        return data_path

    @abstractmethod
    def _download_all(self, *args, **kwargs) -> str:
        """Generic download interface"""
        pass

    def download_all(self, *args, **kwargs) -> str:
        """Generic download interface"""
        self.__set_target_dir(kwargs)
        self.logger.info(f'Downloading {args=}, {kwargs=}')
        start = time.time()
        data_path = self._download_all(*args, **kwargs)
        elapsed_minutes = (time.time() - start) / 60
        self.logger.info(f'Downloaded product path: {data_path}')
        self.logger.debug(f'Download completed in {elapsed_minutes:.2f} minutes')

        return data_path

    def __set_target_dir(self, kwargs):
        """
        Resolve the target directory path.

        If ``target_dir`` is specified as a relative path, it is resolved
        against the backend data directory. Absolute paths are left unchanged.

        :param kwargs: Keyword arguments containing the ``target_dir`` entry.
        :type kwargs: dict
        """
        if not Path(kwargs['target_dir']).is_absolute():
            kwargs['target_dir'] = self.data_dir / kwargs['target_dir']
