from __future__ import annotations
from abc import ABC, abstractmethod
import os

from lib.config import ProjectConfigReader


class DataAcquisitionBackend(ABC):
    def __init__(self):
        self.config: dict = {}
        self.data_dir: str = None
        self.project_config: ProjectConfigReader = None

    def set_config(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def _search(self, *args, **kwargs):
        """Generic search interface"""
        pass

    def search(self, *args, **kwargs):
        """Generic search interface"""
        if kwargs.get('geom') is None and self.project_config is not None:
            kwargs['geom'] = self.project_config.aoi()

        return self._search(*args, **kwargs)

    @abstractmethod
    def _download(self, *args, **kwargs) -> str:
        """Generic download interface"""
        pass

    def download(self, *args, **kwargs) -> str:
        """Generic download interface"""
        if not os.path.isabs(kwargs['target_dir']):
            kwargs['target_dir'] = os.path.join(self.data_dir, kwargs['target_dir'])

        return self._download(*args, **kwargs)

    @abstractmethod
    def _download_all(self, *args, **kwargs) -> str:
        """Generic download interface"""
        pass

    def download_all(self, *args, **kwargs) -> str:
        """Generic download interface"""
        if not os.path.isabs(kwargs['target_dir']):
            kwargs['target_dir'] = os.path.join(self.data_dir, kwargs['target_dir'])

        return self._download_all(*args, **kwargs)
