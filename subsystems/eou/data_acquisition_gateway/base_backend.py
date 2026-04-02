from __future__ import annotations
from abc import ABC, abstractmethod


class DataAcquisitionBackend(ABC):
    @abstractmethod
    def search(self, *args, **kwargs):
        """Generic search interface"""
        pass

    @abstractmethod
    def download(self, *args, **kwargs):
        """Generic download interface"""
        pass

    @abstractmethod
    def set_config(self, config: dict) -> None:
        """Set configuration file"""
        pass
