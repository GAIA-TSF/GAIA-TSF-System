from __future__ import annotations
from abc import ABC, abstractmethod


class DataAcquisitionBackend(ABC):
    def __init__(self):
        self.config: dict = {}

    def set_config(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def search(self, *args, **kwargs):
        """Generic search interface"""
        pass

    @abstractmethod
    def download(self, *args, **kwargs) -> str:
        """Generic download interface"""
        pass
