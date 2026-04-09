from abc import ABC, abstractmethod
from logging import LoggerAdapter
from typing import Any


class DispatcherBase(ABC):
    """
    Abstract base class for active output dispatchers.

    Provides shared initialisation and a guard utility so that concrete
    dispatcher implementations only need to define their ``dispatch()`` logic.
    """

    def __init__(self, logger: LoggerAdapter):
        """
        :param logger: Logger instance used for info/warning messages.
        """
        self._logger = logger

    @abstractmethod
    def dispatch(self, *args: Any, **kwargs: Any) -> None:
        """Send data to the target downstream service."""

    def _guard(self, service: Any, interface_id: str) -> bool:
        """
        Check that a downstream service is available before dispatching.

        Logs a warning and returns ``False`` when the service is ``None``,
        allowing callers to exit early with a single ``if not self._guard(...)``
        check.

        :param service: The downstream service instance (may be ``None``).
        :param interface_id: Interface label used in the warning message (e.g. ``'QCL_I_1'``).
        :returns: ``True`` if the service is available, ``False`` otherwise.
        :rtype: bool
        """
        if service is None:
            self._logger.warning(f'[{interface_id}] No service configured. Skipping.')
            return False
        return True
