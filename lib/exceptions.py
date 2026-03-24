try:
    # TODO
    from subsystems.qcl.logger import Logger
    _logger = Logger(subsystem='QCL')
except ImportError:
    _logger = None

class GaiaError(Exception):
    def __init__(self, msg: str):
        """Initialize generic GaiaError.

        :param str msg: error message
        """
        _logger.critical(msg)


class GaiaConfigError(GaiaError):
    """Configuration error."""

    pass


class GaiaUnsupportedDataError(GaiaError):
    """Unsupported data source or operation."""


class GaiaReadDataError(GaiaError):
    """Read / Parse data error."""
