from logging import LoggerAdapter


class GaiaError(Exception):
    def __init__(self, msg: str, logger: LoggerAdapter):
        """Initialize generic GaiaError.

        :param str msg: error message
        :param LoggerAdapter logger: logger to be used
        """
        if logger is not None:
            logger.critical(msg)


class GaiaConfigError(GaiaError):
    """Configuration error."""

    pass


class GaiaDataError(GaiaError):
    """Read / Parse data error."""

    pass
