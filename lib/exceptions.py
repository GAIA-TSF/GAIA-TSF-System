from logging import LoggerAdapter


class GaiaError(Exception):
    def __init__(self, msg: str, logger: LoggerAdapter):
        """Initialize generic GaiaError.

        :param str msg: error message
        :param LoggerAdapter logger: logger to be used
        """
        if logger is not None:
            logger.error(msg)


class GaiaConfigError(GaiaError):
    """GAIA configuration error."""

    pass


class GaiaDataError(GaiaError):
    """GAIA read/parse data error."""

    pass


class GaiaSdiError(GaiaError):
    """GAIA SDI-related error."""

    pass
