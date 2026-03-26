from lib.logging_provider import get_logger


class GaiaError(Exception):
    log_level = 'critical'  # default

    def __init__(self, msg: str):
        super().__init__(msg)

        logger = get_logger()
        if logger:
            log_method = getattr(logger, self.log_level, None)
            if log_method:
                log_method(msg)


class GaiaConfigError(GaiaError):
    """Configuration error."""

    pass


class GaiaUnsupportedDataError(GaiaError):
    """Unsupported data source or operation."""

    pass


class GaiaReadDataError(GaiaError):
    """Read / Parse data error."""

    pass
