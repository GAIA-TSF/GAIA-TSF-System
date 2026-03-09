from qcl.logger import Logger

_logger = Logger('QCL')

class GaiaError(Exception):
    def __init__(self, msg: str):
        """Initialize generic GaiaError.

        :param str msg: error message
        """
        _logger.critical(msg)

class GaiaConfigError(Exception):
    pass
