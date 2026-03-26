_logger = None


def set_logger(logger):
    global _logger
    _logger = logger


def get_logger():
    return _logger
