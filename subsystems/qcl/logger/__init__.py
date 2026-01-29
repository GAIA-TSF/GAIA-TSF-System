import os
import logging
import logging.config

class QCLLogger(logging.getLoggerClass()):
    pass

def logger():
    """Return a logger.
    """
    logging.config.fileConfig(
        os.path.join(os.path.dirname(__file__), 'logging.conf')
    )

    logging.setLoggerClass(QCLLogger)
    logger = logging.getLogger('GAIA-TSF')

    return logger

Logger = logger()
