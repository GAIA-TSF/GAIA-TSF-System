import os
import sys
import logging


class Logger:
    _configured = False

    def __new__(cls, name='GAIA-TSF', **context):
        base_logger = logging.getLogger(name)

        if not cls._configured:
            base_logger.setLevel(logging.DEBUG)

            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(subsystem)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            base_logger.addHandler(handler)
            base_logger.propagate = False

            cls._configured = True

        return logging.LoggerAdapter(base_logger, context)
