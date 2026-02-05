import os
import logging
import logging.config


class Logger:
    _configured = False

    def __new__(cls, name='GAIA-TSF', **context):
        if not cls._configured:
            logging.config.fileConfig(
                os.path.join(os.path.dirname(__file__), 'logging.conf'),
                disable_existing_loggers=False,
            )
            cls._configured = True

        base_logger = logging.getLogger(name)
        return logging.LoggerAdapter(base_logger, context)
