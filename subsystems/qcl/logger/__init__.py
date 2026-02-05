import os
import logging
import logging.config


def Logger(name='GAIA-TSF', **context):
    logging.config.fileConfig(
        os.path.join(os.path.dirname(__file__), 'logging.conf'),
        disable_existing_loggers=False,
    )

    base_logger = logging.getLogger(name)
    return logging.LoggerAdapter(base_logger, context)
