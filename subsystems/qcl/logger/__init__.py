import logging
from pathlib import Path

# TBD: remove when root path defined
# https://github.com/GAIA-TSF/GAIA-TSF-System/issues/145
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from lib.config import SettingsReader


class Logger:
    _configured = False

    def __new__(cls, name='GAIA-TSF', **context):
        base_logger = logging.getLogger(name)

        if not cls._configured:
            config = SettingsReader()

            try:
                level = logging.getLevelName(config['qcl']['logger']['level'])
            except KeyError:
                level = logging.DEBUG
            base_logger.setLevel(level)

            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(subsystem)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            base_logger.addHandler(handler)
            base_logger.propagate = False

            cls._configured = True

        return logging.LoggerAdapter(base_logger, context)
