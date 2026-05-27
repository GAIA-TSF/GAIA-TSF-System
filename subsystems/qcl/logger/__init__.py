from typing import Dict

import sys
import logging

from subsystems.qcl.logger.db import DbLogger

from lib.config import SettingsReader


class Logger:
    _configured = False

    def __new__(cls, db_config: Dict[str, str], name: str = 'GAIA-TSF', **context):
        """
        Create or retrieve a configured logger instance.

        This class implements a singleton-like pattern for logger configuration.
        The underlying logger (from the standard ``logging`` module) is configured
        only once per application lifecycle. Subsequent instantiations reuse the
        existing configuration while allowing additional contextual data.

        :param dict db_config:
            Database configuration for log persistence (e.g. PostgreSQL/PostGIS).
            Expected keys include ``host``, ``port``, ``dbname``, ``user``, and ``password``.

        :param str name:
            Name of the logger instance. Defaults to ``'GAIA-TSF'``.

        :param context:
            Arbitrary keyword arguments representing contextual metadata
            (e.g. subsystem).
        :type context: dict

        :return: Configured instance of ``logging.Logger``.
        :rtype: logging.Logger

        :raises Exception:
            May raise exceptions depending on logging handler initialization
            (e.g. database connection errors).
        """
        base_logger = logging.getLogger(name)

        if not cls._configured:
            config = SettingsReader()

            try:
                level = logging.getLevelName(config['qcl']['logger']['level'])
            except KeyError:
                level = logging.DEBUG
            base_logger.setLevel(level)

            handler = logging.StreamHandler(sys.stdout)
            formatter_str = (
                '%(asctime)s - %(name)s - %(subsystem)s - %(levelname)s - %(message)s'
            )
            if 'project_path' in context:
                formatter_str += ' [%(project_path)s]'
            formatter = logging.Formatter(formatter_str)
            handler.setFormatter(formatter)
            base_logger.addHandler(handler)
            if db_config is not None:
                base_logger.addHandler(DbLogger(db_config))
            base_logger.propagate = False

            cls._configured = True

        return logging.LoggerAdapter(base_logger, context)
