from typing import Dict

import sys
import logging

from subsystems.qcl.logger.db import DbLogger

from lib.config import SettingsReader


# define custom level
TASK_STARTED = 101
TASK_FINISHED = 102
TASK_FAILED = 103
logging.addLevelName(TASK_STARTED, 'TASK_STARTED')
logging.addLevelName(TASK_FINISHED, 'TASK_FINISHED')
logging.addLevelName(TASK_FAILED, 'TASK_FAILED')


class CustomLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter subclass that exposes DB-related task helpers.

    This adapter keeps the standard LoggerAdapter behavior (injecting the
    provided context into each record) and adds convenience methods to
    create and attach a task_id stored in the DB-backed handler.
    """

    def _db_handler(self):
        """Return DbLogger handler instance or raise if not configured."""
        for handler in getattr(self.logger, 'handlers', []):
            if isinstance(handler, DbLogger):
                return handler
        raise Exception('DbLogger not found')

    def _next_task_id(self):
        """Obtain the next task id from the DbLogger."""
        return self._db_handler().next_task_id()

    def __log_task(self, level: int, obj, *, task_id=None, msg_suffix=None, persist=False, **kwargs):
        """Internal helper to log task-related events and avoid duplication.

        - level: numeric logging level
        - obj: related object (used for class name in message)
        - task_id: optional task id to attach; if None, tries adapter.extra
        - msg_suffix: string appended to class name (e.g. 'started')
        - persist: if True, store task_id in adapter.extra for future logs
        """
        name = obj.__class__.__name__
        msg = f"{name} {msg_suffix}" if msg_suffix else name

        # ensure adapter extra is a mutable dict
        if not isinstance(self.extra, dict):
            self.extra = dict(self.extra or {})

        extra = kwargs.pop('extra', {})

        # resolve task_id: explicit param -> adapter extra -> None
        resolved_task_id = task_id if task_id is not None else self.extra.get('task_id')
        if resolved_task_id is not None:
            extra.update({'task_id': resolved_task_id})

        if persist and resolved_task_id is not None:
            self.extra['task_id'] = resolved_task_id

        self.log(level, msg, extra=extra, **kwargs)
        return resolved_task_id

    def task_started(self, obj, **kwargs):
        """Create new task id, persist it in adapter context and log start."""
        task_id = self._next_task_id()
        return self.__log_task(TASK_STARTED, obj, task_id=task_id, msg_suffix='started', persist=True, **kwargs)

    def task_finished(self, obj, **kwargs):
        """Log a task finished event using either provided or persisted task_id."""
        task_id = kwargs.pop('task_id', None)
        return self.__log_task(TASK_FINISHED, obj, task_id=task_id, msg_suffix='finished', **kwargs)

    def task_failed(self, obj, **kwargs):
        """Log a task failed event using either provided or persisted task_id."""
        task_id = kwargs.pop('task_id', None)
        return self.__log_task(TASK_FAILED, obj, task_id=task_id, msg_suffix='failed', **kwargs)


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

        Returns a CustomLoggerAdapter so task_* helpers are available on the
        returned object (previously the factory returned a plain
        logging.LoggerAdapter which lacked those methods).
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
            if 'site_id' in context and 'project_name' in context:
                formatter_str += ' [%(site_id)s/%(project_name)s]'
            formatter = logging.Formatter(formatter_str)
            handler.setFormatter(formatter)
            base_logger.addHandler(handler)
            if db_config is not None:
                base_logger.addHandler(DbLogger(db_config))
            base_logger.propagate = False

            cls._configured = True

        # Return the custom adapter so callers can use task_started()/task_finished()
        return CustomLoggerAdapter(base_logger, context)
