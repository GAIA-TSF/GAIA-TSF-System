from typing import Dict

import os
import logging
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.exc import OperationalError
from sqlalchemy_utils import database_exists, create_database

from lib.base import SubsystemId
from lib.exceptions import GaiaConfigError

Base = declarative_base()


class DbRecord(Base):
    """Table definition: log"""

    __tablename__ = 'log'

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    subsystem_id = Column(String(10), ForeignKey('subsystem.id'), nullable=False)
    timestamp = Column(DateTime(), nullable=False)
    level_id = Column(Integer, ForeignKey('log_level.id'), nullable=False)
    message = Column(String, nullable=False)
    site_id = Column(String(255), nullable=True)
    project_name = Column(String(255), nullable=True)
    pid = Column(
        Integer,
        nullable=False,
        index=True,
    )
    log_level = relationship('DbLogLevel', back_populates='log')
    subsystem = relationship('DbSubsystem', back_populates='log')


class DbLogLevel(Base):
    """Table definition: log_level"""

    __tablename__ = 'log_level'

    id = Column(Integer, primary_key=True, nullable=False)
    name = Column(String(255), nullable=False)
    severity = Column(Integer, nullable=False)

    log = relationship('DbRecord', back_populates='log_level')


class DbSubsystem(Base):
    """Table definition: subsystems"""

    __tablename__ = 'subsystem'

    id = Column(String(10), primary_key=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(String(255), nullable=True)

    log = relationship('DbRecord', back_populates='subsystem')


class DbLogger(logging.Handler):
    def __init__(self, db_config: Dict[str, str]):
        """A custom handler for DB logging.

        :param dict db_config:
            Database configuration for log persistence (e.g. PostgreSQL/PostGIS).
            Expected keys include ``host``, ``port``, ``dbname``, ``user``, and ``password``.
        """

        super(DbLogger, self).__init__()

        self._session = None
        self._session_maker = None
        self._set_session(db_config)
        self._define_log_levels()
        self._define_subsystems()

    def __del__(self):
        """Destructor."""
        self._close_all()

    def _close_all(self):
        """Close all sessions."""
        if self._session:
            self._session.close()
        if self._session_maker:
            self._session_maker.close_all()
        self._session = self._session_maker = None

    def _set_session(self, db_config: Dict[str, str]):
        """Create a new session.

        :param dict db_config:
            Database configuration for log persistence (e.g. PostgreSQL/PostGIS).
            Expected keys include ``host``, ``port``, ``dbname``, ``user``, and ``password``.
        """
        # create session if not already defined
        if not self._session:
            if self._session_maker:
                self._session_maker.close_all()
            engine = create_engine(
                f'postgresql+psycopg://{db_config["user"]}:{db_config["password"]}'
                f'@{db_config["host"]}:{db_config["port"]}/{db_config["dbname"]}'
            )
            if not database_exists(engine.url):
                create_database(engine.url)

            Base.metadata.bind = engine
            self._session_maker = sessionmaker(engine)
            self._session = self._session_maker()

        # create tables
        try:
            Base.metadata.create_all(Base.metadata.bind)
        except OperationalError as e:
            self._close_all()
            raise GaiaConfigError('{}'.format(e))

    def emit(self, record: logging.LogRecord):
        """Format the record and store in DB log tables.

        Overrides the logging.Handler.emit function.

        :param logging.LogRecord record: record to emit
        """
        if not self._session_maker or not self._session:
            return

        try:
            site_id = record.site_id
            project_name = record.project_name
        except AttributeError:
            site_id = project_name = None
        db_record = DbRecord(
            subsystem_id=record.subsystem,
            timestamp=datetime.strptime(record.asctime, '%Y-%m-%d %H:%M:%S,%f'),
            level_id=record.levelno,
            message=record.getMessage(),
            site_id=site_id,
            project_name=project_name,
            pid=os.getpid(),
        )
        self._session.add(db_record)
        self._session.commit()

    def _define_log_levels(self):
        """Creates table log_level"""
        if self._session.query(DbLogLevel).count() != 0:
            return

        log_levels = [
            DbLogLevel(id=logging.DEBUG, name='DEBUG', severity=logging.DEBUG),
            DbLogLevel(id=logging.INFO, name='INFO', severity=logging.INFO),
            DbLogLevel(id=logging.WARNING, name='WARNING', severity=logging.WARNING),
            DbLogLevel(id=logging.ERROR, name='ERROR', severity=logging.ERROR),
            DbLogLevel(id=logging.CRITICAL, name='CRITICAL', severity=logging.CRITICAL),
        ]

        self._session.add_all(log_levels)
        self._session.commit()

    def _define_subsystems(self):
        """Creates table subsystem"""
        if self._session.query(DbSubsystem).count() != 0:
            return

        descriptions = {
            'ALE': 'Alert & Decision Support Engine',
            'DAG': 'Data Aggregation',
            'DPR': 'Data Processor',
            'EOU': 'Earth Observation Data Uploader',
            'ISU': 'In-Situ Data Uploader',
            'MAP': 'Machine Learning & Predictive Analytics',
            'NTF': 'Notifications',
            'QCL': 'Quality Control and Logging Layer',
            'REP': 'Reporting & Compliance',
            'SDI': 'Spatial Data Infrastructure',
            'VID': 'Visualisation & Dashboard',
        }

        subsystems = [
            DbSubsystem(
                id=subsystem.name,
                name=descriptions[subsystem.name],
                description=None,
            )
            for subsystem in SubsystemId
        ]

        self._session.add_all(subsystems)
        self._session.commit()
