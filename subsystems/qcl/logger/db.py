import os
import logging
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.exc import OperationalError
from sqlalchemy_utils import database_exists, create_database

Base = declarative_base()


class DbConnectionError(Exception):
    """DB connection error."""

    pass


class DbRecord(Base):
    """Table definition: logs"""

    __tablename__ = 'logs'

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    subsystem = Column(String(3), nullable=False)
    timestamp = Column(DateTime(), nullable=False)
    level = Column(Integer, nullable=False)
    message = Column(String, nullable=False)
    project = Column(String(255), nullable=True, index=True)
    pid = Column(
        Integer,
        nullable=False,
    )


class DbLogger(logging.Handler):
    """A custom handler for DB logging."""

    def __init__(self, db_config):
        super(DbLogger, self).__init__()

        self._session = None
        self._session_maker = None
        self._set_session(db_config)

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

    def _set_session(self, db_config):
        """Create a new session.

        :param TODO
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
            raise DbConnectionError('{}'.format(e))

    def emit(self, record):
        """Format the record and store in DB log tables.

        Overrides the logging.Handler.emit function.

        :param record: record to emit
        """
        if not self._session_maker or not self._session:
            return

        db_record = DbRecord(
            subsystem=record.subsystem,
            timestamp=datetime.strptime(record.asctime, '%Y-%m-%d %H:%M:%S,%f'),
            level=record.levelno,
            message=record.getMessage(),
            project=None,
            pid=os.getpid(),
        )
        self._session.add(db_record)
        self._session.commit()
