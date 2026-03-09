from enum import Enum

from .config import SettingsReader, ProjectConfigReader
from subsystems.qcl.logger import Logger


class SubsystemId(Enum):
    ISU = 1
    EOU = 2
    DPR = 3
    SDI = 4
    QCL = 5
    DAG = 6
    MAP = 7
    ALE = 8
    NTF = 9
    REP = 10
    VID = 11


class BaseObject:
    def __init__(self, sid: SubsystemId, project_file: str = None):
        """Initialize base object.

        Reads internal system settings and project configuration file
        if defined.

        :param SubsystemId sid: subsystem id
        :param str project_file: path to project file to be read or None
        """
        self.sid = sid
        # initialize internal settings
        self.settings = SettingsReader()
        # initialize logger
        self.logger = Logger(subsystem=sid.name)
        self.logger.debug(f'{self.__class__.__name__} initialized')
        if project_file is not None:
            self.project_config = ProjectConfigReader(project_file)
        else:
            self.project_config = None
