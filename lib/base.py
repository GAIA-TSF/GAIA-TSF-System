from enum import Enum

from .config import SettingsReader, ProjectConfigReader


class SubsystemId(Enum):
    """Subsystem ID.

    - ALE	Alert & Decision Support Engine
    - DAG	Data Aggregation
    - DPR	Data Processor
    - EOU	Earth Observation Data Uploader
    - ISU	In-Situ Data Uploader
    - MAP	Machine Learning & Predictive Analytics
    - NTF	Notifications
    - QCL	Quality Control and Logging Layer
    - REP	Reporting & Compliance
    - SDI	Spatial Data Infrastructure
    - VID	Visualisation & Dashboard
    """
    ALE = 1
    DAG = 2
    DPR = 3
    EOU = 4
    ISU = 5
    MAP = 6
    NTF = 7
    QCL = 8
    REP = 9
    SDI = 10
    VID = 11


class GaiaBase:
    def __init__(self, sid: SubsystemId, project_file: str = None):
        """Initialize base GAIA-TSF object.

        Reads internal system settings and project configuration file
        if defined. Initialize logger.

        :param SubsystemId sid: subsystem id
        :param str project_file: path to project file to be read or None
        """
        from subsystems.qcl.logger import Logger # avoid circular import

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
