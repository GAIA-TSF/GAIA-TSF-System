import os
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
    def __init__(self, sid: SubsystemId, project_path: str = None):
        """Initialize base GAIA-TSF object.

        Reads internal system settings and project configuration file
        if defined. Initialize logger.

        :param SubsystemId sid: subsystem id
        :param str project_path: path to project file to be read or None
        """
        from subsystems.qcl.logger import Logger  # avoid circular import

        self.sid = sid
        # initialize internal settings
        self.settings = SettingsReader()

        _project_path = (
            project_path
            if project_path is not None
            else os.environ.get('GAIA_PROJECT_PATH')
        )

        # initialize logger
        self.logger = Logger(
            subsystem=sid.name,
            db_config=self.settings['qcl']['logger'].get('db'),
            project_path=_project_path,
        )
        self.logger.debug(f'{self.__class__.__name__} initialized')

        # read project configuration
        if project_path is not None and os.environ.get('GAIA_PROJECT_PATH') is not None:
            self.logger.warning(
                f'Both project_path argument and GAIA_PROJECT_PATH variable defined. Using project_path ({project_path})'
            )

        if _project_path is not None:
            self.project_config = ProjectConfigReader(_project_path)
            self.logger.info(f'Project configuration read from {_project_path}')
        else:
            self.project_config = None
