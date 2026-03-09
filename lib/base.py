from .config import SettingsReader, ProjectConfigReader

class BaseObject:
    def __init__(self, project_file: str = None):
        """Initialize base object.

        Reads internal system settings and project configuration file
        if defined.

        :param str project_file: path to project file to be read or None
        """
        self.settings = SettingsReader()
        if project_file is not None:
            self.project_config = ProjectConfigReader(project_file)
        else:
            self.project_config = None
