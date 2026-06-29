from pathlib import Path


class TestUtils:
    """Test utils."""

    @staticmethod
    def get_data_path(filename):
        """Get test data path.

        :param str filename: test data file name

        :return: full path to test data file
        :rtype: Path
        """
        return Path(__file__).parent / 'assets' / filename

    @staticmethod
    def get_project_config_path(project_name):
        """Return the full path to the configuration file for a given project.

        :param project_name: Name of the project.
        :type project_name: str
        :return: full path to the project's configuration file.
        :rtype: Path
        """
        return Path(__file__).parent / 'projects' / project_name / 'config.yaml'
