from pathlib import Path


def get_data_path(filename):
    return str(Path(__file__).parent / filename)
