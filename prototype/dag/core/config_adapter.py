from types import SimpleNamespace


def to_namespace(d):
    """Recursively convert a dict to a SimpleNamespace, 
    allowing for dot-acces""
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{
            str(k): to_namespace(v)   
            for k, v in d.items()
        })
    elif isinstance(d, list):
        return [to_namespace(i) for i in d]
    return d


class ConfigAdapter:
    """Adapter to convert raw YAML config into a unified, 
    dot-accessible format for DAG building.
    This class also injects unified sections (variables, features, datasets)
    to simplify module instantiation and dependency injection. 
    Arguments:
        raw_config (dict): The raw configuration loaded from YAML.
        variable (str): The name of the variable/pipeline to build.
    Usage:
    config = load_yaml("config.yaml")
    adapter = ConfigAdapter(config, variable="my_pipeline")
    pipeline_cfg = adapter.pipelines.my_pipeline
    """
    def __init__(self, raw_config: dict, variable: str):
        self._raw = raw_config
        self.variable = variable

        # convert to dot-access
        ns = to_namespace(raw_config)
        self.__dict__.update(ns.__dict__)

        # inject unified sections
        self.variables = ns.inputs
        self.features = ns.feature_engineering
        self.datasets = ns.inputs
        self.models = SimpleNamespace()  # placeholder

