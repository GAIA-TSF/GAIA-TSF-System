"""
Module Registry.

Maps operation names (from config.yaml) -> Python classes.

Used by DAGBuilder to dynamically instantiate nodes.
"""


class ModuleRegistry:
    def __init__(self):
        # stores mapping: op_name → class
        self._modules = {}

    def register(self, name: str, cls):
        """
        Register a module.

        Args:
            name (str): operation name used in config (e.g. 'masking')
            cls: class implementing the module
        """
        self._modules[name] = cls

    def get(self, name: str):
        """
        Retrieve a registered module.

        Args:
            name (str): operation name

        Returns:
            class
        """
        if name not in self._modules:
            raise ValueError(f"[Registry] Module '{name}' not registered")

        return self._modules[name]


# Global singleton registry
registry = ModuleRegistry() 
