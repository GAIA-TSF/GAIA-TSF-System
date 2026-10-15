"""Built-in MAP plugins.

Importing this package registers the maintained variable and model plugins.
"""

from subsystems.map.plugins import models, variables

__all__ = ['models', 'variables']
