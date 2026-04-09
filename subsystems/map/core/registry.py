
"""
Registry module

Purpose:
- Central place where plugins (variables, models, features) are registered
- Enables dynamic lookup based on config (no hardcoding)

Design:
- Variables → instantiated immediately
- Models → stored as class (require config at runtime)
- Features → simple functions
"""

# Global registries (simple and explicit)
VARIABLE_REGISTRY = {}
MODEL_REGISTRY = {}
FEATURE_REGISTRY = {}


def register_variable(name):
    """
    Decorator to register a variable plugin.

    Usage:
        @register_variable("slope")
        class SlopeVariable(...)

    Result:
        VARIABLE_REGISTRY["slope"] → instance of SlopeVariable
    """
    def decorator(cls):
        VARIABLE_REGISTRY[name] = cls()
        return cls
    return decorator


def register_model(name):
    """
    Registers model class (not instance!).

    Why:
    - Models require config → instantiated later
    """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def register_feature(name):
    """
    Registers feature engineering function.

    Feature pipelines are:
    - stateless
    - reusable across variables
    """
    def decorator(fn):
        FEATURE_REGISTRY[name] = fn
        return fn
    return decorator 
