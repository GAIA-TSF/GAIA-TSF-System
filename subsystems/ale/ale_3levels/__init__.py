"""GAIA ALE prototype: alert engines over MAP forecast residuals.

The package reads saved MAP outputs only (see :mod:`ale_3levels.contract`) and
never imports MAP. Engines emit advisory levels and text; they never execute
any action.
"""

__version__ = '0.1.0'
