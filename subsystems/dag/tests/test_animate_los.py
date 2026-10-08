"""Focused tests for full-period LOS animation helpers."""

import numpy as np
import pytest

from subsystems.dag.scripts.animate_los import symmetric_color_limit


def test_symmetric_color_limit_uses_absolute_values() -> None:
    values = np.asarray([[[-10.0, -2.0], [1.0, 4.0]]])

    assert symmetric_color_limit(values, 100.0) == pytest.approx(10.0)


def test_symmetric_color_limit_rejects_empty_stack() -> None:
    with pytest.raises(ValueError, match='no finite values'):
        symmetric_color_limit(np.full((1, 2, 2), np.nan))
