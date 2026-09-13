"""Boundary checks for the public duplicate-value helper."""

import subprocess
import sys

import numpy as np
import pytest

from petra.utils import process_array


def test_duplicate_processing_rejects_nan_without_hanging():
    # The old duplicate loop never advanced once its current value was NaN.
    code = """
import numpy as np
from petra.utils import process_array
try:
    process_array(np.array([1.0, 1.0, np.nan]))
except ValueError as error:
    assert 'finite' in str(error)
else:
    raise AssertionError('nonfinite input was accepted')
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=2,
                   capture_output=True, text=True)


@pytest.mark.parametrize("values", [
    np.array([np.inf]), np.array([-np.inf]), np.array([1 + 2j]),
    np.ones((2, 2)), np.array(1.0),
])
def test_duplicate_processing_rejects_invalid_input(values):
    with pytest.raises(ValueError, match="finite|real|one-dimensional"):
        process_array(values)


@pytest.mark.parametrize("values", [
    np.array([1e10, 1e10, 1e10]),
    np.array([1.0, 1.0, np.nextafter(1.0, np.inf)]),
])
def test_duplicate_processing_makes_adjacent_floats_strictly_increasing(values):
    original = values.copy()
    processed = process_array(values)
    assert np.all(np.isfinite(processed))
    assert np.all(np.diff(processed) > 0)
    np.testing.assert_array_equal(values, original)


def test_duplicate_processing_rejects_overflow_instead_of_returning_equal_values():
    with pytest.raises(ValueError, match="finite|representable"):
        process_array(np.full(2, np.finfo(float).max))
