"""Test asymptotic calculator"""

import numpy as np

from spey.hypothesis_testing.asymptotic_calculator import (
    compute_asymptotic_confidence_level,
)
from spey.hypothesis_testing.toy_calculator import compute_toy_confidence_level


def test_compute_asymptotic_confidence_level():
    """Test asymptotic CL computer"""

    a, b = compute_asymptotic_confidence_level(1.0, 2.0, test_stat="qtilde")

    assert np.isclose(a, 0.05933583307142766), "Observed CLs value is wrong."

    assert np.isclose(
        b,
        [
            0.05933583307142766,
            0.14339349869880672,
            0.31731050786291404,
            0.5942867086725301,
            0.8609310408460745,
        ],
    ).all(), "expected CLs value is wrong."

    a, b = compute_asymptotic_confidence_level(1.0, 2.0, "q0")

    assert np.isclose(a, 0.001349898031630116), "Observed CLs value is wrong."
    assert np.isclose(
        b,
        [
            0.001349898031630116,
            0.02275013194817923,
            0.15865525393145702,
            0.5,
            0.841344746068543,
        ],
    ).all(), "expected CLs value is wrong."


def test_compute_toy_confidence_level():
    """Test toy CLs computer"""
    a, b = compute_toy_confidence_level([2.0], [1.0], 1.5)

    assert a == [0.0], "Observed CLs value is wrong."
    assert b == [1.0] * 5, "expected CLs value is wrong."
