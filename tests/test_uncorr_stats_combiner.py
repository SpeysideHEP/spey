"""
Tests for the deprecated :func:`~spey.UnCorrStatisticsCombiner` wrapper.

The function only warns and returns the statistical model built by the
``default.uncorrelated_combiner`` plug-in; the combination itself is tested in
``test_uncorrelated_combiner.py``.
"""

import numpy as np
import pytest

import spey
from spey.combiner import UncorrelatedStatisticsCombiner
from spey.interface.statistical_model import StatisticalModel
from spey.system.exceptions import AnalysisQueryError, InvalidInput


def _models():
    """Two independent models with nuisance parameters and one without."""
    uncorrelated = spey.get_backend("default.uncorrelated_background")
    poisson = spey.get_backend("default.poisson")
    return (
        uncorrelated(
            signal_yields=[12.0, 15.0],
            background_yields=[50.0, 48.0],
            data=[51, 48],
            absolute_uncertainties=[10.0, 8.0],
            analysis="A",
        ),
        uncorrelated(
            signal_yields=[8.0],
            background_yields=[30.0],
            data=[33],
            absolute_uncertainties=[6.0],
            analysis="B",
        ),
        poisson(signal_yields=[3.0], background_yields=[20.0], data=[22], analysis="C"),
    )


def _combine(*models, **kwargs):
    with pytest.warns(FutureWarning, match="default.uncorrelated_combiner"):
        return spey.UnCorrStatisticsCombiner(*(models or _models()), **kwargs)


def test_construction_warns_and_returns_the_plugin_model():
    combined = _combine()
    assert isinstance(combined, StatisticalModel)
    assert isinstance(combined.backend, UncorrelatedStatisticsCombiner)
    assert combined.analysis == "A+B+C"
    assert combined.backend.analyses == ["A", "B", "C"]


def test_is_the_plugin_called_with_the_same_models():
    models = _models()
    plugin = spey.get_backend("default.uncorrelated_combiner")(
        statistical_models=list(models), analysis="A+B+C"
    )
    combined = _combine(*models)
    for expected in spey.ExpectationType:
        assert np.allclose(
            combined.exclusion_confidence_level(expected=expected),
            plugin.exclusion_confidence_level(expected=expected),
        )
    assert combined.likelihood(0.7) == pytest.approx(plugin.likelihood(0.7))


def test_ntoys_is_passed_on():
    assert _combine(ntoys=123).ntoys == 123


def test_combinations_can_be_merged_with_matmul():
    first, second, third = _models()
    merged = _combine(first) @ _combine(second, third)
    assert merged.backend.analyses == ["A", "B", "C"]


def test_invalid_input_is_rejected_by_the_plugin():
    first = _models()[0]
    with pytest.raises(AnalysisQueryError):
        _combine(first, first)
    with pytest.raises(TypeError):
        _combine(first, 3)
    with pytest.raises(InvalidInput):
        with pytest.warns(FutureWarning):
            spey.UnCorrStatisticsCombiner()


def test_legacy_import_path():
    from spey.combiner.uncorrelated_statistics_combiner import UnCorrStatisticsCombiner

    assert UnCorrStatisticsCombiner is spey.UnCorrStatisticsCombiner


def test_plugins_are_not_exported_at_top_level():
    assert not hasattr(spey, "UncorrelatedStatisticsCombiner")
    assert not hasattr(spey, "CorrelatedStatisticsCombiner")
