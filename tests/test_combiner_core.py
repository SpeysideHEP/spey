"""
Tests for :class:`~spey.combiner.combiner_core.CombinerBase`, the extension point
shared by the combiner plug-ins.
"""

import numpy as np
import pytest

import spey
from spey.combiner import (
    CombinerBase,
    CorrelatedStatisticsCombiner,
    UncorrelatedStatisticsCombiner,
)
from spey.system.exceptions import InvalidInput


def _models():
    poisson = spey.get_backend("default.poisson")
    uncorrelated = spey.get_backend("default.uncorrelated_background")
    return [
        poisson(signal_yields=[3.0], background_yields=[20.0], data=[22], analysis="P"),
        uncorrelated(
            signal_yields=[8.0],
            background_yields=[30.0],
            data=[33],
            absolute_uncertainties=[6.0],
            analysis="U",
        ),
    ]


def test_plugins_share_the_base():
    assert issubclass(UncorrelatedStatisticsCombiner, CombinerBase)
    assert issubclass(CorrelatedStatisticsCombiner, CombinerBase)


def test_base_is_abstract():
    with pytest.raises(TypeError):
        CombinerBase(_models())  # pylint: disable=abstract-class-instantiated


def test_minimal_subclass():
    """The module-level example: only `_build_parameter_map` has to be written."""

    class ShareEverythingNamedMu(CombinerBase):
        name = "test.mu_combiner"

        def _build_parameter_map(self, local_names):
            groups = {
                (pos, names.index("mu")): "mu" for pos, names in enumerate(local_names)
            }
            maps, slot_of = self._allocate_slots(local_names, groups)
            return maps, {slot_of["mu"]: "mu"}

    models = _models()
    custom = ShareEverythingNamedMu(models)
    reference = UncorrelatedStatisticsCombiner(models)
    assert custom.parameter_names == reference.parameter_names
    pars = np.array([0.7, 0.2])
    assert custom.get_logpdf_func()(pars) == reference.get_logpdf_func()(pars)
    assert repr(custom).startswith("ShareEverythingNamedMu(")


@pytest.mark.parametrize(
    "maps",
    [
        [np.array([0]), np.array([0, 2])],  # slot 1 is never used
        [np.array([0]), np.array([0])],  # second model has two parameters
    ],
)
def test_invalid_index_maps_are_rejected(maps):
    class Broken(CombinerBase):
        name = "test.broken"

        def _build_parameter_map(self, local_names):
            return maps, {}

    with pytest.raises(InvalidInput):
        Broken(_models())
