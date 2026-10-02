"""
Tests for the deprecated :class:`~spey.UnCorrStatisticsCombiner` wrapper.

The class is now a thin, mutable front-end to the ``default.uncorrelated_combiner``
plug-in.  These tests pin down its backwards-compatible behaviour: the container API,
the deprecation warning, per-analysis data dictionaries, the ``NaN`` convention for
negative expected yields, and the POI-only fit options.
"""

import logging

import numpy as np
import pytest

import spey
from spey.base.backend_base import BackendBase
from spey.base.model_config import ModelConfig
from spey.combiner import UncorrelatedStatisticsCombiner
from spey.interface.statistical_model import StatisticalModel
from spey.system.exceptions import AnalysisQueryError, NegativeExpectedYields

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


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


def _combiner(*models):
    return spey.UnCorrStatisticsCombiner(*(models or _models()))


class _NegativeYieldBackend(BackendBase):
    """Backend whose likelihood always reports negative expected yields."""

    name = "test.negative_yields"
    version = "0.0.0"
    author = "test"
    spey_requires = ">=0.0.0"

    def config(self, allow_negative_signal=True, poi_upper_bound=10.0):
        return ModelConfig(0, -1.0, [1.0], [(-1.0, poi_upper_bound)])

    def get_logpdf_func(self, expected=spey.ExpectationType.observed, data=None):
        def logpdf(pars):
            raise NegativeExpectedYields("negative yields")

        return logpdf


class TestDeprecation:
    def test_construction_warns(self):
        with pytest.warns(FutureWarning, match="default.uncorrelated_combiner"):
            spey.UnCorrStatisticsCombiner(*_models())

    def test_legacy_import_path(self):
        from spey.combiner.uncorrelated_statistics_combiner import (
            UnCorrStatisticsCombiner,
        )

        assert UnCorrStatisticsCombiner is spey.UnCorrStatisticsCombiner

    def test_plugins_are_not_exported_at_top_level(self):
        assert not hasattr(spey, "UncorrelatedStatisticsCombiner")
        assert not hasattr(spey, "CorrelatedStatisticsCombiner")

    def test_combined_model_is_the_plugin(self):
        combiner = _combiner()
        model = combiner.combined_model
        assert isinstance(model, StatisticalModel)
        assert isinstance(model.backend, UncorrelatedStatisticsCombiner)
        assert model.backend.analyses == ["A", "B", "C"]


class TestContainer:
    def test_append_len_analyses_items_and_getitem(self):
        first, second, third = _models()
        comb = _combiner(first)
        comb.append(second)
        comb.append(third)

        assert len(comb) == 3
        assert comb.analyses == ["A", "B", "C"]
        assert dict(comb.items())["B"] is second
        assert comb[0] is first
        assert comb["C"] is third
        assert isinstance(comb[:], tuple) and comb[0:2][1] is second
        with pytest.raises(AnalysisQueryError):
            comb[5]
        with pytest.raises(AnalysisQueryError):
            comb["unknown"]

    def test_append_rejects_duplicates_and_wrong_types(self):
        first, *_ = _models()
        comb = _combiner(first)
        with pytest.raises(AnalysisQueryError):
            comb.append(first)
        with pytest.raises(TypeError):
            comb.append("not a model")

    def test_remove_and_not_found(self):
        comb = _combiner()
        comb.remove("A")
        assert comb.analyses == ["B", "C"]
        with pytest.raises(AnalysisQueryError):
            comb.remove("nonexistent")

    def test_combined_model_follows_the_stack(self):
        first, second, third = _models()
        comb = _combiner(first, second)
        nll_two = comb.likelihood(1.0)
        comb.append(third)
        assert comb.combined_model.backend.analyses == ["A", "B", "C"]
        assert comb.likelihood(1.0) == pytest.approx(nll_two + third.likelihood(1.0))
        comb.remove("C")
        assert comb.likelihood(1.0) == pytest.approx(nll_two)

    def test_matmul(self):
        first, second, third = _models()
        left = _combiner(first)
        merged = left @ second
        assert merged.analyses == ["A", "B"] and left.analyses == ["A"]
        assert (merged @ _combiner(third)).analyses == ["A", "B", "C"]
        with pytest.raises(ValueError):
            left @ 3

    def test_capabilities(self):
        comb = _combiner()
        assert comb.is_alive
        assert comb.is_asymptotic_calculator_available
        assert comb.is_chi_square_calculator_available
        assert comb.is_toy_calculator_available is False
        assert comb.minimum_poi == max(m.backend.config().minimum_poi for m in comb)


class TestLikelihood:
    @pytest.mark.parametrize("mu", [0.0, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize(
        "expected", [spey.ExpectationType.observed, spey.ExpectationType.apriori]
    )
    def test_sum_of_profile_likelihoods(self, mu, expected):
        """Shared mu, independent nuisances: the profile NLLs simply add up."""
        models = _models()
        comb = _combiner(*models)
        manual = sum(m.likelihood(mu, expected=expected) for m in models)
        assert comb.likelihood(mu, expected=expected) == pytest.approx(manual, rel=1e-6)
        assert comb.likelihood(mu, expected=expected, return_nll=False) == pytest.approx(
            np.exp(-manual), rel=1e-5
        )

    def test_per_analysis_data(self):
        models = _models()
        comb = _combiner(*models)
        manual = (
            models[0].likelihood(1.0, data=[60, 52])
            + models[1].likelihood(1.0)
            + models[2].likelihood(1.0)
        )
        assert comb.likelihood(1.0, data={"A": [60, 52]}) == pytest.approx(manual)
        assert comb.likelihood(1.0, data={}) == pytest.approx(comb.likelihood(1.0))

    def test_negative_expected_yields_give_nan(self):
        bad = StatisticalModel(backend=_NegativeYieldBackend(), analysis="bad")
        comb = _combiner(_models()[0], bad)
        with pytest.warns(RuntimeWarning, match="nan"):
            assert np.isnan(comb.likelihood(poi_test=1.0))

    def test_statistical_model_options_are_ignored(self, caplog):
        comb = _combiner()
        options = {"default.uncorrelated_background": {"init_pars": [1.0, 0.0, 0.0]}}
        # the Spey logger does not propagate by default, see spey.system.logger
        spey_logger = logging.getLogger("Spey")
        previous, spey_logger.propagate = spey_logger.propagate, True
        try:
            with caplog.at_level(logging.WARNING, logger="Spey"):
                value = comb.likelihood(1.0, statistical_model_options=options)
        finally:
            spey_logger.propagate = previous
        assert "statistical_model_options" in caplog.text
        assert value == pytest.approx(comb.likelihood(1.0))


class TestAsimov:
    def test_generate_asimov_data_per_analysis(self):
        """Identical to per-model generation up to the optimiser tolerance of the fit."""
        models = _models()
        data = _combiner(*models).generate_asimov_data(test_statistic="qtilde")
        assert list(data) == ["A", "B", "C"]
        for model in models:
            assert isinstance(data[model.analysis], list)
            assert np.allclose(
                data[model.analysis],
                model.generate_asimov_data(test_statistic="qtilde"),
                rtol=1e-3,
                atol=1e-3,
            )

    def test_asimov_likelihood_matches_sum(self):
        models = _models()
        manual = sum(m.asimov_likelihood(1.0) for m in models)
        assert _combiner(*models).asimov_likelihood(1.0) == pytest.approx(
            manual, rel=1e-5
        )


class TestMaximisation:
    def test_maximize_likelihood(self):
        comb = _combiner()
        muhat, nll = comb.maximize_likelihood()
        assert isinstance(muhat, float)
        assert nll == pytest.approx(comb.likelihood(muhat), abs=1e-6)
        assert nll <= comb.likelihood(muhat + 0.05) and nll <= comb.likelihood(
            muhat - 0.05
        )

    def test_poi_only_options(self):
        """``initial_muhat_value`` and ``[(lo, hi)]`` refer to mu only."""
        comb = _combiner()
        muhat, _ = comb.maximize_likelihood(initial_muhat_value=0.3)
        assert muhat == pytest.approx(comb.maximize_likelihood()[0], abs=1e-3)
        bounded, _ = comb.maximize_likelihood(par_bounds=[(0.5, 2.0)])
        assert bounded == pytest.approx(0.5, abs=1e-6)

    def test_maximize_asimov_likelihood(self):
        muhat, nll = _combiner().maximize_asimov_likelihood(test_statistics="qtilde")
        assert muhat == pytest.approx(0.0, abs=1e-3)
        assert np.isfinite(nll)

    def test_chi2_runs(self):
        """Regression: chi2 used to forward ``init_pars`` to the optimiser twice."""
        assert np.isfinite(_combiner().chi2(poi_test=1.0))


class TestHypothesisTesting:
    def test_matches_plugin(self):
        models = _models()
        comb = _combiner(*models)
        plugin = spey.get_backend("default.uncorrelated_combiner")(
            statistical_models=list(models), analysis="ABC"
        )
        for expected in spey.ExpectationType:
            assert comb.exclusion_confidence_level(expected=expected) == pytest.approx(
                plugin.exclusion_confidence_level(expected=expected), rel=1e-4
            )
        assert comb.poi_upper_limit() == pytest.approx(plugin.poi_upper_limit(), rel=1e-4)
