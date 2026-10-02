"""
Unit tests for the ``default.uncorrelated_combiner`` plug-in,
:class:`~spey.combiner.UncorrelatedStatisticsCombiner`.

The key property is exactness of the factorisation: with a shared signal strength and
independent nuisance parameters, the profile negative log-likelihood of the
combination equals the sum of the profile NLLs of the constituents, and the
combination is identical to a correlated combination that shares only ``mu``.
"""

import numpy as np
import pytest

import spey
from spey.base.backend_base import BackendBase
from spey.base.model_config import ModelConfig
from spey.combiner import CorrelatedStatisticsCombiner, UncorrelatedStatisticsCombiner
from spey.interface.statistical_model import StatisticalModel, statistical_model_wrapper
from spey.system.exceptions import AnalysisQueryError, InvalidInput
from spey.utils import ExpectationType


def build_models():
    """Three independent models: 2 + 1 nuisance parameters, and none."""
    uncorrelated = spey.get_backend("default.uncorrelated_background")
    poisson = spey.get_backend("default.poisson")
    return [
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
    ]


def build_combined(models=None):
    return statistical_model_wrapper(UncorrelatedStatisticsCombiner)(
        statistical_models=models or build_models(), analysis="combined"
    )


class TestPluginRegistration:
    def test_metadata(self):
        assert UncorrelatedStatisticsCombiner.name == "default.uncorrelated_combiner"
        assert issubclass(UncorrelatedStatisticsCombiner, BackendBase)
        assert not getattr(UncorrelatedStatisticsCombiner, "__abstractmethods__", False)

    def test_entry_point_is_declared(self):
        if "default.uncorrelated_combiner" not in spey.AvailableBackends():
            pytest.skip(
                "entry points are stale in this environment; reinstall with `pip install -e .`"
            )
        model = spey.get_backend("default.uncorrelated_combiner")(
            statistical_models=build_models(), analysis="combined"
        )
        assert isinstance(model, StatisticalModel)
        assert isinstance(model.backend, UncorrelatedStatisticsCombiner)


class TestParameterMapping:
    def test_poi_is_the_only_shared_slot(self):
        backend = UncorrelatedStatisticsCombiner(build_models())
        assert backend.npar == 1 + 2 + 1 + 0
        assert backend.config().poi_index == 0
        assert backend.parameter_names == ["mu", "A_par_1", "A_par_2", "B_par_1"]
        assert backend.shared_parameter_names == ["mu"]
        assert np.array_equal(backend.index_map["A"], [0, 1, 2])
        assert np.array_equal(backend.index_map["B"], [0, 3])
        assert np.array_equal(backend.index_map["C"], [0])

    def test_poi_named_after_first_model(self):
        normal = spey.get_backend("default.normal")
        models = [
            normal(
                signal_yields=lambda c: c[0] * np.array([2.0]),
                background_yields=[10.0],
                data=[11.0],
                absolute_uncertainties=[3.0],
                n_signal_parameters=1,
                analysis=name,
            )
            for name in ("X", "Y")
        ]
        backend = UncorrelatedStatisticsCombiner(models)
        # the signal parameters keep private, analysis-qualified slots
        assert backend.parameter_names == ["mu", "X::signal_par_0", "Y::signal_par_0"]

    def test_minimum_poi_and_bounds(self):
        backend = UncorrelatedStatisticsCombiner(build_models())
        minima = [m.backend.config().minimum_poi for m in build_models()]
        assert backend.config().minimum_poi == pytest.approx(max(minima))
        assert backend.config(allow_negative_signal=False).suggested_bounds[0][0] == 0.0

    def test_identical_to_correlated_with_only_mu_shared(self):
        models = build_models()
        uncorrelated = UncorrelatedStatisticsCombiner(models)
        correlated = CorrelatedStatisticsCombiner(models, shared_parameters=["mu"])
        assert uncorrelated.parameter_names == correlated.parameter_names
        pars = np.array([0.8, 0.3, -0.2, 0.5])
        assert uncorrelated.get_logpdf_func()(pars) == correlated.get_logpdf_func()(pars)
        assert np.array_equal(
            uncorrelated.get_hessian_logpdf_func()(pars),
            correlated.get_hessian_logpdf_func()(pars),
        )


class TestValidation:
    def test_empty_input(self):
        with pytest.raises(InvalidInput):
            UncorrelatedStatisticsCombiner([])

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            UncorrelatedStatisticsCombiner([build_models()[0], "model"])

    def test_duplicate_analysis_names(self):
        first = build_models()[0]
        with pytest.raises(AnalysisQueryError):
            UncorrelatedStatisticsCombiner([first, first])

    def test_single_model(self):
        model = build_models()[0]
        combined = build_combined([model])
        assert combined.likelihood(1.0) == pytest.approx(model.likelihood(1.0), rel=1e-6)

    def test_model_without_poi(self):
        class NoPoi(BackendBase):
            name, version, author, spey_requires = "test.no_poi", "0", "t", ">=0.0.0"

            def config(self, allow_negative_signal=True, poi_upper_bound=10.0):
                return ModelConfig(None, 0.0, [0.0], [(-1.0, 1.0)])

            def get_logpdf_func(self, expected=ExpectationType.observed, data=None):
                return lambda pars: -float(pars[0] ** 2)

        with pytest.raises(InvalidInput, match="parameter of interest"):
            UncorrelatedStatisticsCombiner(
                [build_models()[0], StatisticalModel(NoPoi(), analysis="nopoi")]
            )


class TestLikelihood:
    def test_logpdf_equals_manual_sum(self):
        models = build_models()
        backend = UncorrelatedStatisticsCombiner(models)
        pars = np.array([1.1, 0.3, -0.2, 0.4])
        manual = sum(
            float(m.backend.get_logpdf_func()(pars[backend.index_map[m.analysis]]))
            for m in models
        )
        assert backend.get_logpdf_func()(pars) == pytest.approx(manual, rel=1e-12)

    @pytest.mark.parametrize("mu", [0.0, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize(
        "expected", [ExpectationType.observed, ExpectationType.apriori]
    )
    def test_profile_nll_is_sum_of_profile_nlls(self, mu, expected):
        models = build_models()
        manual = sum(m.likelihood(mu, expected=expected) for m in models)
        assert build_combined(models).likelihood(mu, expected=expected) == pytest.approx(
            manual, rel=1e-6
        )

    def test_dict_data_matches_flat_data(self):
        combined = build_combined()
        flat = [60.0, 52.0, 0.0, 0.0, 33.0, 0.0, 22.0]
        assert combined.likelihood(1.0, data={"A": [60, 52, 0, 0]}) == pytest.approx(
            combined.likelihood(1.0, data=flat)
        )
        # keys may also be model positions
        assert combined.likelihood(1.0, data={0: [60, 52, 0, 0]}) == pytest.approx(
            combined.likelihood(1.0, data=flat)
        )

    def test_dict_data_with_unknown_analysis(self):
        with pytest.raises(AnalysisQueryError):
            build_combined().likelihood(1.0, data={"unknown": [1.0]})

    def test_split_data_round_trip(self):
        backend = UncorrelatedStatisticsCombiner(build_models())
        expected = backend.expected_data([1.0, 0.1, -0.1, 0.2])
        pieces = backend.split_data(expected)
        assert list(pieces) == ["A", "B", "C"]
        assert np.array_equal(np.concatenate(list(pieces.values())), expected)
        with pytest.raises(InvalidInput):
            backend.split_data(expected[:-1])


class TestCombine:
    def test_matmul_merges_combinations(self):
        first, second, third = build_models()
        left = build_combined([first, second])
        right = build_combined([third])
        merged = left @ right
        assert isinstance(merged.backend, UncorrelatedStatisticsCombiner)
        assert merged.backend.analyses == ["A", "B", "C"]
        assert merged.likelihood(1.0) == pytest.approx(
            build_combined().likelihood(1.0), rel=1e-8
        )

    def test_combine_with_other_backend(self):
        first, second, third = build_models()
        with pytest.raises(TypeError, match="single model"):
            build_combined([first, second]) @ third


class TestStatisticalModelIntegration:
    def test_every_calculator_is_available(self):
        assert sorted(build_combined().available_calculators) == [
            "asymptotic",
            "chi_square",
            "toy",
        ]

    def test_hypothesis_testing(self):
        combined = build_combined()
        assert 0.0 <= combined.exclusion_confidence_level()[0] <= 1.0
        assert 0.0 < combined.poi_upper_limit() < 10.0
        assert np.isfinite(combined.sigma_mu_from_hessian(poi_test=1.0))

    def test_reproduces_multivariate_normal(self):
        """Three independent unit Gaussians equal one diagonal multivariate normal."""
        normal = spey.get_backend("default.normal")
        models = [
            normal(
                signal_yields=[1],
                background_yields=[bkg],
                data=[2],
                absolute_uncertainties=[1],
                analysis=f"norm{idx}",
            )
            for idx, bkg in enumerate([3, 1, 2])
        ]
        multivariate = spey.get_backend("default.multivariate_normal")(
            signal_yields=[1, 1, 1],
            background_yields=[3, 1, 2],
            data=[2, 2, 2],
            covariance_matrix=np.diag([1, 1, 1]),
        )
        assert build_combined(models).exclusion_confidence_level()[0] == pytest.approx(
            multivariate.exclusion_confidence_level()[0]
        )
