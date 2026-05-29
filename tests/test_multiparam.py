"""
Unit tests for multi-parameter-of-interest (multi-POI) support added in the
multiparam-fit branch.  The tests cover:

  * ModelConfig.resolve_poi_indices
  * ModelConfig.fixed_poi_bounds_multi
  * optimizer.fit with dict fixed_poi_value
  * StatisticalModel._resolve_poi_test
  * StatisticalModel.likelihood with dict poi_test
  * StatisticalModel.fixed_poi_sampler with dict poi_test
  * StatisticalModel.sigma_mu_from_hessian with dict poi_test
  * StatisticalModel.maximize_likelihood with poi_indices
  * StatisticalModel.maximize_asimov_likelihood with poi_indices
  * UnCorrStatisticsCombiner.maximize_likelihood with poi_indices
  * UnCorrStatisticsCombiner.maximize_asimov_likelihood with poi_indices
"""

import types
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# ModelConfig helpers
# ---------------------------------------------------------------------------
from spey.base.model_config import ModelConfig


def _make_config(n=3, parameter_names=None):
    """Return a ModelConfig with *n* parameters."""
    return ModelConfig(
        poi_index=0,
        minimum_poi=-5.0,
        suggested_init=[0.0] * n,
        suggested_bounds=[(-10.0, 10.0)] * n,
        parameter_names=parameter_names,
    )


class TestResolvePoiIndices:
    def test_float_returns_primary_poi(self):
        cfg = _make_config()
        result = cfg.resolve_poi_indices(2.5)
        assert result == {0: 2.5}

    def test_int_key_dict(self):
        cfg = _make_config()
        result = cfg.resolve_poi_indices({1: 3.0, 2: -1.5})
        assert result == {1: 3.0, 2: -1.5}

    def test_str_key_dict_resolves_to_indices(self):
        cfg = _make_config(parameter_names=["mu", "theta1", "theta2"])
        result = cfg.resolve_poi_indices({"mu": 1.0, "theta2": 0.5})
        assert result == {0: 1.0, 2: 0.5}

    def test_str_key_raises_when_no_parameter_names(self):
        cfg = _make_config()  # parameter_names=None
        with pytest.raises(ValueError, match="parameter_names not set"):
            cfg.resolve_poi_indices({"mu": 1.0})

    def test_str_key_raises_when_name_not_found(self):
        cfg = _make_config(parameter_names=["mu", "theta"])
        with pytest.raises(ValueError, match="not found in parameter_names"):
            cfg.resolve_poi_indices({"nonexistent": 1.0})

    def test_mixed_int_and_str_keys(self):
        cfg = _make_config(parameter_names=["mu", "theta1", "theta2"])
        result = cfg.resolve_poi_indices({"mu": 1.0, 2: 0.7})
        assert result == {0: 1.0, 2: 0.7}

    def test_float_conversion(self):
        cfg = _make_config()
        result = cfg.resolve_poi_indices(3)  # int, not float
        assert result == {0: 3.0}
        assert isinstance(result[0], float)


class TestFixedPoiBoundsMulti:
    def test_none_returns_suggested_bounds(self):
        cfg = _make_config()
        assert cfg.fixed_poi_bounds_multi(None) == cfg.suggested_bounds

    def test_empty_dict_returns_suggested_bounds(self):
        cfg = _make_config()
        assert cfg.fixed_poi_bounds_multi({}) == cfg.suggested_bounds

    def test_value_inside_bounds_unchanged(self):
        cfg = _make_config()
        bounds = cfg.fixed_poi_bounds_multi({0: 0.5})
        # 0.5 is inside (-10, 10), so bounds[0] should be unchanged
        assert bounds[0] == (-10.0, 10.0)

    def test_value_outside_bounds_widened(self):
        cfg = _make_config()
        bounds = cfg.fixed_poi_bounds_multi({0: 15.0})
        # 15.0 > 10.0, so bounds should be widened
        assert bounds[0][1] >= 15.0

    def test_negative_value_outside_bounds_widened(self):
        cfg = _make_config()
        bounds = cfg.fixed_poi_bounds_multi({0: -20.0})
        # -20.0 < -10.0, should be widened; lower bound should be minimum_poi
        assert bounds[0][0] == cfg.minimum_poi

    def test_multiple_indices_updated_independently(self):
        cfg = _make_config()
        bounds = cfg.fixed_poi_bounds_multi({0: 50.0, 1: 0.5})
        # index 0 should be widened, index 1 unchanged
        assert bounds[0][1] >= 50.0
        assert bounds[1] == (-10.0, 10.0)

    def test_does_not_mutate_suggested_bounds(self):
        cfg = _make_config()
        original = list(cfg.suggested_bounds)
        cfg.fixed_poi_bounds_multi({0: 50.0})
        assert cfg.suggested_bounds == original


# ---------------------------------------------------------------------------
# optimizer.fit with dict fixed_poi_value
# ---------------------------------------------------------------------------
from spey.optimizer import core as core_mod


class _FakeModelConfig:
    def __init__(self, n=3):
        self.suggested_init = [0.0] * n
        self.poi_index = 0
        self.suggested_fixed = None

    def fixed_poi_bounds(self, val):
        return [(-10.0, 10.0)] * len(self.suggested_init)

    def fixed_poi_bounds_multi(self, d):
        return [(-10.0, 10.0)] * len(self.suggested_init)


class TestFitWithDictFixedPoi:
    def _run_fit(self, monkeypatch, fixed_poi_value, n=3):
        captured = {}

        def fake_minimizer(
            func, init_pars, fixed_vals, do_grad, hessian, bounds, constraints, **kw
        ):
            captured["init_pars"] = list(init_pars)
            captured["fixed_vals"] = list(fixed_vals)
            return 0.0, np.array(init_pars)

        monkeypatch.setattr(core_mod, "_get_minimizer", lambda name: fake_minimizer)
        cfg = _FakeModelConfig(n=n)
        core_mod.fit(
            func=lambda p: 0.0,
            model_configuration=cfg,
            fixed_poi_value=fixed_poi_value,
        )
        return captured

    def test_float_fixes_poi_index(self, monkeypatch):
        captured = self._run_fit(monkeypatch, fixed_poi_value=3.14)
        assert captured["init_pars"][0] == pytest.approx(3.14)
        assert captured["fixed_vals"][0] is True
        assert captured["fixed_vals"][1] is False
        assert captured["fixed_vals"][2] is False

    def test_dict_fixes_multiple_indices(self, monkeypatch):
        captured = self._run_fit(monkeypatch, fixed_poi_value={0: 1.0, 2: -2.5})
        assert captured["init_pars"][0] == pytest.approx(1.0)
        assert captured["init_pars"][2] == pytest.approx(-2.5)
        assert captured["fixed_vals"][0] is True
        assert captured["fixed_vals"][1] is False
        assert captured["fixed_vals"][2] is True

    def test_dict_with_single_entry(self, monkeypatch):
        captured = self._run_fit(monkeypatch, fixed_poi_value={1: 7.0})
        assert captured["init_pars"][1] == pytest.approx(7.0)
        assert captured["fixed_vals"][0] is False
        assert captured["fixed_vals"][1] is True
        assert captured["fixed_vals"][2] is False


# ---------------------------------------------------------------------------
# StatisticalModel helpers
# ---------------------------------------------------------------------------
import spey.interface.statistical_model as sm_mod

StatisticalModel = sm_mod.StatisticalModel
PoiTest = sm_mod.PoiTest


def _default_model_config():
    return ModelConfig(
        poi_index=0,
        minimum_poi=-5.0,
        suggested_init=[0.0, 0.0],
        suggested_bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        parameter_names=["mu", "theta"],
    )


class _FakeBackendBase:
    """Minimal BackendBase stand-in."""

    constraints = []

    def get_objective_function(self, expected=None, data=None, do_grad=True):
        if do_grad:
            raise NotImplementedError
        return lambda p: 0.0

    def get_logpdf_func(self, expected=None, data=None):
        return lambda p: -float(np.sum(np.asarray(p) ** 2))

    def expected_data(self, pars):
        return [0.0]

    def get_sampler(self, fit_param):
        raise NotImplementedError

    def get_hessian_logpdf_func(self, expected=None):
        raise NotImplementedError

    def config(self, allow_negative_signal=True):
        return _default_model_config()

    def negative_loglikelihood(self, *args, **kwargs):
        raise NotImplementedError

    def asimov_negative_loglikelihood(self, *args, **kwargs):
        raise NotImplementedError

    def minimize_negative_loglikelihood(self, *args, **kwargs):
        raise NotImplementedError

    def minimize_asimov_negative_loglikelihood(self, *args, **kwargs):
        raise NotImplementedError


def _make_stat_model(monkeypatch, config_override=None):
    monkeypatch.setattr(sm_mod, "BackendBase", _FakeBackendBase)

    class Backend(_FakeBackendBase):
        name = "fake"

        def config(self, allow_negative_signal=True):
            if config_override is not None:
                return config_override
            return _default_model_config()

    return StatisticalModel(backend=Backend(), analysis="test")


class TestResolvePoiTest:
    def test_float_returned_unchanged(self, monkeypatch):
        sm = _make_stat_model(monkeypatch)
        assert sm._resolve_poi_test(3.14) == 3.14

    def test_dict_with_int_keys_returned_unchanged(self, monkeypatch):
        sm = _make_stat_model(monkeypatch)
        result = sm._resolve_poi_test({0: 1.0, 1: 0.5})
        assert result == {0: 1.0, 1: 0.5}

    def test_dict_with_str_keys_resolved_to_indices(self, monkeypatch):
        # backend config has parameter_names=["mu", "theta"]
        cfg = ModelConfig(
            poi_index=0,
            minimum_poi=-5.0,
            suggested_init=[0.0, 0.0],
            suggested_bounds=[(-10.0, 10.0), (-10.0, 10.0)],
            parameter_names=["mu", "theta"],
        )

        class BackendWithConfig(_FakeBackendBase):
            name = "fake"

            def config(self, allow_negative_signal=True):
                return cfg

        monkeypatch.setattr(sm_mod, "BackendBase", _FakeBackendBase)
        sm = StatisticalModel(backend=BackendWithConfig(), analysis="resolve_test")
        result = sm._resolve_poi_test({"mu": 1.0, "theta": 0.5})
        assert result == {0: 1.0, 1: 0.5}


class TestPoiTestTypeAlias:
    def test_poi_test_in_module_all(self):
        assert "PoiTest" in sm_mod.__all__

    def test_poi_test_is_union_type(self):
        # PoiTest = Union[float, Dict[...]] — just check it is accessible
        import typing

        assert PoiTest is not None


# ---------------------------------------------------------------------------
# StatisticalModel.likelihood with dict poi_test
# ---------------------------------------------------------------------------
class TestLikelihoodWithDictPoiTest:
    def test_dict_poi_test_calls_fit_not_direct_logpdf(self, monkeypatch):
        """When poi_test is a dict, the npar==1 shortcut must NOT fire."""
        fit_called = {"called": False}

        def fake_fit(**kwargs):
            fit_called["called"] = True
            fit_called["fixed_poi_value"] = kwargs.get("fixed_poi_value")
            return -5.0, np.array([1.0, 0.5])

        monkeypatch.setattr(sm_mod, "BackendBase", _FakeBackendBase)
        monkeypatch.setattr(sm_mod, "fit", fake_fit)

        sm = _make_stat_model(monkeypatch)
        result = sm.likelihood(poi_test={0: 1.0}, return_nll=True)

        assert fit_called["called"], "fit() should have been called for dict poi_test"
        # fixed_poi_value passed to fit must be the dict (int-keyed)
        assert isinstance(fit_called["fixed_poi_value"], dict)
        assert result == pytest.approx(5.0)

    def test_float_poi_test_uses_direct_logpdf_for_single_par(self, monkeypatch):
        """When npar==1 and poi_test is float, direct logpdf is used."""

        class SingleParBackend(_FakeBackendBase):
            name = "single"

            def config(self, allow_negative_signal=True):
                return types.SimpleNamespace(npar=1, poi_index=0, parameter_names=["mu"])

        monkeypatch.setattr(sm_mod, "BackendBase", _FakeBackendBase)
        sm = StatisticalModel(backend=SingleParBackend(), analysis="single")

        # monkeypatch fit so any accidental call raises
        monkeypatch.setattr(
            sm_mod,
            "fit",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("fit must not be called")),
        )

        # logpdf([2.0]) = -sum([2.0]^2) = -4.0 -> nll = 4.0
        nll = sm.likelihood(poi_test=2.0, return_nll=True)
        assert nll == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# StatisticalModel.maximize_likelihood with poi_indices
# ---------------------------------------------------------------------------
class TestMaximizeLikelihoodPoiIndices:
    def _sm_with_fit(self, monkeypatch, fit_params):
        """Return a StatisticalModel whose fit() returns *fit_params*."""

        def fake_fit(**kwargs):
            return -10.0, np.array(fit_params)

        monkeypatch.setattr(sm_mod, "BackendBase", _FakeBackendBase)
        monkeypatch.setattr(sm_mod, "fit", fake_fit)
        return _make_stat_model(monkeypatch)

    def test_none_returns_primary_poi_as_float(self, monkeypatch):
        sm = self._sm_with_fit(monkeypatch, [3.0, 0.7])
        muhat, nll = sm.maximize_likelihood(poi_indices=None)
        assert isinstance(muhat, float)
        assert muhat == pytest.approx(3.0)
        assert nll == pytest.approx(10.0)

    def test_int_key_list_returns_dict(self, monkeypatch):
        sm = self._sm_with_fit(monkeypatch, [3.0, 0.7])
        result, nll = sm.maximize_likelihood(poi_indices=[0, 1])
        assert isinstance(result, dict)
        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(0.7)
        assert nll == pytest.approx(10.0)

    def test_str_key_list_returns_dict_with_str_keys(self, monkeypatch):
        sm = self._sm_with_fit(monkeypatch, [3.0, 0.7])
        # backend config has parameter_names=["mu", "theta"]
        result, nll = sm.maximize_likelihood(poi_indices=["mu", "theta"])
        assert isinstance(result, dict)
        assert result["mu"] == pytest.approx(3.0)
        assert result["theta"] == pytest.approx(0.7)

    def test_str_key_raises_when_no_parameter_names(self, monkeypatch):
        def fake_fit(**kwargs):
            return -1.0, np.array([0.0, 0.0])

        class NoNamesBackend(_FakeBackendBase):
            name = "nonames"

            def config(self, allow_negative_signal=True):
                return types.SimpleNamespace(npar=2, poi_index=0, parameter_names=None)

        monkeypatch.setattr(sm_mod, "BackendBase", _FakeBackendBase)
        monkeypatch.setattr(sm_mod, "fit", fake_fit)
        sm = StatisticalModel(backend=NoNamesBackend(), analysis="nonames")

        with pytest.raises(ValueError, match="parameter_names not set"):
            sm.maximize_likelihood(poi_indices=["mu"])

    def test_partial_index_list(self, monkeypatch):
        sm = self._sm_with_fit(monkeypatch, [3.0, 0.7])
        result, _ = sm.maximize_likelihood(poi_indices=[1])
        assert list(result.keys()) == [1]
        assert result[1] == pytest.approx(0.7)

    def test_return_nll_false(self, monkeypatch):
        sm = self._sm_with_fit(monkeypatch, [1.0, 0.0])
        # logpdf = -10.0, so when return_nll=False we get exp(-10)
        _, val = sm.maximize_likelihood(poi_indices=None, return_nll=False)
        assert val == pytest.approx(np.exp(-10.0))


# ---------------------------------------------------------------------------
# StatisticalModel.maximize_asimov_likelihood with poi_indices
# ---------------------------------------------------------------------------
class TestMaximizeAsimovLikelihoodPoiIndices:
    def test_poi_indices_passed_through(self, monkeypatch):
        """maximize_asimov_likelihood must forward poi_indices to maximize_likelihood."""
        received = {}

        def fake_maximize_likelihood(self, poi_indices=None, **kwargs):
            received["poi_indices"] = poi_indices
            return {0: 1.5, 1: 0.3}, 7.0

        monkeypatch.setattr(sm_mod, "BackendBase", _FakeBackendBase)

        def fake_fit(**kwargs):
            return -1.0, np.array([0.0, 0.0])

        monkeypatch.setattr(sm_mod, "fit", fake_fit)
        monkeypatch.setattr(
            sm_mod.StatisticalModel, "maximize_likelihood", fake_maximize_likelihood
        )

        sm = _make_stat_model(monkeypatch)
        result, nll = sm.maximize_asimov_likelihood(poi_indices=[0, 1])
        assert received["poi_indices"] == [0, 1]
        assert result == {0: 1.5, 1: 0.3}
        assert nll == pytest.approx(7.0)

    def test_none_poi_indices_default_behaviour(self, monkeypatch):
        received = {}

        def fake_maximize_likelihood(self, poi_indices=None, **kwargs):
            received["poi_indices"] = poi_indices
            return 2.5, 3.0

        monkeypatch.setattr(sm_mod, "BackendBase", _FakeBackendBase)

        def fake_fit(**kwargs):
            return -1.0, np.array([0.0, 0.0])

        monkeypatch.setattr(sm_mod, "fit", fake_fit)
        monkeypatch.setattr(
            sm_mod.StatisticalModel, "maximize_likelihood", fake_maximize_likelihood
        )

        sm = _make_stat_model(monkeypatch)
        muhat, nll = sm.maximize_asimov_likelihood()
        assert received["poi_indices"] is None
        assert muhat == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# StatisticalModel.fixed_poi_sampler with dict poi_test
# ---------------------------------------------------------------------------
class TestFixedPoiSamplerDictPoiTest:
    def test_dict_poi_test_calls_fit(self, monkeypatch):
        fit_called = {"called": False}

        def fake_fit(**kwargs):
            fit_called["called"] = True
            fit_called["fixed_poi_value"] = kwargs.get("fixed_poi_value")
            return -1.0, np.array([1.0, 0.5])

        monkeypatch.setattr(sm_mod, "BackendBase", _FakeBackendBase)
        monkeypatch.setattr(sm_mod, "fit", fake_fit)

        class SamplerBackend(_FakeBackendBase):
            name = "sampler"

            def get_sampler(self, fit_param):
                return lambda size: np.ones((size, 1))

        sm = StatisticalModel(backend=SamplerBackend(), analysis="s")
        sm.fixed_poi_sampler(poi_test={0: 1.0, 1: 0.5}, size=5)

        assert fit_called["called"]
        assert isinstance(fit_called["fixed_poi_value"], dict)


# ---------------------------------------------------------------------------
# StatisticalModel.sigma_mu_from_hessian with dict poi_test
# ---------------------------------------------------------------------------
class TestSigmaMuFromHessianDictPoiTest:
    def test_dict_poi_test_forwarded_to_fit(self, monkeypatch):
        fit_called = {"called": False, "fixed": None}

        def fake_fit(**kwargs):
            fit_called["called"] = True
            fit_called["fixed"] = kwargs.get("fixed_poi_value")
            return -1.0, np.array([1.0, 0.0])

        def fake_hessian(params):
            return np.array([[-4.0, 0.0], [0.0, -9.0]])

        class HessianBackend(_FakeBackendBase):
            name = "hess"

            def get_hessian_logpdf_func(self, expected=None):
                return fake_hessian

        monkeypatch.setattr(sm_mod, "BackendBase", _FakeBackendBase)
        monkeypatch.setattr(sm_mod, "fit", fake_fit)

        sm = StatisticalModel(backend=HessianBackend(), analysis="h")
        sm.sigma_mu_from_hessian(poi_test={0: 1.0, 1: 0.0})

        assert fit_called["called"]
        assert isinstance(fit_called["fixed"], dict)


# ---------------------------------------------------------------------------
# UnCorrStatisticsCombiner.maximize_likelihood with poi_indices
# ---------------------------------------------------------------------------
from spey.combiner import uncorrelated_statistics_combiner as combiner_mod
from spey.combiner.uncorrelated_statistics_combiner import UnCorrStatisticsCombiner


class _FakeStatModel:
    """Minimal fake StatisticalModel for combiner tests."""

    def __init__(self, muhat=0.5, sigma=1.0, likelihood_val=1.0, minimum_poi=-5.0):
        self.analysis = "fake"
        self.backend_type = "fake"
        self.is_alive = True
        self.is_asymptotic_calculator_available = True
        self.is_chi_square_calculator_available = True
        self._muhat = muhat
        self._sigma = sigma
        self._likelihood_val = likelihood_val

        class _FB:
            def config(self):
                return types.SimpleNamespace(minimum_poi=minimum_poi)

        self.backend = _FB()

    def maximize_likelihood(self, expected=None, **kwargs):
        return float(self._muhat), 0.0

    def sigma_mu(self, poi_test=0.0, expected=None, **kwargs):
        return float(self._sigma)

    def likelihood(
        self, poi_test=1.0, expected=None, data=None, return_nll=True, **kwargs
    ):
        return float(self._likelihood_val)

    def generate_asimov_data(self, expected=None, test_statistic="qtilde", **kwargs):
        return []


def _make_combiner(monkeypatch, muhat=0.5, sigma=1.0):
    monkeypatch.setattr(combiner_mod, "StatisticalModel", _FakeStatModel)
    model = _FakeStatModel(muhat=muhat, sigma=sigma)
    combiner = UnCorrStatisticsCombiner(model)
    return combiner


class TestCombinerMaximizeLikelihoodPoiIndices:
    def test_none_returns_float(self, monkeypatch):
        combiner = _make_combiner(monkeypatch)

        def fake_fit(func, model_configuration, **kwargs):
            return 2.0, np.array([0.5])

        monkeypatch.setattr(combiner_mod, "fit", fake_fit)
        result, nll = combiner.maximize_likelihood(poi_indices=None)
        assert isinstance(result, float)
        assert result == pytest.approx(0.5)

    def test_int_key_list_returns_dict(self, monkeypatch):
        combiner = _make_combiner(monkeypatch)

        def fake_fit(func, model_configuration, **kwargs):
            return 2.0, np.array([0.5])

        monkeypatch.setattr(combiner_mod, "fit", fake_fit)
        result, nll = combiner.maximize_likelihood(poi_indices=[0])
        assert isinstance(result, dict)
        assert result[0] == pytest.approx(0.5)

    def test_str_key_list_returns_dict_with_str_keys(self, monkeypatch):
        combiner = _make_combiner(monkeypatch)

        def fake_fit(func, model_configuration, **kwargs):
            return 2.0, np.array([0.75])

        monkeypatch.setattr(combiner_mod, "fit", fake_fit)
        result, nll = combiner.maximize_likelihood(poi_indices=["mu"])
        assert isinstance(result, dict)
        assert result["mu"] == pytest.approx(0.75)

    def test_multiple_keys_all_get_same_value(self, monkeypatch):
        """Combiner has a single scalar POI; all requested keys map to it."""
        combiner = _make_combiner(monkeypatch)

        def fake_fit(func, model_configuration, **kwargs):
            return 2.0, np.array([1.23])

        monkeypatch.setattr(combiner_mod, "fit", fake_fit)
        result, _ = combiner.maximize_likelihood(poi_indices=[0, "mu", "sig"])
        assert set(result.keys()) == {0, "mu", "sig"}
        for v in result.values():
            assert v == pytest.approx(1.23)


class TestCombinerMaximizeAsimovLikelihoodPoiIndices:
    def test_none_returns_float(self, monkeypatch):
        combiner = _make_combiner(monkeypatch)

        def fake_fit(func, model_configuration, **kwargs):
            return 3.0, np.array([0.0])

        monkeypatch.setattr(combiner_mod, "fit", fake_fit)
        result, nll = combiner.maximize_asimov_likelihood(poi_indices=None)
        assert isinstance(result, float)

    def test_int_key_returns_dict(self, monkeypatch):
        combiner = _make_combiner(monkeypatch)

        def fake_fit(func, model_configuration, **kwargs):
            return 3.0, np.array([0.9])

        monkeypatch.setattr(combiner_mod, "fit", fake_fit)
        result, _ = combiner.maximize_asimov_likelihood(poi_indices=[0])
        assert isinstance(result, dict)
        assert result[0] == pytest.approx(0.9)
