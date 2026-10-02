import types
import numpy as np
import pytest

import spey
import spey.interface.statistical_model as sm_mod
from spey.interface.statistical_model import StatisticalModel
from spey.system.exceptions import MethodNotAvailable, UnknownCrossSection


class FakeBackendBase:
    """Minimal BackendBase stand-in for isinstance checks in StatisticalModel"""

    def __init__(self):
        # default constraints list
        self.constraints = []

    # placeholders to be overridden by concrete fake backends in tests
    def expected_data(self, *args, **kwargs):
        raise NotImplementedError

    def get_objective_function(self, *args, **kwargs):
        raise NotImplementedError

    def get_logpdf_func(self, *args, **kwargs):
        raise NotImplementedError

    def config(self, *args, **kwargs):
        return types.SimpleNamespace(npar=1, poi_index=0)

    def get_sampler(self, *args, **kwargs):
        raise NotImplementedError

    def negative_loglikelihood(self, *args, **kwargs):
        raise NotImplementedError

    def asimov_negative_loglikelihood(self, *args, **kwargs):
        raise NotImplementedError

    def minimize_negative_loglikelihood(self, *args, **kwargs):
        raise NotImplementedError

    def minimize_asimov_negative_loglikelihood(self, *args, **kwargs):
        raise NotImplementedError


def make_fake_backend(
    *,
    expected_data_ret=None,
    logpdf_func=None,
    objective_raises_on_grad=False,
    get_sampler_impl=None,
    get_hessian_impl=None,
    name="FakeBackend",
):
    class FakeBackend(FakeBackendBase):
        def __init__(self):
            super().__init__()
            self.name = name

        def expected_data(self, pars):
            return expected_data_ret if expected_data_ret is not None else [0.0]

        def get_logpdf_func(self, expected=None, data=None):
            if logpdf_func is not None:
                return logpdf_func
            return lambda params: -float(np.sum(np.asarray(params) ** 2))

        def get_objective_function(self, expected=None, data=None, do_grad=True):
            if objective_raises_on_grad and do_grad:
                raise NotImplementedError
            # return a simple objective that sums negative squares
            if do_grad:
                return lambda pars: (
                    float(-np.sum(np.asarray(pars) ** 2)),
                    -2.0 * np.asarray(pars),
                )
            return lambda pars: float(-np.sum(np.asarray(pars) ** 2))

        def get_sampler(self, fit_param):
            if get_sampler_impl is None:
                raise NotImplementedError
            return get_sampler_impl

        def get_hessian_logpdf_func(self, expected=None):
            if get_hessian_impl is None:
                raise NotImplementedError
            return get_hessian_impl

        def config(self, *args, **kwargs):
            # default config: single parameter with poi_index 0
            return types.SimpleNamespace(npar=1, poi_index=0)

    return FakeBackend()


def test_constructor_repr_and_available_calculators(monkeypatch):
    # ensure StatisticalModel sees our FakeBackendBase as the abstract BackendBase
    monkeypatch.setattr(sm_mod, "BackendBase", FakeBackendBase)

    fb = make_fake_backend()
    sm = StatisticalModel(backend=fb, analysis="test_analysis", xsection=np.nan)
    r = repr(sm)
    assert "test_analysis" in r
    # available_calculators should always include 'chi_square'
    assert "chi_square" in sm.available_calculators


def test_prepare_for_fit_handles_missing_gradient(monkeypatch):
    monkeypatch.setattr(sm_mod, "BackendBase", FakeBackendBase)

    # backend that raises when do_grad=True
    fb = make_fake_backend(objective_raises_on_grad=True)
    sm = StatisticalModel(backend=fb, analysis="gfit")
    # prepare_for_fit should fall back to do_grad=False
    opts = sm.prepare_for_fit(expected=None)
    assert opts["do_grad"] is False
    assert "func" in opts and callable(opts["func"])
    assert "logpdf" in opts and callable(opts["logpdf"])


def test_likelihood_direct_logpdf_branch(monkeypatch):
    monkeypatch.setattr(sm_mod, "BackendBase", FakeBackendBase)

    # fake backend with npar=1 and poi_index set -> direct logpdf call path
    fb = make_fake_backend()
    # override config to signal single-parameter with poi_index
    fb.config = lambda **kw: types.SimpleNamespace(npar=1, poi_index=0)
    sm = StatisticalModel(backend=fb, analysis="onepar")
    # stub fit to ensure it's not used; if used tests will fail
    monkeypatch.setattr(
        sm_mod,
        "fit",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fit should not be called")),
    )
    # logpdf([poi_test]) will be computed by backend.get_logpdf_func
    fval_nll = sm.likelihood(poi_test=2.0, return_nll=True)
    # backend logpdf for params [2.0] = -sum(params^2) = -4.0 -> likelihood returns -logpdf = 4.0
    assert pytest.approx(4.0, rel=1e-12) == fval_nll


def test_generate_asimov_and_asimov_likelihood(monkeypatch):
    monkeypatch.setattr(sm_mod, "BackendBase", FakeBackendBase)

    # prepare backend whose expected_data echoes fit parameters
    fb = make_fake_backend(expected_data_ret=[42.0, 7.0])
    # ensure config indicates multi-parameter so fit() is used
    fb.config = lambda **kw: types.SimpleNamespace(npar=2, poi_index=0)
    sm = StatisticalModel(backend=fb, analysis="asimov_test")

    # monkeypatch fit to return a known logpdf and fit_params
    def fake_fit(func, model_configuration=None, **kwargs):
        # return a logpdf and fit parameters (array-like)
        return -3.5, [1.0, 0.5]

    monkeypatch.setattr(sm_mod, "fit", fake_fit)

    data = sm.generate_asimov_data(expected=None, test_statistic="qtilde")
    assert data == [42.0, 7.0]

    # asimov_likelihood should call generate_asimov_data and then likelihood; monkeypatch likelihood to observe call
    monkeypatch.setattr(sm_mod.StatisticalModel, "likelihood", lambda self, **kw: 123.0)
    al = sm.asimov_likelihood()
    assert al == 123.0


def test_fixed_poi_sampler_raises_when_sampler_not_available(monkeypatch):
    monkeypatch.setattr(sm_mod, "BackendBase", FakeBackendBase)

    fb = make_fake_backend()
    fb.config = lambda **kw: types.SimpleNamespace(npar=2, poi_index=0)
    sm = StatisticalModel(backend=fb, analysis="sampler_test")

    # monkeypatch fit to supply fit parameters (unused since get_sampler raises)
    monkeypatch.setattr(sm_mod, "fit", lambda **kwargs: (-1.0, [0.1, 0.2]))

    with pytest.raises(MethodNotAvailable):
        sm.fixed_poi_sampler(poi_test=1.0, size=10)


def test_sigma_mu_from_hessian_behaviour(monkeypatch):
    monkeypatch.setattr(sm_mod, "BackendBase", FakeBackendBase)

    # provide a hessian function returning a 2x2 matrix; inverse's [0,0] sqrt should be known
    def fake_hessian_func(params):
        # return negative Hessian input expected by sigma_mu_from_hessian: function returns hessian of logpdf
        # In StatisticalModel.sigma_mu_from_hessian they call hessian = -1.0 * hessian_func(fit_param)
        # So provide hessian_func that returns -H, so final hessian = -(-H) = H
        return np.array([[-4.0, 0.0], [0.0, -9.0]])

    fb = make_fake_backend(get_hessian_impl=fake_hessian_func)
    fb.config = lambda **kw: types.SimpleNamespace(npar=2, poi_index=0)
    sm = StatisticalModel(backend=fb, analysis="hessian_test")

    # monkeypatch fit to return fit parameters (unused in hessian computation beyond index)
    monkeypatch.setattr(sm_mod, "fit", lambda **kwargs: (-2.0, [0.0, 0.0]))

    sigma = sm.sigma_mu_from_hessian(poi_test=1.0)
    # Hessian after sign correction becomes [[4,0],[0,9]] -> inv has [0,0]=1/4 -> sqrt = 0.5
    assert pytest.approx(0.5, rel=1e-12) == sigma


def test_excluded_cross_section_raises_on_nan_xsection(monkeypatch):
    monkeypatch.setattr(sm_mod, "BackendBase", FakeBackendBase)
    fb = make_fake_backend()
    sm = StatisticalModel(backend=fb, analysis="xs_test", xsection=np.nan)
    with pytest.raises(UnknownCrossSection):
        _ = sm.excluded_cross_section()


# ---------------------------------------------------------------------------
# Capability reporting (regression: bound method vs plain function comparison)
# ---------------------------------------------------------------------------
class _BareBackend(spey.BackendBase):
    """Backend implementing only the two abstract methods."""

    name = "test.bare"
    version = "1.0.0"
    author = "test"
    spey_requires = spey.__version__

    def config(self, allow_negative_signal=True, poi_upper_bound=10.0):
        from spey.base.model_config import ModelConfig

        return ModelConfig(0, -1.0, [1.0], [(-1.0, poi_upper_bound)])

    def get_logpdf_func(self, expected=spey.ExpectationType.observed, data=None):
        return lambda pars: -0.5 * (pars[0] - 0.5) ** 2


class _RicherBackend(_BareBackend):
    """Backend that also implements the optional Asimov and sampling hooks."""

    name = "test.richer"

    def expected_data(self, pars, **kwargs):
        return [float(pars[0])]

    def get_sampler(self, pars):
        return lambda size, *args, **kwargs: np.zeros((size, 1))


def test_calculator_availability_reflects_the_backend():
    """A backend that implements nothing optional only offers the chi-square path."""
    bare = StatisticalModel(backend=_BareBackend(), analysis="bare")
    assert bare.is_asymptotic_calculator_available is False
    assert bare.is_toy_calculator_available is False
    assert bare.is_chi_square_calculator_available is True
    assert bare.available_calculators == ["chi_square"]

    richer = StatisticalModel(backend=_RicherBackend(), analysis="richer")
    assert richer.is_asymptotic_calculator_available is True
    assert richer.is_toy_calculator_available is True
    assert sorted(richer.available_calculators) == ["asymptotic", "chi_square", "toy"]


def test_calculator_availability_of_default_backends():
    """The built-in backends must keep advertising every calculator."""
    model = spey.get_backend("default.poisson")(
        signal_yields=[3.0], background_yields=[10.0], data=[11], analysis="p"
    )
    assert model.is_asymptotic_calculator_available
    assert model.is_toy_calculator_available


def test_capability_can_be_disabled_per_instance():
    """Binding the base implementation onto an instance disables the capability.

    This is how `CorrelatedStatisticsCombiner` reports that one of its constituent
    models cannot generate expected data.
    """
    backend = _RicherBackend()
    backend.expected_data = spey.BackendBase.expected_data.__get__(backend, type(backend))
    model = StatisticalModel(backend=backend, analysis="disabled")
    assert model.is_asymptotic_calculator_available is False
    assert model.is_toy_calculator_available is True


def test_sigma_mu_from_hessian_warns_on_indefinite_information(monkeypatch, caplog):
    """A negative POI variance is reported explicitly rather than as a bare nan."""
    import logging

    monkeypatch.setattr(sm_mod, "BackendBase", FakeBackendBase)
    fb = make_fake_backend()
    # log-likelihood Hessian whose sign-flipped inverse has a negative (0, 0) entry
    fb.get_hessian_logpdf_func = lambda *args, **kwargs: (
        lambda pars: np.array([[4.0, 0.0], [0.0, -9.0]])
    )
    fb.config = lambda *args, **kwargs: types.SimpleNamespace(
        npar=2, poi_index=0, suggested_init=[1.0, 0.0]
    )
    monkeypatch.setattr(sm_mod, "fit", lambda **kwargs: (-2.0, [0.0, 0.0]))

    model = StatisticalModel(backend=fb, analysis="indefinite")

    spey_logger = logging.getLogger("Spey")
    previous = spey_logger.propagate
    spey_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="Spey"):
            sigma = model.sigma_mu_from_hessian(poi_test=1.0)
    finally:
        spey_logger.propagate = previous

    assert np.isnan(sigma)
    assert "not positive definite" in caplog.text
