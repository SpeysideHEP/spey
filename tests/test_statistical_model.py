import types
import numpy as np
import pytest

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
