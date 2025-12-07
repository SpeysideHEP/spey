import numpy as np
import types

import pytest

from spey.optimizer import core as core_mod


class FakeModelConfig:
    def __init__(self, suggested_init, poi_index=0, suggested_fixed=None):
        self.suggested_init = list(suggested_init)
        self.poi_index = poi_index
        self.suggested_fixed = (
            list(suggested_fixed) if suggested_fixed is not None else None
        )

    def fixed_poi_bounds(self, fixed_poi_value):
        # return same-length bounds as suggested_init
        return [(-10.0, 10.0) for _ in self.suggested_init]


def test_fit_respects_fixed_poi_and_suggested_fixed_and_passes_poi_index(monkeypatch):
    calls = {}

    def fake_minimizer(
        func,
        init_pars,
        fixed_vals,
        do_grad,
        hessian,
        bounds,
        constraints,
        **kwargs,
    ):
        # capture what was passed in
        calls["init_pars"] = np.asarray(init_pars).copy()
        calls["fixed_vals"] = list(fixed_vals)
        calls["do_grad"] = do_grad
        calls["bounds"] = list(bounds)
        calls["constraints"] = list(constraints)
        calls["kwargs"] = dict(kwargs)
        # return a dummy function value and parameters
        return 7.7, np.asarray(init_pars)

    # monkeypatch the internal _get_minimizer to return our fake_minimizer
    def fake_get_minimizer(name):
        calls["minimizer_name"] = name
        return fake_minimizer

    monkeypatch.setattr(core_mod, "_get_minimizer", fake_get_minimizer)

    cfg = FakeModelConfig(
        suggested_init=[0.0, 1.0], poi_index=0, suggested_fixed=[False, True]
    )

    # call fit with a fixed_poi_value -> should set init_pars[0] and fixed_vals[0]
    fun, x = core_mod.fit(
        func=lambda p: np.sum(np.asarray(p) ** 2),
        model_configuration=cfg,
        do_grad=False,
        hessian=None,
        initial_parameters=None,
        bounds=None,
        fixed_poi_value=3.14,
        logpdf=None,
        constraints=None,
        verbose=False,  # extra option to ensure kwargs capturing works
    )

    assert pytest.approx(7.7, rel=1e-12) == fun
    # init_pars[poi_index] should be set to fixed_poi_value
    assert np.allclose(calls["init_pars"][0], 3.14)
    # suggested_fixed second parameter was True -> fixed_vals[1] should be True
    assert calls["fixed_vals"][0] is True
    assert calls["fixed_vals"][1] is True
    # poi_index must have been injected into kwargs passed to minimizer
    assert "poi_index" in calls["kwargs"]

    # and _get_minimizer should have been invoked with 'scipy' (default)
    assert calls["minimizer_name"] == "scipy"


def test_fit_returns_logpdf_when_provided_and_uses_initial_parameters(monkeypatch):
    captured = {}

    def fake_minimizer(
        func,
        init_pars,
        fixed_vals,
        do_grad,
        hessian,
        bounds,
        constraints,
        **kwargs,
    ):
        # return a nontrivial x so we can validate logpdf is evaluated on it
        return -0.1, np.asarray([9.9, -1.1])

    monkeypatch.setattr(core_mod, "_get_minimizer", lambda name: fake_minimizer)

    cfg = FakeModelConfig(suggested_init=[0.0, 0.0], poi_index=1, suggested_fixed=None)

    # define a logpdf that inspects the provided x
    def my_logpdf(x):
        captured["x"] = np.asarray(x).copy()
        return float(100.0 + np.sum(np.asarray(x)))

    # supply initial_parameters to ensure they are used as starting point
    fun_or_logpdf, x = core_mod.fit(
        func=lambda p: np.sum(np.asarray(p) ** 2),
        model_configuration=cfg,
        do_grad=True,
        initial_parameters=[1.0, 2.0],
        logpdf=my_logpdf,
        poi_index=cfg.poi_index,
    )

    # fit should return (logpdf(x), x) as per implementation
    assert np.allclose(x, np.asarray([9.9, -1.1]))
    # logpdf should have been called with x returned by minimizer
    assert "x" in captured
    assert np.allclose(captured["x"], x)
    assert pytest.approx(100.0 + np.sum(x), rel=1e-12) == fun_or_logpdf


def test_invalid_minimizer_option_is_replaced_with_scipy(monkeypatch):
    seen = {}

    def fake_get_minimizer(name):
        # record name argument; return a simple fake minimizer
        seen["name"] = name

        def fake_min(
            func, init_pars, fixed_vals, do_grad, hessian, bounds, constraints, **kwargs
        ):
            return 0.0, np.asarray(init_pars)

        return fake_min

    monkeypatch.setattr(core_mod, "_get_minimizer", fake_get_minimizer)

    cfg = FakeModelConfig(suggested_init=[0.0], poi_index=0)
    # pass an invalid minimizer name via options -> code should warn and switch to 'scipy'
    fun, x = core_mod.fit(
        func=lambda p: 0.0,
        model_configuration=cfg,
        minimizer="INVALID_NAME",
    )

    # _get_minimizer should have been called with 'scipy' after the fallback
    assert seen.get("name") == "scipy"
    assert pytest.approx(0.0, rel=1e-12) == fun
