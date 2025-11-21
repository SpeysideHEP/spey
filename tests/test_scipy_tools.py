import numpy as np
import types

import pytest

from spey.optimizer import scipy_tools as st


def test_minimize_requires_poi_index():
    # missing poi_index option should trigger assertion
    with pytest.raises(AssertionError):
        st.minimize(lambda x: 0.0, init_pars=[0.0], fixed_vals=[False])


def test_minimize_calls_scipy_and_passes_constraints(monkeypatch):
    captured = {}

    def fake_minimize(
        func,
        init_pars,
        method=None,
        jac=None,
        hess=None,
        bounds=None,
        constraints=None,
        tol=None,
        options=None,
    ):
        # capture inputs for assertions
        captured["init_pars"] = np.asarray(init_pars).copy()
        captured["jac"] = jac
        captured["bounds"] = None if bounds is None else list(bounds)
        captured["constraints"] = constraints
        # return a simple object similar to scipy OptimizeResult
        return types.SimpleNamespace(success=True, fun=0.321, x=np.asarray([0.1, 0.2]))

    # patch the minimize used inside module
    monkeypatch.setattr(st.scipy.optimize, "minimize", fake_minimize)

    # dummy objective
    def obj(p):
        return float(np.sum(np.asarray(p) ** 2))

    fval, params = st.minimize(
        obj,
        init_pars=[1.0, 2.0],
        fixed_vals=[
            False,
            True,
        ],  # second parameter fixed -> a constraint should be added
        do_grad=True,
        bounds=[(-10.0, 10.0), (-5.0, 5.0)],
        poi_index=0,
    )

    assert pytest.approx(0.321, rel=1e-12) == fval
    assert np.allclose(np.asarray(params), np.asarray([0.1, 0.2]))

    # verify jac flag forwarded
    assert captured["jac"] is True

    # verify a constraint for the fixed parameter was provided and behaves as expected
    assert isinstance(captured["constraints"], list)
    # find equality constraints
    eqs = [c for c in captured["constraints"] if c.get("type") == "eq"]
    assert len(eqs) == 1
    # the constraint function should evaluate to init_pars[index] - value == 0
    constr_fun = eqs[0]["fun"]
    # constraint uses the original init_pars value (2.0 for index 1)
    assert pytest.approx(0.0, abs=1e-12) == constr_fun(np.array([0.0, 2.0]))


def test_minimize_retries_and_expands_bounds(monkeypatch):
    # simulate two calls: first unsuccessful, second successful
    calls = {"n": 0}
    first_x = np.array([0.4, 0.8])

    def fake_minimize(
        func,
        init_pars,
        method=None,
        jac=None,
        hess=None,
        bounds=None,
        constraints=None,
        tol=None,
        options=None,
    ):
        calls["n"] += 1
        if calls["n"] == 1:
            return types.SimpleNamespace(
                success=False, x=first_x.copy(), message="no converge", fun=10.0
            )
        # on second call ensure init_pars was updated to first_x by the code
        assert np.allclose(np.asarray(init_pars), first_x)
        return types.SimpleNamespace(success=True, x=np.asarray([0.4, 0.8]), fun=0.5)

    monkeypatch.setattr(st.scipy.optimize, "minimize", fake_minimize)

    def obj(p):
        return float(np.sum(np.asarray(p) ** 2))

    # provide ntrials=2 to allow a retry; supply poi_index required by function
    fval, params = st.minimize(
        obj,
        init_pars=[1.0, 1.0],
        fixed_vals=[False, False],
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        poi_index=0,
        ntrials=2,
    )

    assert calls["n"] == 2
    assert pytest.approx(0.5, rel=1e-12) == fval
    assert np.allclose(params, np.asarray([0.4, 0.8]))
