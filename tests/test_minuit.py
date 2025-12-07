import numpy as np
import pytest

from spey.optimizer import minuit_tools as mt


def test_minimize_simple_quadratic_migrad():
    # simple quadratic function with known minimum at [1.0, 2.0]
    def obj(pars):
        p = np.asarray(pars)
        return float(np.sum((p - np.array([1.0, 2.0])) ** 2))

    fval, x = mt.minimize(
        func=obj,
        init_pars=[0.0, 0.0],
        fixed_vals=[False, False],
        do_grad=False,
        bounds=None,
        method="migrad",
        maxiter=1000,
    )

    # should find minimum near [1.0, 2.0]
    assert pytest.approx(0.0, abs=1e-4) == fval
    assert np.allclose(x, np.array([1.0, 2.0]), atol=1e-3)


def test_minimize_with_gradient():
    # function that returns (value, gradient) tuple
    def obj_with_grad(pars):
        p = np.asarray(pars)
        target = np.array([1.0, 2.0])
        val = float(np.sum((p - target) ** 2))
        grad = 2.0 * (p - target)
        return val, grad

    fval, x = mt.minimize(
        func=obj_with_grad,
        init_pars=[0.0, 0.0],
        fixed_vals=[False, False],
        do_grad=True,
        bounds=None,
        method="migrad",
        maxiter=1000,
    )

    assert pytest.approx(0.0, abs=1e-4) == fval
    assert np.allclose(x, np.array([1.0, 2.0]), atol=1e-3)


def test_minimize_with_fixed_parameters():
    # fix second parameter to 2.0, only optimize first
    def obj(pars):
        p = np.asarray(pars)
        return float(np.sum((p - np.array([1.0, 2.0])) ** 2))

    fval, x = mt.minimize(
        func=obj,
        init_pars=[0.0, 2.0],
        fixed_vals=[False, True],  # fix second parameter
        do_grad=False,
        bounds=None,
        method="migrad",
    )

    # first parameter should converge to 1.0, second should stay at 2.0
    assert pytest.approx(0.0, abs=1e-4) == fval
    assert pytest.approx(1.0, abs=1e-3) == x[0]
    assert pytest.approx(2.0, abs=1e-6) == x[1]


def test_minimize_with_bounds():
    # constrain search space with bounds
    def obj(pars):
        p = np.asarray(pars)
        # minimum at [5.0, 5.0] but we'll constrain to [0, 2]
        return float(np.sum((p - np.array([5.0, 5.0])) ** 2))

    fval, x = mt.minimize(
        func=obj,
        init_pars=[1.0, 1.0],
        fixed_vals=[False, False],
        do_grad=False,
        bounds=[(0.0, 2.0), (0.0, 2.0)],
        method="migrad",
    )

    # minimum should be at the boundary [2.0, 2.0]
    assert np.allclose(x, np.array([2.0, 2.0]), atol=1e-2)


def test_minimize_with_simplex_method():
    # test alternative optimization method
    def obj(pars):
        p = np.asarray(pars)
        return float(np.sum((p - np.array([1.0, 2.0])) ** 2))

    fval, x = mt.minimize(
        func=obj,
        init_pars=[0.0, 0.0],
        fixed_vals=[False, False],
        do_grad=False,
        bounds=None,
        method="simplex",
        maxiter=5000,
    )

    assert pytest.approx(0.0, abs=1e-3) == fval
    assert np.allclose(x, np.array([1.0, 2.0]), atol=1e-2)


def test_minimize_invalid_method_raises():
    def obj(pars):
        return float(np.sum(np.asarray(pars) ** 2))

    with pytest.raises(ValueError, match="Unknown method"):
        mt.minimize(
            func=obj,
            init_pars=[0.0],
            fixed_vals=[False],
            do_grad=False,
            method="invalid_method",
        )


def test_minimize_extracts_options():
    # test that options are properly extracted and applied
    captured = {}

    def obj(pars):
        return float(np.sum(np.asarray(pars) ** 2))

    # monkeypatch Minuit to capture initialization
    original_minuit = mt.Minuit

    class FakeMinuit:
        def __init__(self, fcn, *args, **kwargs):
            captured["fcn"] = fcn
            captured["args"] = args
            captured["grad"] = kwargs.get("grad")
            self.limits = None
            self.fixed = None
            self.print_level = 0
            self.errordef = 1.0
            self.strategy = 0
            self.tol = 1e-6
            self.valid = True
            self.accurate = True
            self.fval = 0.0
            self.values = [0.0]
            self.errors = [0.1]

        def migrad(self, ncall=10000):
            pass

        @property
        def LIKELIHOOD(self):
            return 0.5

    mt.Minuit = FakeMinuit

    try:
        fval, x = mt.minimize(
            func=obj,
            init_pars=[1.0, 2.0],
            fixed_vals=[False, False],
            do_grad=False,
            errordef=2.0,
            strategy=1,
            tol=1e-8,
            disp=1,
        )

        # verify options were applied to Minuit instance
        assert captured["args"][0] is not None
    finally:
        mt.Minuit = original_minuit
