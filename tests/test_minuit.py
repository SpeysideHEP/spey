# python
import numpy as np
import pytest

# relative import of the module and the function to test
from spey.optimizer import minuit_tools as mt
from spey.optimizer.minuit_tools import minimize


class FakeFMin:
    def __init__(self, call_limit=False, above_max_edm=False):
        self.has_reached_call_limit = call_limit
        self.is_above_max_edm = above_max_edm


class FakeMinuit:
    """
    Simple fake replacement for iminuit.Minuit for 1D objectives.
    Stores last_instance for inspection by tests.
    Performs a brute-force 1D scan during migrad to find a minimum.
    """

    last_instance = None

    def __init__(self, objective, init_vals, grad=None):
        # objective: callable that accepts a numpy array and returns a float (or tuple when using do_grad)
        self.objective = objective
        self.init = np.atleast_1d(init_vals).astype(float)
        self.grad = grad
        self.limits = None
        self.fixed = None
        self.print_level = None
        self.errordef = None
        self.strategy = None
        self.tol = None
        self.errors = None
        # initialize values and fval using the provided objective
        try:
            val = self.objective(self.init)
        except Exception:
            # objective might return (value, grad) in some cases; handle gracefully
            try:
                val = self.objective(self.init)[0]
            except Exception:
                val = float("inf")
        self.values = self.init.copy()
        self.fval = float(val)
        self.valid = True
        self.fmin = FakeFMin(call_limit=False, above_max_edm=False)
        FakeMinuit.last_instance = self

    def migrad(self, ncall=10000):
        # only implement a simple 1D brute-force minimizer for tests
        if self.init.size != 1:
            # for multi-dim, just keep initial
            return
        # use bounds if set, otherwise default search range
        low, high = -10.0, 10.0
        if self.limits and self.limits[0] is not None:
            low = self.limits[0][0] if self.limits[0][0] is not None else low
            high = self.limits[0][1] if self.limits[0][1] is not None else high
        xs = np.linspace(low, high, 1001)
        vals = []
        for x in xs:
            try:
                v = self.objective(np.array([x]))
            except Exception:
                # if objective returns tuple (value, grad)
                v = self.objective(np.array([x]))[0]
            vals.append(float(v))
        idx = int(np.argmin(vals))
        self.values = np.array([xs[idx]])
        self.fval = float(vals[idx])
        # set a dummy error
        self.errors = np.array([0.0])


class FakeMinuitInvalid(FakeMinuit):
    def __init__(self, objective, init_vals, grad=None):
        super().__init__(objective, init_vals, grad=grad)
        self.valid = False
        # make fmin show both problematic flags to follow the failure branch
        self.fmin = FakeFMin(call_limit=True, above_max_edm=True)


def test_minimize_success(monkeypatch):
    # replace iminuit.Minuit with our fake
    monkeypatch.setattr(mt.iminuit, "Minuit", FakeMinuit)

    # simple 1D quadratic objective with minimum at x=3
    def obj(pars):
        x = float(np.atleast_1d(pars)[0])
        return (x - 3.0) ** 2

    fval, vals = minimize(
        obj,
        init_pars=[0.0],
        fixed_vals=[False],
        do_grad=False,
        bounds=[(-10.0, 10.0)],
        errordef=1,
        maxiter=500,
        disp=True,
        tol=1e-4,
    )

    # returned fval should be approximately zero at the minimum and value close to 3
    assert pytest.approx(0.0, rel=1e-6, abs=1e-6) == fval
    assert pytest.approx(3.0, rel=1e-6, abs=1e-6) == float(np.atleast_1d(vals)[0])

    # inspect the FakeMinuit instance to ensure options were applied
    inst = FakeMinuit.last_instance
    assert inst is not None
    assert inst.print_level is True  # disp was True
    assert inst.errordef == 1
    assert inst.tol == 1e-4
    # strategy should be int(do_grad) -> 0 here
    assert inst.strategy == 0


def test_minimize_with_grad_and_failed_opt(monkeypatch):
    # replace iminuit.Minuit with an invalid fake
    monkeypatch.setattr(mt.iminuit, "Minuit", FakeMinuitInvalid)

    # objective returns (value, gradient) to trigger the do_grad branch
    def func_with_grad(pars):
        x = float(np.atleast_1d(pars)[0])
        val = (x - (-2.0)) ** 2  # minimum at x=-2
        grad = np.array([2.0 * (x + 2.0)])
        return val, grad

    fval, vals = minimize(
        func_with_grad,
        init_pars=[0.0],
        fixed_vals=[False],
        do_grad=True,
        bounds=[(-10.0, 10.0)],
        disp=False,
        errordef=0.5,
        strategy=1,
        tol=1e-5,
        maxiter=10,
    )

    # result should still exist and be near the minimum (-2.0)
    assert pytest.approx(0.0, rel=1e-6, abs=1e-6) == fval
    assert pytest.approx(-2.0, rel=1e-6, abs=1e-6) == float(np.atleast_1d(vals)[0])

    # verify that the FakeMinuitInvalid instance reported invalid and had fmin flags set
    inst = FakeMinuitInvalid.last_instance
    assert inst is not None
    assert inst.valid is False
    assert inst.fmin.has_reached_call_limit is True
    assert inst.fmin.is_above_max_edm is True
    # ensure the strategy (should be int(do_grad) == 1) was set from options or do_grad
    # minimize sets strategy = options.get("strategy", int(do_grad))
    # we passed strategy=1 explicitly; ensure it was applied
    assert inst.strategy == 1
