import types

import numpy as np
import pytest

from spey.base import backend_base


class SimpleBackend(backend_base.BackendBase):
    """Minimal concrete BackendBase implementation for testing defaults."""

    def config(self, allow_negative_signal: bool = True, poi_upper_bound: float = 10.0):
        return types.SimpleNamespace(npar=1, poi_index=0)

    def get_logpdf_func(self, expected=None, data=None):
        # simple logpdf: sum of params
        return lambda pars: float(np.sum(np.atleast_1d(pars)))

    def expected_data(self, pars):
        return [1.0]


def test_is_alive_default_true():
    b = SimpleBackend()
    assert b.is_alive is True


def test_get_objective_function_without_grad_returns_negative_logpdf():
    b = SimpleBackend()
    obj = b.get_objective_function(do_grad=False)
    # logpdf([1,2]) = 3.0 -> objective should be -3.0
    val = obj(np.array([1.0, 2.0]))
    assert pytest.approx(-3.0, rel=1e-12) == float(val)


def test_get_objective_function_with_grad_raises():
    b = SimpleBackend()
    with pytest.raises(NotImplementedError):
        b.get_objective_function(do_grad=True)


def test_hessian_sampler_and_negative_loglikelihood_raise_not_implemented():
    b = SimpleBackend()
    with pytest.raises(NotImplementedError):
        b.get_hessian_logpdf_func()
    with pytest.raises(NotImplementedError):
        b.get_sampler(np.array([0.0]))
    with pytest.raises(NotImplementedError):
        b.negative_loglikelihood()


def test_converterbase_call_raises_not_implemented():
    conv = backend_base.ConverterBase()
    with pytest.raises(NotImplementedError):
        conv()
