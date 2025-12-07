import numpy as onp
import math

import pytest

from spey.backends.distributions import (
    Poisson,
    Normal,
    MultivariateNormal,
    MainModel,
    ConstraintModel,
    MixtureModel,
)


def test_poisson_expected_and_logprob_and_sample():
    # negative loc gets clipped to small positive value
    p = Poisson(onp.array([-1.0]))
    exp = p.expected_data()
    assert float(exp[0]) >= 1e-20

    # explicit positive loc: check log_prob formula for a float input
    loc = onp.array([3.0])
    p2 = Poisson(loc)
    val = onp.array([2.0])
    got = float(p2.log_prob(val)[0])
    # manual expected: value*log(loc) - loc - gammaln(value+1)
    # use scipy.special.gammaln for reference via math.lgamma (lgamma(x) = gammaln(x))
    expected = float(
        val[0] * math.log(float(loc[0])) - float(loc[0]) - math.lgamma(val[0] + 1.0)
    )
    assert pytest.approx(expected, rel=1e-6) == got

    # sampling shape
    samp = p2.sample(4)
    assert samp.shape[0] == 4


def test_normal_log_prob_simple():
    loc = onp.array([0.0])
    scale = onp.array([1.0])
    domain = slice(0, 1)
    nm = Normal(loc=loc, scale=scale, domain=domain)
    pars = onp.array([0.5])
    got = float(nm.log_prob(pars)[0])
    s = float(scale[0])
    diff = float(pars[domain][0] - loc[0])
    expected = -math.log(s) - 0.5 * math.log(2.0 * math.pi) - 0.5 * (diff / s) ** 2
    assert pytest.approx(expected, rel=1e-7) == got


def test_multivariate_normal_log_prob_identity_cov():
    mean = onp.array([0.0, 0.0])
    cov = onp.eye(2)
    mv = MultivariateNormal(mean=mean, cov=cov)
    pars = onp.array([0.0, 0.0])
    got = float(mv.log_prob(pars))
    # for zero vector and identity cov, expected = -0.5*(0) -0.5*(2*log(2π)+log(1)) = -ln(2π)
    expected = -math.log(2.0 * math.pi)
    assert pytest.approx(expected, rel=1e-7) == got


def test_mainmodel_expected_and_sample_poiss():
    loc_fn = lambda pars: onp.array([pars[0] * 2.0])
    mm = MainModel(loc=loc_fn, pdf_type="poiss")
    pars = onp.array([3.0])
    assert onp.allclose(mm.expected_data(pars), onp.array([6.0]))
    s = mm.sample(pars, sample_size=5)
    assert s.shape[0] == 5
    # returned samples length corresponds to loc length
    assert s.shape[1] == 1


def test_constraint_model_expected_and_logprob_with_domains():
    # Normal uses first parameter, multivariate normal uses next two parameters
    descs = [
        {
            "distribution_type": "normal",
            "args": [onp.array([0.0]), onp.array([1.0])],
            "kwargs": {"domain": slice(0, 1)},
        },
        {
            "distribution_type": "multivariatenormal",
            "args": [onp.array([0.0, 0.0]), onp.eye(2)],
            "kwargs": {"domain": slice(1, 3)},
        },
    ]
    cm = ConstraintModel(descs)
    pars = onp.array([0.0, 0.0, 0.0])
    exp = cm.expected_data()
    # expect [0.0] hstack [0.0, 0.0]
    assert onp.allclose(exp, onp.array([0.0, 0.0, 0.0]))

    lp = float(cm.log_prob(pars))
    # normal part: -0.5*ln(2π) ; multivariate (2D identity at zero): -ln(2π)
    expected = -0.5 * math.log(2.0 * math.pi) + (-math.log(2.0 * math.pi))
    assert pytest.approx(expected, rel=1e-7) == lp


def test_mixture_model_log_prob_sum():
    class FakeDist:
        def __init__(self, val):
            self._val = val

        def log_prob(self, value):
            return onp.array(self._val)

    d1 = FakeDist(-1.0)
    d2 = FakeDist(-2.0)
    mix = MixtureModel(d1, d2)
    got = float(mix.log_prob(onp.array([0.0])))
    assert pytest.approx(-3.0, rel=1e-12) == got
