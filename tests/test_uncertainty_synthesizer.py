# python
import numpy as np
import pytest

# relative imports as this test file lives in the same package
from spey.backends.default_pdf.uncertainty_synthesizer import (
    signal_uncertainty_synthesizer,
    constraint_from_corr,
)
from spey.backends.default_pdf import uncertainty_synthesizer as us


def test_constraint_from_corr_with_matrix_and_none():
    corr = [[1.0, 0.5], [0.5, 1.0]]
    dom = slice(0, 2)

    # with provided correlation matrix
    term = constraint_from_corr(corr, 2, dom)
    assert isinstance(term, list) and len(term) == 1
    c = term[0]
    assert c["distribution_type"] == "multivariatenormal"
    args = c["args"]
    assert np.allclose(np.asarray(args[0]), np.zeros(2))
    assert np.allclose(np.asarray(args[1]), np.array(corr))
    assert c["kwargs"]["domain"] == dom

    # without correlation matrix -> normal with ones
    term2 = constraint_from_corr(None, 3, slice(0, 3))
    assert isinstance(term2, list) and len(term2) == 1
    c2 = term2[0]
    assert c2["distribution_type"] == "normal"
    a0, a1 = c2["args"]
    assert np.allclose(np.asarray(a0), np.zeros(3))
    assert np.allclose(np.asarray(a1), np.ones(3))
    assert c2["kwargs"]["domain"] == slice(0, 3)


def test_signal_uncertainty_absolute_branch():
    signal_yields = [10.0, 20.0]
    absolute_uncertainties = [1.5, 2.0]
    domain = slice(0, 2)

    res = signal_uncertainty_synthesizer(
        signal_yields,
        absolute_uncertainties=absolute_uncertainties,
        correlation_matrix=None,
        domain=domain,
    )
    lam = res["lambda"]
    constraint = res["constraint"]
    # check constraint type and shape
    assert isinstance(constraint, list) and len(constraint) == 1
    assert constraint[0]["distribution_type"] == "normal"
    # prepare a parameter vector (longer than domain to ensure slicing works)
    pars = np.array([0.1, 0.2, 5.0])
    expected = np.array(absolute_uncertainties) * pars[domain]
    got = np.asarray(lam(pars))
    assert np.allclose(got, expected)


def test_signal_uncertainty_envelops_branch():
    signal_yields = [5.0]
    # asymmetric envelopes: (upper, lower)
    env = [(2.0, -1.0)]
    domain = slice(0, 1)

    res = signal_uncertainty_synthesizer(
        signal_yields,
        absolute_uncertainty_envelops=env,
        correlation_matrix=None,
        domain=domain,
    )
    lam = res["lambda"]
    pars = np.array([6.0])  # choose pars so formula is straightforward
    # compute expected effective sigma manually:
    sigma_plus = abs(env[0][0])
    sigma_minus = abs(env[0][1])
    effective = np.sqrt(
        sigma_plus * sigma_minus
        + (sigma_plus - sigma_minus) * (pars[domain] - np.array(signal_yields))
    )
    expected = np.asarray(effective) * pars[domain]
    got = np.asarray(lam(pars))
    assert np.allclose(got, expected)


def test_signal_uncertainty_third_moment_branch(monkeypatch):
    # Prepare inputs
    signal_yields = [1.0, 2.0]
    absolute_uncertainties = [0.1, 0.2]
    correlation_matrix = [[1.0, 0.0], [0.0, 1.0]]
    third_moments = [0.0, 0.0]
    domain = slice(0, 2)

    # Monkeypatch correlation_to_covariance to return a deterministic covariance
    def fake_corr_to_cov(corr_mat, abs_unc):
        # a simple covariance: diag(abs_unc^2) for test purposes
        arr = np.array(abs_unc)
        return np.diag(arr * arr)

    # Monkeypatch third_moment_expansion to return simple A, B, C, corr
    def fake_third_moment_expansion(signal_yields_in, cov, third_moments_in, flag):
        # Return A, B, C such that lam = A + B * pars + C * pars^2
        A = np.array([0.5, 1.0])
        B = np.array([2.0, -1.0])
        C = np.array([0.0, 0.0])
        corr = np.eye(len(signal_yields_in))
        return A, B, C, corr

    monkeypatch.setattr(us, "correlation_to_covariance", fake_corr_to_cov)
    monkeypatch.setattr(us, "third_moment_expansion", fake_third_moment_expansion)

    res = signal_uncertainty_synthesizer(
        signal_yields,
        absolute_uncertainties=absolute_uncertainties,
        correlation_matrix=correlation_matrix,
        third_moments=third_moments,
        domain=domain,
    )
    lam = res["lambda"]
    pars = np.array([0.2, -0.5, 10.0])  # extra entry to ensure domain slicing
    got = np.asarray(lam(pars))
    # expected using fake A and B
    A = np.array([0.5, 1.0])
    B = np.array([2.0, -1.0])
    expected = A + B * pars[domain]
    assert np.allclose(got, expected)


def test_signal_uncertainty_inconsistent_raises():
    # only signal_yields provided (and domain) -> inconsistent -> ValueError
    with pytest.raises((AssertionError, ValueError)):
        # domain must not be None per function; provide a domain but no other valid combination
        signal_uncertainty_synthesizer([1.0, 2.0], domain=slice(0, 2))
