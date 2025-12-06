import numpy as onp

from spey.backends.default_pdf.third_moment import (
    compute_third_moments,
    third_moment_expansion,
)


def test_compute_third_moments_symmetric_returns_zero():
    uppers = onp.array([1.0, 2.0])
    lowers = uppers.copy()
    m3 = compute_third_moments(uppers, lowers)
    # symmetric (upper == lower) -> third moments should be ~0
    assert m3.shape == (2,)
    assert onp.allclose(m3, onp.zeros_like(m3), atol=1e-10)


def test_compute_third_moments_returns_errors_when_requested():
    uppers = onp.array([1.0, 2.0])
    lowers = onp.array([0.5, 1.5])
    m3, err = compute_third_moments(uppers, lowers, return_integration_error=True)
    assert m3.shape == (2,)
    assert err.shape == (2,)
    # values should be finite and errors non-negative
    assert onp.all(onp.isfinite(m3))
    assert onp.all(err >= 0.0)


def test_third_moment_expansion_with_zero_third_moment():
    expectation = onp.array([10.0, 20.0])
    cov = onp.diag(onp.array([4.0, 9.0]))  # variances 4 and 9
    third_mom = onp.array([0.0, 0.0])

    A, B, C = third_moment_expansion(
        expectation, cov, third_mom, return_correlation_matrix=False
    )
    # With zero third moment the implementation sets C -> 0, B -> sqrt(var), A -> expectation
    assert onp.allclose(onp.asarray(A), expectation, atol=1e-12)
    assert onp.allclose(onp.asarray(B), onp.sqrt(onp.diag(cov)), atol=1e-12)
    assert onp.allclose(onp.asarray(C), onp.zeros_like(C), atol=1e-12)


def test_third_moment_expansion_returns_correlation_matrix_and_is_symmetric():
    expectation = onp.array([1.0, 2.0])
    cov = onp.array([[4.0, 1.0], [1.0, 9.0]])
    # choose small third moments that satisfy the condition in most realistic cases
    third_mom = onp.array([0.5, 0.5])

    A, B, C, corr = third_moment_expansion(
        expectation, cov, third_mom, return_correlation_matrix=True
    )
    corr = onp.asarray(corr)
    # correlation matrix should be square, symmetric and finite
    assert corr.shape == (2, 2)
    assert onp.allclose(corr, corr.T, atol=1e-12)
    assert onp.all(onp.isfinite(corr))
