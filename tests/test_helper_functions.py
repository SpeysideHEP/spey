"""Tests for the shared helper functions."""

import numpy as np
import pytest

import spey
from spey.helper_functions import (
    correlation_to_covariance,
    covariance_to_correlation,
    ensure_positive_definite,
    symmetrise_matrix,
)
from spey.system.exceptions import InvalidInput

ASYMMETRIC = np.array([[144.0, 13.0], [25.0, 256.0]])
SYMMETRIC = np.array([[144.0, 19.0], [19.0, 256.0]])


def test_symmetrise_matrix_leaves_a_symmetric_matrix_alone():
    """A valid covariance matrix must pass through untouched."""
    result = symmetrise_matrix(SYMMETRIC)
    assert np.array_equal(result, SYMMETRIC)


def test_symmetrise_matrix_takes_the_symmetric_part():
    """The off-diagonal entries are averaged, the diagonal is untouched."""
    result = symmetrise_matrix(ASYMMETRIC)
    assert np.allclose(result, SYMMETRIC)
    assert np.allclose(result, result.T)
    # the symmetric part is the closest symmetric matrix in Frobenius norm
    for candidate in (13.0, 25.0, 0.0, 19.5):
        other = np.array([[144.0, candidate], [candidate, 256.0]])
        assert np.linalg.norm(ASYMMETRIC - result) <= np.linalg.norm(ASYMMETRIC - other)


def test_symmetrise_matrix_warns():
    """The user has to be told that their input was modified, but only then."""
    import io
    import logging

    from spey.system.logger import capture_logs

    # `log_once` is lru_cached, so the message may already have been consumed
    # by another test; clear it to keep this test independent of ordering.
    spey.log_once.cache_clear()

    stream = io.StringIO()
    with capture_logs(logging.WARNING, stream=stream):
        symmetrise_matrix(ASYMMETRIC, "covariance matrix")
    assert "not symmetric" in stream.getvalue()

    stream = io.StringIO()
    with capture_logs(logging.WARNING, stream=stream):
        symmetrise_matrix(SYMMETRIC, "covariance matrix")
    assert "not symmetric" not in stream.getvalue()


def test_symmetrise_matrix_rejects_a_non_square_matrix():
    """A non-square input is a genuine error, not something to repair."""
    with pytest.raises(InvalidInput, match="square matrix"):
        symmetrise_matrix(np.ones((2, 3)))
    with pytest.raises(InvalidInput, match="square matrix"):
        symmetrise_matrix(np.ones(3))


def test_covariance_to_correlation_is_symmetric():
    """The correlation matrix of an asymmetric input must still be symmetric."""
    correlation = covariance_to_correlation(ASYMMETRIC)
    assert np.allclose(correlation, correlation.T)
    assert np.allclose(correlation, covariance_to_correlation(SYMMETRIC))
    assert np.allclose(np.diag(correlation), [1.0, 1.0])


def test_correlation_to_covariance_is_symmetric():
    """Round-tripping through the two helpers is stable and symmetric."""
    covariance = correlation_to_covariance(
        np.array([[1.0, 0.07], [0.13, 1.0]]), np.array([12.0, 16.0])
    )
    assert np.allclose(covariance, covariance.T)
    assert np.allclose(covariance_to_correlation(covariance), [[1.0, 0.1], [0.1, 1.0]])


@pytest.mark.parametrize(
    "backend, key",
    [
        ("default.correlated_background", "covariance_matrix"),
        ("default.multivariate_normal", "covariance_matrix"),
    ],
)
def test_backends_symmetrise_the_covariance_matrix(backend, key):
    """An asymmetric input must give the same model as its symmetric part."""
    common = {
        "signal_yields": [12.0, 15.0],
        "background_yields": [50.0, 48.0],
        "data": [36, 33],
    }
    asymmetric = spey.get_backend(backend)(**common, **{key: ASYMMETRIC})
    symmetric = spey.get_backend(backend)(**common, **{key: SYMMETRIC})

    assert np.allclose(asymmetric.backend.covariance_matrix, SYMMETRIC)
    pars = np.array([1.0, 0.3, -0.4])
    assert asymmetric.backend.get_logpdf_func()(pars) == pytest.approx(
        symmetric.backend.get_logpdf_func()(pars)
    )
    assert asymmetric.exclusion_confidence_level()[0] == pytest.approx(
        symmetric.exclusion_confidence_level()[0]
    )


def test_effective_sigma_symmetrises_the_correlation_matrix():
    """``default.effective_sigma`` takes a correlation matrix directly."""
    common = {
        "signal_yields": [12.0, 15.0],
        "background_yields": [50.0, 48.0],
        "data": [36, 33],
        "absolute_uncertainty_envelops": [(10.0, 15.0), (13.0, 18.0)],
    }
    asymmetric = spey.get_backend("default.effective_sigma")(
        **common, correlation_matrix=[[1.0, 0.07], [0.13, 1.0]]
    )
    symmetric = spey.get_backend("default.effective_sigma")(
        **common, correlation_matrix=[[1.0, 0.1], [0.1, 1.0]]
    )
    pars = np.array([1.0, 0.3, -0.4])
    assert asymmetric.backend.get_logpdf_func()(pars) == pytest.approx(
        symmetric.backend.get_logpdf_func()(pars)
    )


NOT_POSITIVE_DEFINITE = np.array([[1.0, 2.0], [2.0, 1.0]])
SINGULAR = np.array([[1.0, 1.0], [1.0, 1.0]])


def test_ensure_positive_definite_accepts_a_valid_matrix():
    """A genuine covariance matrix passes through unchanged."""
    result = ensure_positive_definite(SYMMETRIC)
    assert np.array_equal(result, SYMMETRIC)


def test_ensure_positive_definite_symmetrises_first():
    """Positive definiteness is only meaningful for a symmetric matrix."""
    assert np.allclose(ensure_positive_definite(ASYMMETRIC), SYMMETRIC)


def test_ensure_positive_definite_rejects_a_negative_eigenvalue():
    """An indefinite matrix makes the normal distribution unbounded."""
    with pytest.raises(InvalidInput, match="negative eigenvalue"):
        ensure_positive_definite(NOT_POSITIVE_DEFINITE)


def test_ensure_positive_definite_rejects_a_singular_matrix():
    """A rank-deficient matrix has no inverse, so the density does not exist."""
    with pytest.raises(InvalidInput, match="singular"):
        ensure_positive_definite(SINGULAR)
    # the error has to point at the way out
    with pytest.raises(InvalidInput, match="merge_correlated_bins"):
        ensure_positive_definite(SINGULAR)


def _with_eigenvalues(largest, smallest):
    """Build a symmetric 2x2 matrix with the requested eigenvalues."""
    rotation = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2.0)
    return rotation @ np.diag([largest, smallest]) @ rotation.T


def test_ensure_positive_definite_accepts_an_ill_conditioned_matrix():
    """
    A positive definite matrix is accepted however badly conditioned it is.

    The criterion is whether the Cholesky decomposition exists, i.e. whether the
    inverse and determinant the likelihood needs can be computed at all. The
    condition number is deliberately not policed.
    """
    matrix = _with_eigenvalues(200.0, 1e-12)
    assert ensure_positive_definite(matrix) is not None


def test_ensure_positive_definite_reports_a_near_singular_matrix():
    """A matrix that is only negative at round-off level is still unusable."""
    with pytest.raises(InvalidInput, match="singular"):
        ensure_positive_definite(_with_eigenvalues(200.0, -1e-9))


def test_ensure_positive_definite_reports_a_clearly_negative_eigenvalue():
    """A genuinely indefinite matrix is named as such."""
    with pytest.raises(InvalidInput, match="negative eigenvalue"):
        ensure_positive_definite(_with_eigenvalues(200.0, -1e-3))


def test_ensure_positive_definite_rejects_a_non_square_matrix():
    """Squareness is checked before anything else."""
    with pytest.raises(InvalidInput, match="square matrix"):
        ensure_positive_definite(np.ones((2, 3)))


@pytest.mark.parametrize(
    "backend, key, matrix",
    [
        ("default.correlated_background", "covariance_matrix", NOT_POSITIVE_DEFINITE),
        ("default.correlated_background", "covariance_matrix", SINGULAR),
        ("default.multivariate_normal", "covariance_matrix", NOT_POSITIVE_DEFINITE),
        ("default.multivariate_normal", "covariance_matrix", SINGULAR),
    ],
)
def test_backends_reject_an_invalid_covariance_matrix(backend, key, matrix):
    """A model cannot be built on a matrix that defines no distribution."""
    with pytest.raises(InvalidInput, match="not positive definite"):
        spey.get_backend(backend)(
            signal_yields=[12.0, 15.0],
            background_yields=[50.0, 48.0],
            data=[36, 33],
            **{key: matrix},
        )


def test_effective_sigma_rejects_an_invalid_correlation_matrix():
    """``default.effective_sigma`` validates the correlation matrix it is given."""
    with pytest.raises(InvalidInput, match="not positive definite"):
        spey.get_backend("default.effective_sigma")(
            signal_yields=[12.0, 15.0],
            background_yields=[50.0, 48.0],
            data=[36, 33],
            correlation_matrix=[[1.0, 1.2], [1.2, 1.0]],
            absolute_uncertainty_envelops=[(10.0, 15.0), (13.0, 18.0)],
        )


def test_third_moment_expansion_validates_the_derived_correlation():
    """The matrix that enters the likelihood is the derived one, not the input."""
    model = spey.get_backend("default.third_moment_expansion")(
        signal_yields=[12.0, 15.0],
        background_yields=[50.0, 48.0],
        data=[36, 33],
        covariance_matrix=SYMMETRIC,
        third_moment=[0.5, 0.8],
    )
    # the valid case still builds, and the derived matrix is positive definite
    correlation = model.backend.constraint_model._pdfs[0].cov(np.zeros(2))
    assert np.linalg.eigvalsh(correlation).min() > 0.0
