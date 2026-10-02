"""Test test statistics"""

import numpy as np
import pytest
from spey.hypothesis_testing.test_statistics import (
    ASIMOV_TESTSTAT_TOLERANCE,
    get_test_statistic,
    compute_teststatistics,
)
from spey.system.exceptions import AsimovTestStatZero, UnknownTestStatistics


def test_test_statistic():
    """Validate test statistic functions"""

    logpdf = lambda x: -10.0

    q0 = get_test_statistic("q0")

    assert 0.0 == q0(
        np.random.rand(), -1.0, np.random.rand(), logpdf
    ), "q0 should return zero for negative muhat"
    assert 16.0 == q0(np.random.rand(), 1.0, -2.0, logpdf), "q0 returns wrong value."

    qmu = get_test_statistic("qmu")
    assert 0.0 == qmu(1, 2, -10, logpdf), "qmu should return zero if muhat > mu"
    assert 16.0 == qmu(3, 2, -2, logpdf), "qmu returns wrong value."

    qmu_tilde = get_test_statistic("qmutilde")

    assert 0.0 == qmu_tilde(
        1, 2, -10, logpdf
    ), "qmu_tilde should return zero if muhat > mu"
    assert 16.0 == qmu_tilde(3, 2, -2, logpdf), "qmu_tilde returns wrong value"

    with pytest.raises(
        UnknownTestStatistics, match="Requested test statistics bla does not exist."
    ):
        _ = get_test_statistic("bla")


def test_computation():
    """test compute_teststatistics function"""

    logpdf = lambda x: -10.0

    mu = 3.0
    muhat = 2.0
    muhatA = 2.0
    min_nll = 2.0
    min_nllA = 1.0

    logpdf = lambda x: -10.0
    asimov_logpdf = lambda x: -8

    sqrt_qmu, sqrt_qmuA, delta_teststat = compute_teststatistics(
        mu=mu,
        maximum_likelihood=(muhat, min_nll),
        logpdf=logpdf,
        maximum_asimov_likelihood=(muhatA, min_nllA),
        asimov_logpdf=asimov_logpdf,
        teststat="qtilde",
    )

    assert sqrt_qmu == 4.0, "computation of sqrt(mu) is wrong"
    assert sqrt_qmuA == 3.7416573867739413, "computation of sqrt(muA) is wrong"
    assert delta_teststat == 0.2672612419124244, "computation of delta_teststat is wrong"

    asimov_logpdf = lambda x: -12

    sqrt_qmu, sqrt_qmuA, delta_teststat = compute_teststatistics(
        mu=mu,
        maximum_likelihood=(muhat, min_nll),
        logpdf=logpdf,
        maximum_asimov_likelihood=(muhatA, min_nllA),
        asimov_logpdf=asimov_logpdf,
        teststat="qtilde",
    )

    assert sqrt_qmu == 4.0, "computation of sqrt(mu) is wrong"
    assert sqrt_qmuA == 4.69041575982343, "computation of sqrt(muA) is wrong"
    assert delta_teststat == -0.6904157598234297, "computation of delta_teststat is wrong"


# ---------------------------------------------------------------------------
# Asimov test statistic tolerance (regression: exact float equality with zero)
# ---------------------------------------------------------------------------
def _asimov_case(sigma_asimov):
    r"""Gaussian observed likelihood plus an Asimov one of the requested width.

    A large ``sigma_asimov`` means the Asimov dataset barely constrains :math:`\mu`,
    which drives :math:`\sqrt{q_{\mu,A}}` towards zero.
    """
    logpdf = lambda mu: -0.5 * float(mu) ** 2
    asimov_logpdf = lambda mu: -0.5 * (float(mu) / sigma_asimov) ** 2
    return (0.0, 0.0), logpdf, (0.0, 0.0), asimov_logpdf


def test_near_zero_asimov_teststat_raises():
    """A tiny but non-zero sqrt(q_muA) must not be divided by.

    Eq. (66) of arXiv:1007.1727 divides by ``2 * sqrt(q_muA)``. The check used to be
    ``sqrt_qmuA == 0``, which never fired for a merely tiny value, so the p-value came
    back as numerical noise (typically a spurious, near-certain exclusion).
    """
    maximum_likelihood, logpdf, maximum_asimov, asimov_logpdf = _asimov_case(1e8)
    with pytest.raises(AsimovTestStatZero, match="numerically zero"):
        compute_teststatistics(
            1.0, maximum_likelihood, logpdf, maximum_asimov, asimov_logpdf, "qtilde"
        )


def test_exactly_zero_asimov_teststat_still_raises():
    """The original exact-zero behaviour is preserved.

    A perfectly flat Asimov likelihood gives ``q_muA = 0`` exactly while the observed
    statistic stays positive, which is the case the old ``== 0`` check caught.
    """
    logpdf = lambda mu: -0.5 * float(mu) ** 2
    flat_asimov_logpdf = lambda mu: 0.0
    with pytest.raises(AsimovTestStatZero):
        compute_teststatistics(
            1.0, (0.0, 0.0), logpdf, (0.0, 0.0), flat_asimov_logpdf, "qtilde"
        )


def test_tolerance_is_configurable():
    """Setting the tolerance to zero restores the unchecked division."""
    maximum_likelihood, logpdf, maximum_asimov, asimov_logpdf = _asimov_case(1e8)
    _, sqrt_qmuA, delta_teststat = compute_teststatistics(
        1.0,
        maximum_likelihood,
        logpdf,
        maximum_asimov,
        asimov_logpdf,
        "qtilde",
        asimov_teststat_tolerance=0.0,
    )
    assert sqrt_qmuA < ASIMOV_TESTSTAT_TOLERANCE
    assert delta_teststat > 1e6  # the near-zero divisor inflates the statistic


def test_healthy_asimov_teststat_is_untouched():
    """A well-behaved model must not be affected by the tolerance."""
    maximum_likelihood, logpdf, maximum_asimov, asimov_logpdf = _asimov_case(1.0)
    sqrt_qmu, sqrt_qmuA, delta_teststat = compute_teststatistics(
        2.0, maximum_likelihood, logpdf, maximum_asimov, asimov_logpdf, "qtilde"
    )
    assert sqrt_qmuA == pytest.approx(2.0)
    assert sqrt_qmu == pytest.approx(2.0)
    assert delta_teststat == pytest.approx(0.0)


def test_tolerance_only_applies_to_qtilde_division_branch():
    """``q`` and ``q0`` take the subtraction branch and never divide."""
    maximum_likelihood, logpdf, maximum_asimov, asimov_logpdf = _asimov_case(1e8)
    for teststat in ("q", "q0"):
        _, sqrt_qmuA, delta_teststat = compute_teststatistics(
            1.0, maximum_likelihood, logpdf, maximum_asimov, asimov_logpdf, teststat
        )
        assert np.isfinite(delta_teststat)
        assert sqrt_qmuA < ASIMOV_TESTSTAT_TOLERANCE
