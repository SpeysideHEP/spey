"""Test test statistics"""

import numpy as np
import pytest
from spey.hypothesis_testing.test_statistics import (
    get_test_statistic,
    compute_teststatistics,
)
from spey.system.exceptions import UnknownTestStatistics


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
