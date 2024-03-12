"""Functions for computation of test statistic"""

import warnings
from typing import Callable, Text, Tuple

import numpy as np

from spey.system.exceptions import UnknownTestStatistics

__all__ = ["qmu", "qmu_tilde", "q0", "get_test_statistic", "compute_teststatistics"]


def qmu_tilde(
    mu: float, muhat: float, max_logpdf: float, logpdf: Callable[[float], float]
) -> float:
    r"""
    Alternative test statistics, :math:`\tilde{q}_{\mu}`, see eq. (62) of :xref:`1007.1727`.

    .. math::

        \tilde{q}_{\mu} = \begin{cases}
            0 & \text{if}\ \hat{\mu} > \mu\ , \\
            -2\log\left( \frac{\mathcal{L}(\mu, \theta_\mu)}{\mathcal{L}(\max(\hat{\mu}, 0), \hat{\theta})} \right) & \text{otherwise}
        \end{cases}

    .. warning::

        Note that this assumes that :math:`\hat\mu\geq0`, hence :obj:`allow_negative_signal`
        assumed to be ``False``. If this function has been executed by user, :obj:`spey`
        assumes that this is taken care of throughout the external code consistently.
        Whilst computing p-values or upper limit on :math:`\mu` through :obj:`spey` this
        is taken care of automatically in the backend.

    Args:
        mu (``float``): parameter of interest, :math:`\mu`.
        muhat (``float``): :math:`\hat\mu` value that maximizes the likelihood.
        max_logpdf (``float``): maximum value of :math:`\log\mathcal{L}`.
        logpdf (``Callable[[float], float]``): :math:`\log\mathcal{L}(\mu, \theta_\mu)`.

    Returns:
        ``float``:
        the value of :math:`\tilde{q}_{\mu}`.
    """
    return (
        0.0
        if muhat > mu
        else np.clip(
            -2.0 * (logpdf(mu) - (max_logpdf if muhat >= 0.0 else logpdf(0.0))), 0.0, None
        )
    )


def qmu(
    mu: float, muhat: float, max_logpdf: float, logpdf: Callable[[float], float]
) -> float:
    r"""
    Test statistic :math:`q_{\mu}`, see eq. (54) of :xref:`1007.1727`
    
    .. math::

        q_{\mu} = \begin{cases}
            0 & \text{if}\ \hat{\mu} > \mu\ ,\\
            -2\log\left( \frac{\mathcal{L}(\mu, \theta_\mu)}{\mathcal{L}(\hat{\mu}, \hat{\theta})} \right) & \text{otherwise}
        \end{cases}

    Args:
        mu (``float``): parameter of interest, :math:`\mu`.
        muhat (``float``): :math:`\hat\mu` value that maximizes the likelihood.
        max_logpdf (``float``): maximum value of :math:`\log\mathcal{L}`.
        logpdf (``Callable[[float], float]``): :math:`\log\mathcal{L}(\mu, \theta_\mu)`.

    Returns:
        ``float``:
        the value of :math:`q_{\mu}`.
    """
    return 0.0 if muhat > mu else np.clip(-2.0 * (logpdf(mu) - max_logpdf), 0.0, None)


def q0(
    mu: float, muhat: float, max_logpdf: float, logpdf: Callable[[float], float]
) -> float:
    r"""
    Discovery test statistics, :math:`q_{0}` see eq. (47) of :xref:`1007.1727`.

    .. math::

        q_0 = \begin{cases}
            0 & \text{if}\ \hat{\mu} < 0\ ,\\
            -2\log\left( \frac{\mathcal{L}(0, \theta_0)}{\mathcal{L}(\hat{\mu}, \hat{\theta})} \right) & \text{otherwise}
        \end{cases}

    Args:
        mu (``float``): parameter of interest, :math:`\mu`.

          .. note::

            ``mu`` argument is overwritten by zero. This input is only for consistency with
            other test statistic functions.

        muhat (``float``): :math:`\hat\mu` value that maximizes the likelihood.
        max_logpdf (``float``): maximum value of :math:`\log\mathcal{L}`.
        logpdf (``Callable[[float], float]``): :math:`\log\mathcal{L}(\mu, \theta_\mu)`.

    Returns:
        ``float``:
        the value of :math:`q_{0}`.
    """
    mu = 0.0
    return 0.0 if muhat < 0.0 else np.clip(-2.0 * (logpdf(mu) - max_logpdf), 0.0, None)


def get_test_statistic(
    test_stat: Text,
) -> Callable[[float, float, float, Callable[[float], float]], float]:
    r"""
    Retreive the test statistic function

    Args:
        test_stat (``Text``): test statistic.

          * ``'qtilde'``: (default) performs the calculation using the alternative test statistic,
            :math:`\tilde{q}_{\mu}`, see eq. (62) of :xref:`1007.1727`
            (:func:`~spey.hypothesis_testing.test_statistics.qmu_tilde`).

            .. warning::

                Note that this assumes that :math:`\hat\mu\geq0`, hence :obj:`allow_negative_signal`
                assumed to be ``False``. If this function has been executed by user, :obj:`spey`
                assumes that this is taken care of throughout the external code consistently.
                Whilst computing p-values or upper limit on :math:`\mu` through :obj:`spey` this
                is taken care of automatically in the backend.

          * ``'q'``: performs the calculation using the test statistic :math:`q_{\mu}`, see
            eq. (54) of :xref:`1007.1727` (:func:`~spey.hypothesis_testing.test_statistics.qmu`).
          * ``'q0'``: performs the calculation using the discovery test statistic, see eq. (47)
            of :xref:`1007.1727` :math:`q_{0}` (:func:`~spey.hypothesis_testing.test_statistics.q0`).

    Raises:
        :obj:`~spey.system.exceptions.UnknownTestStatistics`: If the ``test_stat`` input does not match
          any of the above.

    Returns:
        ``Callable[[float, float, float, Callable[[float], float]], float]``:
        returns the function to compute test statistic
    """
    # syntax with mu is not necessary so accomodate both
    if test_stat in ["qmu", "q"]:
        test_stat = "q"
    elif test_stat in ["qtilde", "qmutilde"]:
        test_stat = "qmutilde"
    options = {"qmutilde": qmu_tilde, "q": qmu, "q0": q0}

    if options.get(test_stat, False) is False:
        raise UnknownTestStatistics(
            f"Requested test statistics {test_stat} does not exist."
        )

    return options[test_stat]


def compute_teststatistics(
    mu: float,
    maximum_likelihood: Tuple[float, float],
    logpdf: Callable[[float], float],
    maximum_asimov_likelihood: Tuple[float, float],
    asimov_logpdf: Callable[[float], float],
    teststat: Text,
) -> Tuple[float, float, float]:
    r"""
    Compute test statistics

    Args:
        mu (``float``): parameter of interest, :math:`\mu`.
        maximum_likelihood (``Tuple[float, float]``): (:math:`\hat\mu` and :math:`\arg\min\log\mathcal{L}`)
        logpdf (``Callable[[float], float]``): function to compute :math:`\log\mathcal{L}` with fixed :math:`\mu`.
        maximum_asimov_likelihood (``Tuple[float, float]``): (:math:`\hat\mu_A` and :math:`\arg\min\log\mathcal{L}_A`)
        asimov_logpdf (``Callable[[float], float]``): function to compute :math:`\log\mathcal{L}_A` with fixed :math:`\mu`.
        teststat (``Text``): test statistic.

          * ``'qtilde'``: (default) performs the calculation using the alternative test statistic,
            :math:`\tilde{q}_{\mu}`, see eq. (62) of :xref:`1007.1727`
            (:func:`~spey.hypothesis_testing.test_statistics.qmu_tilde`).

            .. warning::

                Note that this assumes that :math:`\hat\mu\geq0`, hence :obj:`allow_negative_signal`
                assumed to be ``False``. If this function has been executed by user, :obj:`spey`
                assumes that this is taken care of throughout the external code consistently.
                Whilst computing p-values or upper limit on :math:`\mu` through :obj:`spey` this
                is taken care of automatically in the backend.

          * ``'q'``: performs the calculation using the test statistic :math:`q_{\mu}`, see
            eq. (54) of :xref:`1007.1727` (:func:`~spey.hypothesis_testing.test_statistics.qmu`).
          * ``'q0'``: performs the calculation using the discovery test statistic, see eq. (47)
            of :xref:`1007.1727` :math:`q_{0}` (:func:`~spey.hypothesis_testing.test_statistics.q0`).

    Returns:
        ``Tuple[float, float, float]``:
        :math:`\sqrt{q_\mu}`, :math:`\sqrt{q_{\mu,A}}` and :math:`\Delta(\sqrt{q_\mu}, \sqrt{q_{\mu,A}})`
    """
    teststat_func = get_test_statistic(teststat)

    muhat, min_nll = maximum_likelihood
    muhatA, min_nllA = maximum_asimov_likelihood

    # max_logpdf = -min_nll
    qmu_ = teststat_func(mu, muhat, -min_nll, logpdf)
    qmuA = teststat_func(mu, muhatA, -min_nllA, asimov_logpdf)
    sqrt_qmu, sqrt_qmuA = np.sqrt(qmu_), np.sqrt(qmuA)

    if teststat in ["q", "q0", "qmu"]:
        delta_teststat = sqrt_qmu - sqrt_qmuA
    else:
        # arXiv:1007.1727 eq. 66
        if sqrt_qmu <= sqrt_qmuA:
            delta_teststat = sqrt_qmu - sqrt_qmuA
        else:
            with warnings.catch_warnings(record=True):
                delta_teststat = np.true_divide(qmu_ - qmuA, 2.0 * sqrt_qmuA)

    return sqrt_qmu, sqrt_qmuA, delta_teststat
