import numpy as np
from typing import Callable, Text, Tuple
import warnings

from spey.system.exceptions import UnknownTestStatistics

__all__ = ["qmu", "qmu_tilde", "q0", "get_test_statistic", "compute_teststatistics"]


def _tmu_tilde(
    mu: float, muhat: float, min_logpdf: float, logpdf: Callable[[float], float]
) -> float:
    r"""
    The test statistic,`\tilde{t}_{\mu}`, for establishing a two-sided
    interval on the strength parameter,`\mu`, for models with
    bounded POI, as defined in Equation (11) in `arXiv:1007.1727`

    :param mu: Signal strength
    :param muhat: signal strength that minimizes logpdf
    :param min_logpdf: minimum value of logpdf
    :param logpdf: logpdf function which takes mu as an input
    :return: The calculated test statistic
    """
    return -2 * (logpdf(mu) - (min_logpdf if muhat >= 0.0 else logpdf(0.0)))


def _tmu(mu: float, min_logpdf: float, logpdf: Callable[[float], float]) -> float:
    r"""
    The test statistic,`t_{\mu}`, for establishing a two-sided
    interval on the strength parameter,`\mu`, as defined in Equation (8) in `arXiv:1007.1727`

    :param mu: Signal strength
    :param muhat: signal strength that minimizes logpdf
    :param min_logpdf: minimum value of logpdf
    :param logpdf: logpdf function which takes mu as an input
    :return: The calculated test statistic
    """
    return -2 * (logpdf(mu) - min_logpdf)


def qmu_tilde(
    mu: float, muhat: float, min_logpdf: float, logpdf: Callable[[float], float]
) -> float:
    r"""
    The "alternative" test statistic, `\tilde{q}_{\mu}`, for establishing
    an upper limit on the strength parameter,`\mu`, for models with
    bounded POI, as defined in Equation (16) in `arXiv:1007.1727`

    **qmu_tilde test statistic used for fit configuration with POI bounded at zero.**

    :param mu: Signal strength
    :param muhat: signal strength that minimizes logpdf
    :param min_logpdf: minimum value of logpdf
    :param logpdf: logpdf function which takes mu as an input
    :return: The calculated test statistic
    """
    return 0.0 if muhat > mu else _tmu_tilde(mu, muhat, min_logpdf, logpdf)


def qmu(mu: float, muhat: float, min_logpdf: float, logpdf: Callable[[float], float]) -> float:
    r"""
    The test statistic, `q_{\mu}`, for establishing an upper
    limit on the strength parameter, `\mu`, as defined in
    Equation (14) in arXiv:1007.1727

    **qmu test statistic used for fit configuration with POI not bounded at zero.**

    :param mu: Signal strength
    :param muhat: signal strength that minimizes logpdf
    :param min_logpdf: minimum value of logpdf
    :param logpdf: logpdf function which takes mu as an input
    :return: The calculated test statistic
    """
    return 0.0 if muhat > mu else _tmu(mu, min_logpdf, logpdf)


def q0(mu: float, muhat: float, min_logpdf: float, logpdf: Callable[[float], float]) -> float:
    r"""
    The test statistic,`q_{0}`, for discovery of a positive signal
    as defined in Equation (12) in `arXiv:1007.1727`, for `\mu=0`.

    :param mu: Signal strength (only for function consistency, its overwritten by zero)
    :param muhat: signal strength that minimizes logpdf
    :param min_logpdf: minimum value of logpdf
    :param logpdf: logpdf function which takes mu as an input
    :return: The calculated test statistic
    """
    return 0.0 if muhat < 0.0 else _tmu(0.0, min_logpdf, logpdf)


def get_test_statistic(test_stat: Text) -> Callable:
    """
    Retrieve test statistic function

    :raises UnknownTestStatistics: if input doesn't match any available function.
    """
    # syntax with mu is not necessary so accomodate both
    if test_stat in ["qmu", "q"]:
        test_stat = "q"
    elif test_stat in ["qtilde", "qmutilde"]:
        test_stat = "qmutilde"
    options = {"qmutilde": qmu_tilde, "q": qmu, "q0": q0}

    if options.get(test_stat, False) is False:
        raise UnknownTestStatistics(f"Requested test statistics {test_stat} does not exist.")

    return options[test_stat]


def compute_teststatistics(
    mu: float,
    maximum_likelihood: Tuple[float, float],
    logpdf: Callable[[float], float],
    maximum_asimov_likelihood: Tuple[float, float],
    asimov_logpdf: Callable[[float], float],
    teststat: Text,
) -> Tuple[float, float, float]:
    """
    Compute the test statistic for the observed data under the studied model.

    :param mu (`float`): Signal strength
    :param maximum_likelihood (`Tuple[float, float]`): muhat and minimum negative log-likelihood
    :param logpdf (`Callable[[float], float]`): log of the full density
    :param maximum_asimov_likelihood (`Tuple[float, float]`): muhat and minimum negative
                                                              log-likelihood for asimov data
    :param asimov_logpdf (`Callable[[float], float]`): log of the full density for asimov data
    :param teststat (`Text`): `"qmutilde"`, `"q"` or `"q0"`
    :return `Tuple[float, float, float]`: sqrt(qmu), sqrt(qmuA) and distance between them
    :raises `UnknownTestStatistics`: if input doesn't match any available function.
    """
    teststat_func = get_test_statistic(teststat)

    muhat, min_nll = maximum_likelihood
    muhatA, min_nllA = maximum_asimov_likelihood

    # min_logpdf = -min_nll
    qmu = teststat_func(mu, muhat, -min_nll, logpdf)
    qmuA = teststat_func(mu, muhatA, -min_nllA, asimov_logpdf)
    with warnings.catch_warnings(record=True):
        sqrt_qmu = np.sqrt(qmu)
        sqrt_qmuA = np.sqrt(qmuA)

        if teststat in ["q", "q0", "qmu"]:
            delta_teststat = sqrt_qmu - sqrt_qmuA
        else:
            delta_teststat = (
                sqrt_qmu - sqrt_qmuA
                if sqrt_qmu <= sqrt_qmuA
                else np.true_divide(qmu - qmuA, 2.0 * sqrt_qmuA)
            )

    return sqrt_qmu, sqrt_qmuA, delta_teststat
