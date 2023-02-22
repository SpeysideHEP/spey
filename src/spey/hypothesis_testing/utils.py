import numpy as np
import warnings, scipy
from functools import partial

from typing import Callable, Tuple, Text, List, Union

from spey.hypothesis_testing.asymptotic_calculator import AsymptoticTestStatisticsDistribution
from spey.hypothesis_testing.test_statistics import compute_teststatistics
from spey.utils import ExpectationType

__all__ = [
    "compute_confidence_level",
    "find_root_limits",
    "pvalues",
    "expected_pvalues",
    "find_poi_upper_limit",
]


def compute_confidence_level(
    sqrt_qmuA: float, delta_test_statistic: float, test_stat: Text = "qtilde"
) -> Tuple[List[float], List[float]]:
    """
    Compute confidence level

    :param sqrt_qmuA: The calculated test statistic for asimov data
    :param delta_test_statistic: test statistics
    :param expected: observed, apriori or aposteriori
    :param test_stat: type of test statistics, qtilde, or q0
    :return: confidence limit
    """
    sig_plus_bkg_distribution = AsymptoticTestStatisticsDistribution(-sqrt_qmuA, -np.inf)
    bkg_only_distribution = AsymptoticTestStatisticsDistribution(0.0, -np.inf)

    CLsb_obs, CLb_obs, CLs_obs = pvalues(
        delta_test_statistic, sig_plus_bkg_distribution, bkg_only_distribution
    )
    CLsb_exp, CLb_exp, CLs_exp = expected_pvalues(sig_plus_bkg_distribution, bkg_only_distribution)

    return ([CLsb_obs], CLsb_exp) if test_stat == "q0" else ([CLs_obs], CLs_exp)


def pvalues(
    delta_test_statistic: float,
    sig_plus_bkg_distribution: AsymptoticTestStatisticsDistribution,
    bkg_only_distribution: AsymptoticTestStatisticsDistribution,
) -> Tuple[float, float, float]:
    """
    Calculate the :math:`p`-values for the observed test statistic under the
    signal + background and background-only model hypotheses.

    :param delta_test_statistic: The test statistic.
    :param sig_plus_bkg_distribution: The distribution for the signal + background hypothesis.
    :param bkg_only_distribution: The distribution for the background-only hypothesis.
    :return: The p-values for the test statistic corresponding to the `\mathrm{CL}_{s+b}`,
            `\mathrm{CL}_{b}`, and `\mathrm{CL}_{s}`.
    """
    CLsb = sig_plus_bkg_distribution.pvalue(delta_test_statistic)
    CLb = bkg_only_distribution.pvalue(delta_test_statistic)
    with warnings.catch_warnings(record=True):
        CLs = np.true_divide(CLsb, CLb, dtype=np.float32)
    return CLsb, CLb, CLs if CLb != 0.0 else 0.0


def expected_pvalues(
    sig_plus_bkg_distribution: AsymptoticTestStatisticsDistribution,
    bkg_only_distribution: AsymptoticTestStatisticsDistribution,
) -> List[List]:
    """
    Calculate the `\mathrm{CL}_{s}` values corresponding to the
    median significance of variations of the signal strength from the
    background only hypothesis `\left(\mu=0\right)` at
    `(-2,-1,0,1,2)\sigma`.

    :param sig_plus_bkg_distribution: The distribution for the signal + background hypothesis.
    :param bkg_only_distribution: The distribution for the background-only hypothesis.
    :return: The p-values for the test statistic corresponding to the `\mathrm{CL}_{s+b}`,
            `\mathrm{CL}_{b}`, and `\mathrm{CL}_{s}`.
    """
    return list(
        map(
            list,
            zip(
                *[
                    pvalues(
                        bkg_only_distribution.expected_value(nsigma),
                        sig_plus_bkg_distribution,
                        bkg_only_distribution,
                    )
                    for nsigma in [2.0, 1.0, 0.0, -1.0, -2.0]
                ]
            ),
        )
    )


def find_root_limits(
    computer: Callable[[float], float], loc: float = 0.0, low_ini: float = 1.0, hig_ini: float = 1.0
) -> Tuple[float, float]:
    """
    Find limits for brent bracketing

    :param hig_ini:
    :param low_ini:
    :param computer: POI dependent function
    :param loc: location of the root
    :return: lower and upper bound
    """
    assert callable(computer), "Invalid input. Computer must be callable."

    low, hig = low_ini, hig_ini
    while computer(low) > loc:
        low *= 0.5
        if low < 1e-10:
            break
    while computer(hig) < loc:
        hig *= 2.0
        if hig > 1e10:
            break
    return low, hig


def find_poi_upper_limit(
    maximum_likelihood: Tuple[float, float],
    logpdf: Callable[[float], float],
    maximum_asimov_likelihood: Tuple[float, float],
    asimov_logpdf: Callable[[float], float],
    expected: ExpectationType,
    confidence_level: float = 0.95,
    allow_negative_signal: bool = True,
    low_init: float = 1.0,
    hig_init: float = 1.0,
    expected_pvalue: Text = "nominal",
) -> Union[float, List[float]]:
    """
    Compute the upper limit on parameter of interest, described by the confidence level

    :param maximum_likelihood (`Tuple[float, float]`): muhat and minimum negative log-likelihood
    :param logpdf (`Callable[[float], float]`): log of the full density
    :param maximum_asimov_likelihood (`Tuple[float, float]`): muhat and minimum negative
                                                              log-likelihood for asimov data
    :param asimov_logpdf (`Callable[[float], float]`): log of the full density for asimov data
    :param expected (`ExpectationType`): observed, apriori or aposteriori
    :param confidence_level (`float`, default `0.95`): exclusion confidence level (default 1 - CLs = 95%).
    :param allow_negative_signal (`bool`, default `True`): allow negative signals while
                                                           minimising negative log-likelihood.
    :param low_init (`float`, default `1.0`): initialized lower bound for bracketing.
    :param hig_init (`float`, default `1.0`): initialised upper bound for bracketing.
    :param expected_pvalue (`Text`, default `"nominal"`): find the upper limit for pvalue range,
                                                    only for expected. `nominal`, `1sigma`, `2sigma`
    :return `float`: excluded parameter of interest
    """
    assert expected_pvalue in [
        "nominal",
        "1sigma",
        "2sigma",
    ], f"Unknown pvalue range {expected_pvalue}"
    if expected is ExpectationType.observed:
        expected_pvalue = "nominal"
    test_stat = "q" if allow_negative_signal else "qtilde"

    def computer(poi_test: float, pvalue_idx: int) -> float:
        """Compute 1 - CLs(POI) = `confidence_level`"""
        _, sqrt_qmuA, delta_teststat = compute_teststatistics(
            poi_test,
            maximum_likelihood,
            logpdf,
            maximum_asimov_likelihood,
            asimov_logpdf,
            test_stat,
        )
        pvalue = list(
            map(
                lambda x: 1.0 - x,
                compute_confidence_level(sqrt_qmuA, delta_teststat, test_stat)[
                    0 if expected == ExpectationType.observed else 1
                ],
            )
        )
        # always get the median
        return pvalue[pvalue_idx] - confidence_level

    result = []
    index_range = {
        "nominal": [0 if expected is ExpectationType.observed else 2],
        "1sigma": range(1, 4),
        "2sigma": range(0, 5),
    }
    for pvalue_idx in index_range[expected_pvalue]:
        comp = partial(computer, pvalue_idx=pvalue_idx)
        low, hig = find_root_limits(comp, loc=0.0, low_ini=low_init, hig_ini=hig_init)
        with warnings.catch_warnings(record=True):
            x0, r = scipy.optimize.toms748(
                comp, low, hig, k=2, xtol=2e-12, rtol=1e-4, full_output=True
            )
        if not r.converged:
            warnings.warn("Optimiser did not converge.", category=RuntimeWarning)
        result.append(x0)
    return result if len(result) > 1 else result[0]
