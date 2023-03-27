"""Tools for computing confidence level and pvalues"""

from typing import Tuple, Text, List
import warnings
import numpy as np

from spey.hypothesis_testing.asymptotic_calculator import AsymptoticTestStatisticsDistribution

__all__ = ["compute_confidence_level", "pvalues", "expected_pvalues"]


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
    r"""
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
    r"""
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
