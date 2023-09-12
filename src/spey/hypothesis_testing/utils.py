"""
Routines for computing pvalues

Similar implementation can be found in https://github.com/scikit-hep/pyhf/blob/main/src/pyhf/infer/calculators.py
"""

import warnings
from typing import List, Tuple, Union

import numpy as np

from .distributions import (
    AsymptoticTestStatisticsDistribution,
    EmpricTestStatisticsDistribution,
)

__all__ = ["pvalues", "expected_pvalues"]


def __dir__():
    return __all__


def pvalues(
    delta_test_statistic: float,
    sig_plus_bkg_distribution: Union[
        AsymptoticTestStatisticsDistribution, EmpricTestStatisticsDistribution
    ],
    bkg_only_distribution: Union[
        AsymptoticTestStatisticsDistribution, EmpricTestStatisticsDistribution
    ],
) -> Tuple[float, float, float]:
    r"""
    Calculate the p-values for the observed test statistic under the
    signal + background and background-only model hypotheses.

    Args:
        delta_test_statistic (``float``): :math:`\Delta{\sqrt{q_{\mu}},\sqrt{q_{\mu,A}}}`
        sig_plus_bkg_distribution (~spey.hypothesis_testing.asymptotic_calculator.AsymptoticTestStatisticsDistribution):
          the distribution for the signal + background hypothesis.
        bkg_only_distribution (~spey.hypothesis_testing.asymptotic_calculator.AsymptoticTestStatisticsDistribution):
          The distribution for the background-only hypothesis.

    Returns:
        ``Tuple[float, float, float]``:
        The p-values for the test statistic corresponding to the :math:`CL_{s+b}`,
        :math:`CL_{b}`, and :math:`CL_{s}`.
    """
    CLsb = sig_plus_bkg_distribution.pvalue(delta_test_statistic)
    CLb = bkg_only_distribution.pvalue(delta_test_statistic)
    with warnings.catch_warnings(record=True):
        CLs = np.true_divide(CLsb, CLb, dtype=np.float64)
    return CLsb, CLb, CLs if CLb != 0.0 else 0.0


def expected_pvalues(
    sig_plus_bkg_distribution: Union[
        AsymptoticTestStatisticsDistribution, EmpricTestStatisticsDistribution
    ],
    bkg_only_distribution: Union[
        AsymptoticTestStatisticsDistribution, EmpricTestStatisticsDistribution
    ],
) -> List[List]:
    r"""
    Calculate the :math:`CL_s` values corresponding to the
    median significance of variations of the signal strength from the
    background only hypothesis :math:`\mu=0` at :math:`(-2,-1,0,1,2)\sigma`.

    Args:
        sig_plus_bkg_distribution (~spey.hypothesis_testing.asymptotic_calculator.AsymptoticTestStatisticsDistribution):
          The distribution for the signal + background hypothesis.
        bkg_only_distribution (~spey.hypothesis_testing.asymptotic_calculator.AsymptoticTestStatisticsDistribution):
          The distribution for the background-only hypothesis.

    Returns:
        ``List[List]``:
        The p-values for the test statistic corresponding to the :math:`CL_{s+b}`,
        :math:`CL_{b}`, and :math:`CL_{s}`.
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
