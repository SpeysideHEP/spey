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

    .. math::

        p_{s+b}&=& \int_{-\infty}^{-\sqrt{q_{\mu,A}} - \Delta q_\mu} \mathcal{N}(x| 0, 1) dx \\
        p_{b}&=& \int_{-\infty}^{-\Delta q_\mu} \mathcal{N}(x| 0, 1) dx \\
        p_{s} &=& p_{s+b}/ p_{b}

    where :math:`q_\mu` stands for the test statistic and A stands for Assimov.

    Args:
        delta_test_statistic (``float``): :math:`\Delta q_\mu`
        sig_plus_bkg_distribution (~spey.hypothesis_testing.asymptotic_calculator.AsymptoticTestStatisticsDistribution):
          the distribution for the signal + background hypothesis.
        bkg_only_distribution (~spey.hypothesis_testing.asymptotic_calculator.AsymptoticTestStatisticsDistribution):
          The distribution for the background-only hypothesis.

    Returns:
        ``Tuple[float, float, float]``:
        The p-values for the test statistic corresponding to the :math:`p_{s+b}`,
        :math:`p_{b}`, and :math:`p_{s}`.

    .. seealso::

        :func:`~spey.hypothesis_testing.test_statistics.compute_teststatistics`,
        :func:`~spey.hypothesis_testing.asymptotic_calculator.compute_asymptotic_confidence_level`,
        :func:`~spey.hypothesis_testing.utils.expected_pvalues`
    """
    # delta_test_statistic is sqrt_qmu - sqrt_qmuA
    # p_sb is CDF(-sqrt_qmuA - delta_test_statistic) = CDF(-sqrt_qmuA - (sqrt_qmu - sqrt_qmuA))
    # CDF(-sqrt_qmu) = 1-CDF(sqrt_qmu)
    # See eq 58-59 in https://arxiv.org/pdf/1007.1727.pdf
    p_sb = sig_plus_bkg_distribution.pvalue(delta_test_statistic)
    # p_b is CDF(delta_test_statistic) = CDF(-(sqrt_qmu - sqrt_qmuA))
    p_b = bkg_only_distribution.pvalue(delta_test_statistic)
    with warnings.catch_warnings(record=True):
        p_s = np.true_divide(p_sb, p_b, dtype=np.float64)
    return p_sb, p_b, p_s if p_b != 0.0 else 0.0


def expected_pvalues(
    sig_plus_bkg_distribution: Union[
        AsymptoticTestStatisticsDistribution, EmpricTestStatisticsDistribution
    ],
    bkg_only_distribution: Union[
        AsymptoticTestStatisticsDistribution, EmpricTestStatisticsDistribution
    ],
) -> List[List]:
    r"""
    Calculate the :math:`p` values corresponding to the
    median significance of variations of the signal strength from the
    background only hypothesis :math:`\mu=0` at :math:`(-2,-1,0,1,2)\sigma`.

    .. math::

        p_{s+b}&=& \int_{-\infty}^{-\sqrt{q_{\mu,A}} - N\sigma} \mathcal{N}(x| 0, 1) dx \\
        p_{b}&=& \int_{-\infty}^{-N\sigma} \mathcal{N}(x| 0, 1) dx \\
        p_{s} &=& p_{s+b}/ p_{b}

    where :math:`q_\mu` stands for the test statistic and A stands for Assimov.
    :math:`N\sigma\in[-2,-1,0,1,2]`.

    Args:
        sig_plus_bkg_distribution (~spey.hypothesis_testing.asymptotic_calculator.AsymptoticTestStatisticsDistribution):
          The distribution for the signal + background hypothesis.
        bkg_only_distribution (~spey.hypothesis_testing.asymptotic_calculator.AsymptoticTestStatisticsDistribution):
          The distribution for the background-only hypothesis.

    Returns:
        ``List[List]``:
        The p-values for the test statistic corresponding to the :math:`p_{s+b}`,
        :math:`p_{b}`, and :math:`p_{s}`.

    .. seealso::

        :func:`~spey.hypothesis_testing.test_statistics.compute_teststatistics`,
        :func:`~spey.hypothesis_testing.asymptotic_calculator.compute_asymptotic_confidence_level`,
        :func:`~spey.hypothesis_testing.utils.pvalues`
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
