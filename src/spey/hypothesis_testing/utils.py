"""Tools for computing confidence level and pvalues"""

from typing import Tuple, Text, List
import warnings
import numpy as np

from spey.hypothesis_testing.asymptotic_calculator import AsymptoticTestStatisticsDistribution

__all__ = ["compute_confidence_level", "pvalues", "expected_pvalues"]


def compute_confidence_level(
    sqrt_qmuA: float, delta_test_statistic: float, test_stat: Text = "qtilde"
) -> Tuple[List[float], List[float]]:
    r"""
    Compute confidence limits i.e. :math:`CL_{s+b}`, :math:`CL_b` and :math:`CL_s`

    .. note::

        see :func:`~spey.hypothesis_testing.test_statistics.compute_teststatistics` for
        details regarding the arguments.

    Args:
        sqrt_qmuA (``float``): test statistic for Asimov data :math:`\sqrt{q_{\mu,A}}`.
        delta_test_statistic (``float``): :math:`\Delta{\sqrt{q_{\mu}},\sqrt{q_{\mu,A}}}`
        test_statistics (``Text``, default ``"qtilde"``): test statistics.

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
        ``Tuple[List[float], List[float]]``:
        returns p-values and expected p-values.
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
        CLs = np.true_divide(CLsb, CLb, dtype=np.float32)
    return CLsb, CLb, CLs if CLb != 0.0 else 0.0


def expected_pvalues(
    sig_plus_bkg_distribution: AsymptoticTestStatisticsDistribution,
    bkg_only_distribution: AsymptoticTestStatisticsDistribution,
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
