"""Tools for computing confidence level and pvalues at the asymptotic limit"""

from typing import List, Text, Tuple

from .distributions import AsymptoticTestStatisticsDistribution
from .utils import expected_pvalues, pvalues

__all__ = ["compute_asymptotic_confidence_level"]


def __dir__():
    return __all__


def compute_asymptotic_confidence_level(
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
        test_stat (``Text``, default ``"qtilde"``): test statistics.

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
    cutoff = -sqrt_qmuA  # use clipped normal -> normal will mean -np.inf
    # gives more stable result for cases that \hat\mu > \mu : see eq 14, 16 :xref:`1007.1727`

    sig_plus_bkg_distribution = AsymptoticTestStatisticsDistribution(-sqrt_qmuA, cutoff)
    bkg_only_distribution = AsymptoticTestStatisticsDistribution(0.0, cutoff)

    CLsb_obs, CLb_obs, CLs_obs = pvalues(
        delta_test_statistic, sig_plus_bkg_distribution, bkg_only_distribution
    )
    CLsb_exp, CLb_exp, CLs_exp = expected_pvalues(
        sig_plus_bkg_distribution, bkg_only_distribution
    )

    return ([CLsb_obs], CLsb_exp) if test_stat == "q0" else ([CLs_obs], CLs_exp)
