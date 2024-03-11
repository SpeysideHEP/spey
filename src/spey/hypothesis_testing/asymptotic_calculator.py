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
    Compute p values i.e. :math:`p_{s+b}`, :math:`p_b` and :math:`p_s`
        
    .. math::
    
        p_{s+b}&=& \int_{-\infty}^{-\sqrt{q_{\mu,A}} - \Delta q_\mu} \mathcal{N}(x| 0, 1) dx \\
        p_{b}&=& \int_{-\infty}^{-\Delta q_\mu} \mathcal{N}(x| 0, 1) dx \\
        p_{s} &=& p_{s+b}/ p_{b}

    where :math:`q_\mu` stands for the test statistic and A stands for Assimov.
    
    .. math::
        
        \Delta q_\mu = \begin{cases} 
        \sqrt{q_{\mu}} - \sqrt{q_{\mu,A}}, &  \text{if}\ \sqrt{q_{\mu}} \leq \sqrt{q_{\mu,A}} \\
        \frac{\sqrt{q_{\mu}} - \sqrt{q_{\mu,A}}}{2\ \sqrt{q_{\mu,A}}}, & \text{otherwise} 
        \end{cases}

    Note that the CDF has a cutoff at :math:`-\sqrt{q_{\mu,A}}`, hence if 
    :math:`p_{s\ {\rm or}\ s+b} < -\sqrt{q_{\mu,A}}` p-value will not be computed.

    .. seealso::
    
        eq. 66 of :xref:`1007.1727`

    Args:
        sqrt_qmuA (``float``): test statistic for Asimov data :math:`\sqrt{q_{\mu,A}}`.
        delta_test_statistic (``float``): :math:`\Delta q_\mu`
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
        
    .. seealso::

        :func:`~spey.hypothesis_testing.test_statistics.compute_teststatistics`, 
        :func:`~spey.hypothesis_testing.utils.pvalues`,
        :func:`~spey.hypothesis_testing.utils.expected_pvalues`
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
