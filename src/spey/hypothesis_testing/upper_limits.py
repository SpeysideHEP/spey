"""Tools for computing upper limit on parameter of interest"""

from typing import Callable, Tuple, Text, List, Union
from functools import partial
import warnings, scipy

from spey.hypothesis_testing.test_statistics import compute_teststatistics
from spey.utils import ExpectationType
from .asymptotic_calculator import compute_asymptotic_confidence_level

__all__ = ["find_poi_upper_limit", "find_root_limits"]


def __dir__():
    return __all__


def find_root_limits(
    computer: Callable[[float], float],
    loc: float = 0.0,
    low_ini: float = 1.0,
    hig_ini: float = 1.0,
) -> Tuple[float, float]:
    """
    Find upper and lower bracket limits for the root finding algorithm

    Args:
        computer (``Callable[[float], float]``): Function that we want to find the root
        loc (``float``, default ``0.0``): location of the root e.g. ``0.95`` for :math:`1-CL_s` value
        low_ini (``float``, default ``1.0``): Initial value for low bracket
        hig_ini (``float``, default ``1.0``): initial value for high bracket

    Returns:
        ``Tuple[float, float]``:
        Returns lower and upper limits for the bracketing.
    """
    assert callable(computer), "Invalid input. Computer must be callable."

    low, hig = low_ini, hig_ini
    while computer(low) > loc:
        low *= 0.5
        if low < 1e-10:
            break
    while computer(hig) < loc:
        hig *= 2.0
        if hig > 1e3:
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
    maxiter: int = 10000,
) -> Union[float, List[float]]:
    r"""
    Find upper limit for parameter of interest, :math:`\mu`

    Args:
        maximum_likelihood (``Tuple[float, float]``): Tuple including :math:`\hat\mu`
          and minimum negative log-likelihood.
        logpdf (``Callable[[float], float]``): log-likelihood as function of POI,
          :math:`\log\mathcal{L}(\mu)`
        maximum_asimov_likelihood (``Tuple[float, float]``): Tuple including
          :math:`\hat\mu_A` and minimum negative log-likelihood for Asimov data.
        asimov_logpdf (``Callable[[float], float]``): log-likelihood as function of POI,
          :math:`\log\mathcal{L}_A(\mu)` for Asimov data.
        expected (~spey.ExpectationType): Sets which values the fitting algorithm should
          focus and p-values to be computed.

          * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
            prescriotion which means that the experimental data will be assumed to be the truth
          * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
            post-fit prescriotion which means that the experimental data will be assumed to be
            the truth.
          * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
            prescription which means that the SM will be assumed to be the truth.

        confidence_level (``float``, default ``0.95``): Determines the confidence level of the upper
              limit i.e. the value of :math:`1-CL_s`. It needs to be between ``[0,1]``.
        allow_negative_signal (``bool``, default ``True``): _description_
        low_init (``float``, default ``None``): Lower limit for the search algorithm to start
        hig_init (``float``, default ``None``): Upper limit for the search algorithm to start
        expected_pvalue (``Text``, default ``"nominal"``): In case of :obj:`~spey.ExpectationType.aposteriori`
          and :obj:`~spey.ExpectationType.apriori` expectation, gives the choice to find excluded upper
          limit for statistical deviations as well.

          * ``"nominal"``: only find the upper limit for the central p-value. Returns a single value.
          * ``"1sigma"``: find the upper limit for central p-value and :math:`1\sigma` fluctuation from
            background. Returns 3 values.
          * ``"2sigma"``: find the upper limit for central p-value and :math:`1\sigma` and
            :math:`2\sigma` fluctuation from background. Returns 5 values.

            .. note::

              For ``expected=spey.ExpectationType.observed``, ``expected_pvalue`` argument will
              be overwritten to ``"nominal"``.

        maxiter (``int``, default ``10000``): Maximum iteration limit for the optimiser.

    Returns:
        ``Union[float, List[float]]``:
        In case of nominal values it returns a single value for the upper limit. In case of
        ``expected_pvalue="1sigma"`` or ``expected_pvalue="2sigma"`` it will return a list of
        multiple upper limit values for fluctuations as well as the central value. The
        output order is :math:`-2\sigma` value, :math:`-1\sigma` value, central value,
        :math:`1\sigma` and :math:`2\sigma` value.
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
                compute_asymptotic_confidence_level(sqrt_qmuA, delta_teststat, test_stat)[
                    0 if expected == ExpectationType.observed else 1
                ],
            )
        )
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
                comp,
                low,
                hig,
                k=2,
                xtol=2e-12,
                rtol=1e-4,
                full_output=True,
                maxiter=maxiter,
            )
        if not r.converged:
            warnings.warn(f"Optimiser did not converge.\n{r}", category=RuntimeWarning)
        result.append(x0)
    return result if len(result) > 1 else result[0]
