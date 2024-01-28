"""Tools for computing upper limit on parameter of interest"""

import logging
import warnings
from functools import partial
from typing import Callable, List, Tuple, Union, Literal

import numpy as np
import scipy

from spey.hypothesis_testing.test_statistics import compute_teststatistics
from spey.utils import ExpectationType

from .asymptotic_calculator import compute_asymptotic_confidence_level

__all__ = ["find_poi_upper_limit", "find_root_limits"]


def __dir__():
    return __all__


log = logging.getLogger("Spey")

# pylint: disable=W1203,C0103


class ComputerWrapper:
    """
    Wrapper for the computer function to track inputs and outputs

    Args:
        computer (``Callable[[float], float]``): desired function to be computed
    """

    def __init__(self, computer: Callable[[float], float]):
        self.computer = computer
        self._results = []

    def __call__(self, value: float) -> float:
        """Compute the input function and return its value"""
        self._results.append((value, self.computer(value)))
        return self[-1]

    def __getitem__(self, item: int) -> float:
        """Return result"""
        return self._results[item][1]

    def get_value(self, index: int) -> float:
        """
        Get input value of the execution

        Args:
            index (``int``): index of the execution

        Returns:
            ``float``:
            returns the input value of the execution
        """
        return self._results[index][0]


def find_root_limits(
    computer: Callable[[float], float],
    loc: float = 0.0,
    low_ini: float = 1.0,
    hig_ini: float = 1.0,
    low_bound: float = 1e-10,
    hig_bound: float = 1e5,
) -> Tuple[ComputerWrapper, ComputerWrapper]:
    """
    Find upper and lower bracket limits for the root finding algorithm

    Args:
        computer (``Callable[[float], float]``): Function that we want to find the root
        loc (``float``, default ``0.0``): location of the root e.g. ``0.95`` for :math:`1-CL_s` value
        low_ini (``float``, default ``1.0``): Initial value for low bracket
        hig_ini (``float``, default ``1.0``): initial value for high bracket
        low_bound (``float``, default ``1e-10``): Stop the execution below this value
        hig_bound (``float``, default ``1e5``): Stop the execution above this value

    Returns:
        ``Tuple[ComputerWrapper, ComputerWrapper]``:
        Returns lower and upper limits for the bracketing within a computer wrapper object.
    """
    assert callable(computer), "Invalid input. Computer must be callable."

    low, hig = low_ini, hig_ini

    low_computer = ComputerWrapper(computer)
    while low_computer(low) > loc:
        low *= 0.5
        if low < low_bound:
            break
    log.debug(f"Low results: {low_computer._results}")

    hig_computer = ComputerWrapper(computer)
    while hig_computer(hig) < loc:
        hig *= 2.0
        if hig > hig_bound:
            break
    log.debug(f"High results: {hig_computer._results}")

    return low_computer, hig_computer


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
    expected_pvalue: Literal["nominal", "1sigma", "2sigma"] = "nominal",
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
        log.debug(f"Running for p-value idx: {pvalue_idx}")
        comp = partial(computer, pvalue_idx=pvalue_idx)
        # Set an upper bound for the computation
        hig_bound = 1e5
        low, hig = find_root_limits(
            comp, loc=0.0, low_ini=low_init, hig_ini=hig_init, hig_bound=hig_bound
        )
        log.debug(f"low: {low[-1]}, hig: {hig[-1]}")

        # Check if its possible to find roots
        if np.sign(low[-1]) * np.sign(hig[-1]) > 0.0:
            log.warning(
                "Can not find the roots of the function, returning `inf`"
                + f"\n hig, low must bracket a root f({low.get_value(-1):.5e})={low[-1]:.5e}, "
                + f"f({hig.get_value(-1):.5e})={hig[-1]:.5e}. "
                + "This is likely due to low number of signal yields."
                * (hig.get_value(-1) >= hig_bound),
            )
            result.append(np.inf)
            continue

        with warnings.catch_warnings(record=True) as w:
            x0, r = scipy.optimize.toms748(
                comp,
                low.get_value(-1),
                hig.get_value(-1),
                k=2,
                xtol=2e-12,
                rtol=1e-4,
                full_output=True,
                maxiter=maxiter,
            )

        log.debug("Warnings:")
        if log.level == logging.DEBUG:
            for warning in w:
                log.debug(f"\t{warning.message}")
            log.debug("<><>" * 10)

        if not r.converged:
            log.warning(f"Optimiser did not converge.\n{r}")
        result.append(x0)
    return result if len(result) > 1 else result[0]
