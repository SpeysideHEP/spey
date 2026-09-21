"""Tools for computing upper limit on parameter of interest"""

import logging
from collections import deque
from functools import partial
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import scipy

from spey.hypothesis_testing.test_statistics import compute_teststatistics
from spey.system.exceptions import AsimovTestStatZero
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
        self._results = deque(maxlen=10)

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
        loc (``float``, default ``0.0``): location of the root e.g. ``0.95`` for :math:`CL_s` value
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
    while low_computer(low) > loc and low > low_bound:
        low = max(low * 0.2, low_bound)
    log.debug(f"Low results: {low_computer._results}")

    hig_computer = ComputerWrapper(computer)
    while hig_computer(hig) < loc and hig < hig_bound:
        hig = min(hig * 5.0, hig_bound)
    log.debug(f"High results: {hig_computer._results}")

    return low_computer, hig_computer


def _exclusion_persists(
    computer: Callable[[float], float],
    root: float,
    hig_bound: float,
    probes: Tuple[float, ...] = (1.5, 3.0, 10.0),
) -> bool:
    r"""
    Check that the exclusion still holds above a candidate upper limit.

    A :math:`(1-\alpha)` CL upper limit is the boundary of the excluded set
    :math:`\{\mu : CL_s(\mu) < \alpha\}`, which is only an *upper* limit when that set
    is a ray, i.e. when every :math:`\mu` above the root stays excluded.  With the
    sign convention of :func:`find_poi_upper_limit` — ``computer(mu) > 0`` outside the
    confidence interval — that means ``computer`` must remain non-negative above the
    root.

    A root can otherwise be returned for a :math:`CL_s` curve that is not monotonically
    decreasing.  This happens when the asymptotic approximation degrades, most commonly
    when :math:`\sqrt{q_{\mu,A}} \to 0` because the Asimov dataset barely constrains
    :math:`\mu` (e.g. a signal strength nearly degenerate with another free parameter).
    The test statistic then depends on a near-zero divisor and its sign is numerically
    meaningless, so a bracketing solver happily converges onto noise.

    Args:
        computer (``Callable[[float], float]``): Function whose root is the upper
          limit; negative inside the confidence interval, positive outside.
        root (``float``): Candidate upper limit returned by the root finder.
        hig_bound (``float``): Largest :math:`\mu` worth probing.
        probes (``Tuple[float, ...]``, default ``(1.5, 3.0, 10.0)``): Multiples of
          ``root`` at which the exclusion is re-tested.

    Returns:
        ``bool``:
        ``True`` when the exclusion holds at every probed point above ``root`` (or when
        no probe is usable), ``False`` when the exclusion is lost, which marks the root
        as spurious.
    """
    if not np.isfinite(root) or root <= 0.0:
        return True

    for factor in probes:
        point = root * factor
        if point > hig_bound:
            break
        value = computer(point)
        if not np.isfinite(value):
            continue
        if value < 0.0:
            log.debug(
                "Exclusion lost above the candidate limit: f(%.5e) = %.5e < 0",
                point,
                value,
            )
            return False
    return True


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
    validate_limit: bool = True,
    asimov_teststat_tolerance: float = 1e-3,
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
              limit i.e. the value of :math:`CL_s`. It needs to be between ``[0,1]``.
        allow_negative_signal (``bool``, default ``True``): allow for negative signal values. This will
            change the computation of the test statistic.
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
        validate_limit (``bool``, default ``True``): Reject a root that is not a genuine
          upper limit.  A :math:`CL_s` curve that is not monotonically decreasing can
          present a sign change to the bracketing solver even though no upper limit
          exists; when this happens ``inf`` is returned together with a warning.  See
          :func:`_exclusion_persists` for the failure mode this protects against.  Set
          to ``False`` to recover the unchecked behaviour.
        asimov_teststat_tolerance (``float``, default ``1e-3``): Smallest
          :math:`\sqrt{q_{\mu,A}}` at the root for which the asymptotic :math:`CL_s`
          is still considered meaningful.  Eq. (66) of :xref:`1007.1727` divides by this
          quantity, so below the tolerance the p-value is numerical noise and ``inf``
          is returned.  Only used when ``validate_limit`` is ``True``.

    Returns:
        ``Union[float, List[float]]``:
        In case of nominal values it returns a single value for the upper limit. In case of
        ``expected_pvalue="1sigma"`` or ``expected_pvalue="2sigma"`` it will return a list of
        multiple upper limit values for fluctuations as well as the central value. The
        output order is :math:`-2\sigma` value, :math:`-1\sigma` value, central value,
        :math:`1\sigma` and :math:`2\sigma` value.

    .. versionchanged:: 0.2.8
        Roots found on a non-monotonic :math:`CL_s` curve are rejected: the bracket must
        be oriented so that the exclusion is *entered* with increasing :math:`\mu`, and
        the exclusion must persist above the root.  Previously such a root was returned
        as if it were a valid upper limit.
    """
    assert expected_pvalue in [
        "nominal",
        "1sigma",
        "2sigma",
    ], f"Unknown pvalue range {expected_pvalue}"
    if expected is ExpectationType.observed:
        expected_pvalue = "nominal"
    test_stat = "q" if allow_negative_signal else "qtilde"

    def asimov_teststatistic(poi_test: float) -> float:
        r"""
        Evaluate :math:`\sqrt{q_{\mu,A}}`, the Asimov test statistic, at ``poi_test``.

        This is the quantity the asymptotic p-values are built on: eq. (66) of
        :xref:`1007.1727` divides by it, so when it approaches zero the resulting
        :math:`CL_s` is dominated by numerical noise rather than by the data.

        Args:
            poi_test (``float``): Parameter of interest value.

        Returns:
            ``float``:
            :math:`\sqrt{q_{\mu,A}}`, or ``0.0`` when the Asimov test statistic vanishes.
        """
        try:
            _, sqrt_qmuA, _ = compute_teststatistics(
                poi_test,
                maximum_likelihood,
                logpdf,
                maximum_asimov_likelihood,
                asimov_logpdf,
                test_stat,
            )
        except AsimovTestStatZero:
            return 0.0
        return float(sqrt_qmuA)

    def computer(poi_test: float, pvalue_idx: int) -> float:
        """Compute 1 - CLs(POI) = `confidence_level`"""
        try:
            _, sqrt_qmuA, delta_teststat = compute_teststatistics(
                poi_test,
                maximum_likelihood,
                logpdf,
                maximum_asimov_likelihood,
                asimov_logpdf,
                test_stat,
            )
            pvalue = [
                1.0 - x
                for x in compute_asymptotic_confidence_level(
                    sqrt_qmuA, delta_teststat, test_stat
                )[0 if expected == ExpectationType.observed else 1]
            ]
        except AsimovTestStatZero as err:
            log.debug(err)
            pvalue = [0.0] if expected == ExpectationType.observed else [0.0] * 5
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

        low_poi, hig_poi = low.get_value(-1), hig.get_value(-1)
        low_val, hig_val = low[-1], hig[-1]
        del low, hig

        # The bracket has to be oriented so that increasing mu *enters* the exclusion:
        # negative (inside the interval) at the low end, positive (excluded) at the
        # high end. The reverse ordering is a sign change of the CLs curve in the wrong
        # direction, which is not an upper limit.
        if validate_limit and low_poi < hig_poi and not low_val < 0.0 < hig_val:
            log.warning(
                "CLs is not decreasing with the parameter of interest "
                f"(f({low_poi:.5e})={low_val:.5e}, f({hig_poi:.5e})={hig_val:.5e}), "
                "so the sign change is not an upper limit. Returning `inf`."
            )
            result.append(np.inf)
            continue

        x0, r = scipy.optimize.toms748(
            comp,
            low_poi,
            hig_poi,
            k=2,
            xtol=2e-12,
            rtol=1e-4,
            full_output=True,
            maxiter=maxiter,
        )

        if not r.converged:
            log.warning(f"Optimiser did not converge.\n{r}")

        if validate_limit:
            # The asymptotic CLs is only meaningful while sqrt(q_{mu,A}) stays away
            # from zero; eq. (66) of arXiv:1007.1727 divides by it. A vanishing Asimov
            # test statistic means the Asimov dataset does not constrain the POI, and
            # the sign changes the solver locks onto are then numerical noise.
            sqrt_qmuA = asimov_teststatistic(x0)
            if sqrt_qmuA < asimov_teststat_tolerance:
                log.warning(
                    f"The Asimov test statistic at mu = {x0:.5e} is numerically zero "
                    f"(sqrt(q_muA) = {sqrt_qmuA:.3e} < {asimov_teststat_tolerance:.1e}), "
                    "so the asymptotic CLs is dominated by numerical noise and this root "
                    "is not a meaningful upper limit. Returning `inf`. This happens when "
                    "the Asimov dataset barely constrains the parameter of interest, for "
                    "instance when it is nearly degenerate with another free parameter."
                )
                result.append(np.inf)
                continue

            if not _exclusion_persists(comp, x0, hig_bound):
                log.warning(
                    f"The exclusion does not hold above mu = {x0:.5e}, so the CLs curve "
                    "is not monotonically decreasing and this root is not an upper "
                    "limit. Returning `inf`."
                )
                result.append(np.inf)
                continue

        result.append(x0)
    return result if len(result) > 1 else result[0]


def bracket_and_solve(
    computer: Callable[[float], float],
    inner: float,
    outer: float,
    expand: Callable[[float], float],
    contract: Callable[[float], float],
    outer_stop: Callable[[float], bool],
    inner_stop: Callable[[float], bool],
    *,
    retry_inner: Optional[float] = None,
    retry_stop: Optional[Callable[[float], bool]] = None,
    debug_tag: str = "",
    xtol: float = 2e-12,
    rtol: float = 1e-4,
    maxiter: int = 10000,
) -> Tuple[float, float, float]:
    r"""
    Bracket a root of ``computer`` and solve with :func:`~scipy.optimize.toms748`.

    The function works in two phases:

    1. **Bracketing** — walks ``outer`` outward via ``expand`` until
       ``computer(outer) > 0`` (outside the confidence interval) or
       ``outer_stop(outer)`` fires, then walks ``inner`` inward via ``contract``
       until ``computer(inner) < 0`` (inside the interval) or
       ``inner_stop(inner)`` fires.  If the two sides still share the same sign
       and ``retry_inner`` is provided, a second contraction from ``retry_inner``
       is attempted (used in the POI case when :math:`\hat\mu` is far from zero).

    2. **Root-finding** — once a valid bracket ``[a, b]`` with opposite signs is
       established, :func:`~scipy.optimize.toms748` (TOMS Algorithm 748, a
       superlinearly convergent bracketing method) is called to locate the root to
       the requested tolerances.

    Args:
        computer (``Callable[[float], float]``): Function whose sign changes at
            the interval boundary.  By convention ``computer(val) < 0`` *inside*
            the confidence interval (near the peak) and ``> 0`` *outside*.
        inner (``float``): Initial bracketing value near the likelihood peak,
            expected to satisfy ``computer(inner) < 0``.
        outer (``float``): Initial bracketing value far from the likelihood peak,
            expected to satisfy ``computer(outer) > 0``.
        expand (``Callable[[float], float]``): Maps the current ``outer`` to a
            new value further from the peak (e.g. ``lambda h: h * 2``).
        contract (``Callable[[float], float]``): Maps the current ``inner`` to a
            new value closer to the peak (e.g. ``lambda l: l * 0.5``).
        outer_stop (``Callable[[float], bool]``): Returns ``True`` when ``outer``
            has reached an absolute boundary beyond which further expansion should
            not proceed.
        inner_stop (``Callable[[float], bool]``): Returns ``True`` when ``inner``
            is so close to the peak that further contraction would be meaningless.
        retry_inner (``float``, optional): Alternative starting point for a second
            contraction pass when the first pass fails to produce a valid bracket.
        retry_stop (``Callable[[float], bool]``, optional): Stop condition applied
            during the retry contraction; must be provided when ``retry_inner`` is
            not ``None``.
        debug_tag (``str``, default ``""``): Short label prepended to debug-level
            log messages, useful for identifying which root (e.g. ``"left"`` or
            ``"right"``) is currently being solved.
        xtol (``float``, default ``2e-12``): Absolute tolerance for
            :func:`~scipy.optimize.toms748`.  The solver stops when the bracket
            width drops below this value.
        rtol (``float``, default ``1e-4``): Relative tolerance for
            :func:`~scipy.optimize.toms748`.  The solver stops when the bracket
            width is smaller than ``rtol * |root|``.
        maxiter (``int``, default ``10000``): Maximum number of function
            evaluations allowed inside :func:`~scipy.optimize.toms748`.

    Returns:
        ``Tuple[float, float, float]``:
        ``(final_inner, final_outer, root)`` where ``final_inner`` and
        ``final_outer`` are the last bracket endpoints reached during the
        expansion/contraction phase, and ``root`` is the value returned by
        :func:`~scipy.optimize.toms748`.  ``root`` is ``nan`` when no bracket
        with opposite signs could be established.
    """
    outer_cw = ComputerWrapper(computer)
    while outer_cw(outer) < 0.0 and not outer_stop(outer):
        outer = expand(outer)

    inner_cw = ComputerWrapper(computer)
    while inner_cw(inner) > 0.0 and not inner_stop(inner):
        inner = contract(inner)

    log.debug(
        f"[{debug_tag}] inner f({inner:.5e})={inner_cw[-1]:.5e},"
        f" outer f({outer:.5e})={outer_cw[-1]:.5e}"
    )

    if np.sign(inner_cw[-1]) == np.sign(outer_cw[-1]) and retry_inner is not None:
        inner = retry_inner
        inner_cw = ComputerWrapper(computer)
        while inner_cw(inner) > 0.0 and not retry_stop(inner):
            inner = contract(inner)
        log.debug(
            f"[{debug_tag} retry] inner f({inner:.5e})={inner_cw[-1]:.5e},"
            f" outer f({outer:.5e})={outer_cw[-1]:.5e}"
        )

    if np.sign(inner_cw[-1]) != np.sign(outer_cw[-1]):
        a, b = min(inner, outer), max(inner, outer)
        x0, _ = scipy.optimize.toms748(
            computer,
            a,
            b,
            k=2,
            xtol=xtol,
            rtol=rtol,
            full_output=True,
            maxiter=maxiter,
        )
        return inner, outer, x0

    return inner, outer, float("nan")
