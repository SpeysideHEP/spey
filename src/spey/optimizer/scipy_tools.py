"""Optimisation tools based on scipy"""

import logging
import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np
import scipy

from . import ValidateOpts

# pylint: disable=W1201, W1203, R0913

log = logging.getLogger("Spey")

_scipy_opts = ValidateOpts(
    opt_list=["method", "maxiter", "disp", "tol", "ntrials"], must_list=["poi_index"]
)


def minimize(
    func: Callable[[np.ndarray], float],
    init_pars: List[float],
    fixed_vals: List[bool],
    do_grad: bool = False,
    hessian: Callable[[np.ndarray], np.ndarray] = None,
    bounds: List[Tuple[float, float]] = None,
    constraints: List[Dict] = None,
    **options,
) -> Tuple[float, np.ndarray]:
    """
    Minimise an objective with :func:`scipy.optimize.minimize` and a retry loop.

    On failed convergence the loop widens the parameter bounds (POI by 2x,
    others by 10x per side) up to ``ntrials`` times before giving up. The
    caller-supplied ``init_pars``, ``bounds`` and ``constraints`` lists are
    copied internally so they are never mutated.

    Args:
        func (``Callable[[np.ndarray], float]``): Objective function. When
          ``do_grad=True`` it must instead return ``(value, gradient)``.
        init_pars (``List[float]``): Initial parameter values.
        fixed_vals (``List[bool]``): Boolean mask the same length as ``init_pars``;
          entries that are ``True`` are converted into equality constraints that
          pin the parameter to its initial value.
        do_grad (``bool``, default ``False``): If ``True``, ``func`` is expected
          to return ``(value, gradient)``.
        hessian (``Callable[[np.ndarray], np.ndarray]``, default ``None``):
          Hessian forwarded to scipy as the ``hess`` argument.
        bounds (``List[Tuple[float, float]]``, default ``None``): Per-parameter
          ``(lower, upper)`` bounds for the SLSQP/L-BFGS-B/TNC/Nelder-Mead/Powell/
          trust-constr methods.
        constraints (``List[Dict]``, default ``None``): Extra scipy-style
          constraint dicts (COBYLA/SLSQP/trust-constr only).
        **options: Recognised keys are ``method`` (default ``"SLSQP"``),
          ``maxiter`` (default ``10000``), ``disp`` (default ``False``),
          ``tol`` (default ``1e-6``), and ``ntrials`` (default ``1``).
          ``poi_index`` is required and is removed before being forwarded to
          scipy.

    Returns:
        ``Tuple[float, np.ndarray]``:
        The minimum objective value and the corresponding fit parameter vector.
    """
    assert "poi_index" in list(options), "Please include `poi_index` in the options."

    options = _scipy_opts(options)

    method = options.pop("method", "SLSQP")
    tol = options.pop("tol", 1e-6)
    ntrials = max(options.pop("ntrials", 1), 1)
    poi_index = options.pop("poi_index")

    options.setdefault("maxiter", 10000)
    options.setdefault("disp", False)

    def make_constraint(index: int, value: float) -> Callable[[np.ndarray], float]:
        def func(vector: np.ndarray) -> float:
            return vector[index] - value

        return func

    # Copy caller-owned containers so the retry loop's in-place mutations
    # below don't leak back to the caller across multiple fit() calls.
    constraints = [] if constraints is None else list(constraints)
    init_pars = list(init_pars)
    bounds = [tuple(b) for b in bounds] if bounds is not None else None
    for idx, isfixed in enumerate(fixed_vals):
        if isfixed:
            log.debug(f"Constraining {idx} to value {init_pars[idx]}")
            constraints.append(
                {
                    "type": "eq",
                    "fun": make_constraint(idx, init_pars[idx]),
                }
            )

    ntrial = 0
    while ntrial < ntrials:
        ntrial += 1
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=RuntimeWarning)
            opt = scipy.optimize.minimize(
                func,
                init_pars,
                method=method,
                jac=do_grad,
                hess=hessian,
                bounds=bounds,
                constraints=constraints,
                tol=tol,
                options=options,
            )
        if not opt.success and ntrial < ntrials:
            # Expand the boundaries of the statistical model dynamically to be able to converge
            # Note that it is important to respect the lower bounds especially since there might
            # be bounds that lead to negative yields in the statistical model, e.g. Asimov data,
            # background + mu * signal etc.
            log.warning("Optimiser has not been able to satisfy all the conditions:")
            log.warning(opt.message + " Expanding the bounds...")
            init_pars = list(opt.x)
            for bdx, bound in enumerate(bounds):
                if bdx == poi_index:
                    bounds[bdx] = (bound[0], bound[1] * 2.0)
                else:
                    bounds[bdx] = (min(bound[0], bound[0] * 10.0), bound[1] * 10.0)
        elif opt.success:
            break

    if not opt.success:
        log.warning("Optimiser has not been able to satisfy all the conditions:")
        log.warning(opt.message)
        log.debug(f"func: {opt.fun}")
        log.debug(f"parameters: {opt.x}")
        if do_grad:
            log.debug(f"gradients: {func(opt.x)[-1]}")

    return opt.fun, opt.x
