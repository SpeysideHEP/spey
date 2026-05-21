import logging
from typing import Callable, Dict, List, Tuple

import numpy as np
from iminuit import Minuit

from . import ValidateOpts

# pylint: disable=unnecessary-lambda-assignment

log = logging.getLogger("Spey")

_minuit_opts = ValidateOpts(
    opt_list=["errordef", "maxiter", "disp", "tol", "strategy", "method"],
    remove_list=["poi_index"],
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
    Minimise an objective using :class:`iminuit.Minuit`.

    The ``constraints`` and ``hessian`` arguments are accepted for API
    compatibility with :func:`spey.optimizer.scipy_tools.minimize` but are
    ignored by minuit (minuit applies bounds and fixed parameters directly).

    Args:
        func (``Callable[[np.ndarray], float]``): Objective function. When
          ``do_grad=True`` it must return ``(value, gradient)``; ``gradient``
          is then forwarded to ``Minuit(..., grad=grad)``.
        init_pars (``List[float]``): Initial parameter values.
        fixed_vals (``List[bool]``): Boolean mask the same length as
          ``init_pars``; ``True`` entries are passed to ``Minuit.fixed``.
        do_grad (``bool``, default ``False``): Whether ``func`` returns its
          gradient alongside the objective.
        hessian (``Callable[[np.ndarray], np.ndarray]``, default ``None``):
          Accepted for compatibility; not used by minuit.
        bounds (``List[Tuple[float, float]]``, default ``None``): Per-parameter
          ``(lower, upper)`` bounds for ``Minuit.limits``. Use ``(None, None)``
          for an unbounded parameter; when the whole list is ``None`` all
          parameters become unbounded.
        constraints (``List[Dict]``, default ``None``): Accepted for
          compatibility; not used by minuit.
        **options: Recognised keys are ``method`` (``"migrad"`` (default) or
          ``"simplex"``), ``maxiter`` (default ``10000``, passed as
          ``ncall``), ``tol`` (default ``1e-6``), ``disp`` (default ``0``,
          minuit print level), ``strategy`` (default ``0``), and ``errordef``
          (default ``Minuit.LIKELIHOOD``). ``poi_index`` is stripped.

    Returns:
        ``Tuple[float, np.ndarray]``:
        Final objective value (``opt.fval``) and the corresponding parameter
        vector (``opt.values``).

    Raises:
        ``ValueError``: If ``method`` is neither ``"migrad"`` nor ``"simplex"``.
    """

    options = _minuit_opts(options)
    method = options.get("method", "migrad")

    if do_grad:
        objective = lambda pars: func(pars)[0]
        grad = lambda pars: func(pars)[1]
    else:
        objective = func
        grad = None

    opt = Minuit(objective, np.atleast_1d(init_pars), grad=grad)
    opt.limits = [(None, None)] * len(init_pars) if bounds is None else bounds
    opt.fixed = fixed_vals
    opt.print_level = options.get("disp", 0)
    opt.errordef = options.get("errordef", Minuit.LIKELIHOOD)
    opt.strategy = options.get("strategy", 0)
    opt.tol = options.get("tol", 1e-6)
    ncall = options.get("maxiter", 10000)
    if method == "migrad":
        opt.migrad(ncall=ncall)
    elif method == "simplex":
        opt.simplex(ncall=ncall)
    else:
        raise ValueError(f"Unknown method: {method}")

    # https://github.com/scikit-hep/iminuit/blob/1fb039cba09417cdf5a4f67749f58e9030dc619b/src/iminuit/minimize.py#L124C1-L136C67
    if opt.valid:
        message = "Optimization terminated successfully"
        if opt.accurate:
            message += "."
        else:
            message += ", but uncertainties are unrealiable."
            log.debug(message)
    else:
        message = "Optimization failed."
        fmin = opt.fmin
        if fmin.has_reached_call_limit:
            message += " Call limit was reached."
        if fmin.is_above_max_edm:
            message += " Estimated distance to minimum too large."
        log.warning(message)

    # if opt.valid:
    #     # Extra call to hesse() after migrad() is always needed for good error estimates. If you pass a user-provided gradient to MINUIT, convergence is faster.
    #     opt.hesse()
    #     hess_inv = opt.covariance
    #     corr = hess_inv.correlation()
    #     log.debug(f"corr: {corr}")

    log.debug("unc: %s", opt.errors)
    return float(opt.fval), np.array(opt.values)
