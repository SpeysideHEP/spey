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
    fixed_vals: list[bool],
    do_grad: bool = False,
    hessian: Callable[[np.ndarray], np.ndarray] = None,
    bounds: List[Tuple[float, float]] = None,
    constraints: List[Dict] = None,
    **options,
) -> Tuple[float, np.ndarray]:

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
