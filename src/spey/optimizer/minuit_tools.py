import logging
from typing import Callable, Dict, List, Tuple

import iminuit
import numpy as np

from . import ValidateOpts

# pylint: disable=unnecessary-lambda-assignment

log = logging.getLogger("Spey")

_minuit_opts = ValidateOpts(
    opt_list=["errordef", "maxiter", "disp", "tol", "strategy"], remove_list=["poi_index"]
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

    if do_grad:
        objective = lambda pars: func(pars)[0]
        grad = lambda pars: func(pars)[1]
    else:
        objective = func
        grad = None

    opt = iminuit.Minuit(objective, np.atleast_1d(init_pars), grad=grad)
    opt.limits = bounds
    opt.fixed = fixed_vals
    opt.print_level = options.get("disp", False)
    opt.errordef = options.get("errordef", 1)
    opt.strategy = options.get("strategy", int(do_grad))
    opt.tol = options.get("tol", 1e-6)
    opt.migrad(ncall=options.get("maxiter", 10000))

    # https://github.com/scikit-hep/iminuit/blob/23bad7697e39d363f259ca8349684df939b1b2e6/src/iminuit/_minimize.py#L111-L130
    message = "Optimization terminated successfully."
    if not opt.valid:
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

    log.debug(f"unc: {opt.errors}")
    return opt.fval, opt.values
