"""Optimisation tools based on scipy"""

import logging
import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np
import scipy

# pylint: disable=W1201, W1203, R0913

log = logging.getLogger("Spey")


def minimize(
    func: Callable[[np.ndarray], float],
    init_pars: List[float],
    do_grad: bool = False,
    hessian: Callable[[np.ndarray], np.ndarray] = None,
    bounds: List[Tuple[float, float]] = None,
    constraints: List[Dict] = None,
    **options,
) -> Tuple[float, np.ndarray]:
    """
    Minimize given function using scipy optimiser functionality

    Default options:
        method = "SLSQP"
        maxiter = 10000
        disp = False
        tol = 1e-6
        ntrials = 1

    :param func (`Callable[[np.ndarray], float]`): the objective function to be minimized.
    :param init_pars (`List[float]`): initial set of parameters
    :param do_grad (`bool`, default `False`): if true func is expected to return both objective and its gradient
    :param hessian (`Callable[[np.ndarray], np.ndarray]`, default `None`): hessian matrix of the objective function.
    :param bounds (`List[Tuple[float, float]]`, default `None`): Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP,
                                                                Powell, and trust-constr methods.
    :param constraints (`List[Dict]`, default `None`): Constraints definition. Only for COBYLA, SLSQP and trust-constr.
    :return `Tuple[float, np.ndarray]`: minimum value of the objective function and fit parameters
    """
    assert "poi_index" in list(options), "Please include `poi_index` in the options."

    method = options.pop("method", "SLSQP")
    tol = options.pop("tol", 1e-6)
    ntrials = max(options.pop("ntrials", 1), 1)
    poi_index = options.pop("poi_index")

    options.update({"maxiter": options.get("maxiter", 10000)})
    options.update({"disp": options.get("disp", False)})

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
            init_pars = opt.x
            for bdx, bound in enumerate(bounds):
                if bdx == poi_index and constraints is None:
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
