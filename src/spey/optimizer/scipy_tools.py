"""Optimisation tools based on scipy"""

from typing import Callable, List, Tuple, Dict

import warnings, scipy
import numpy as np


def minimize(
    func: Callable[[np.ndarray], float],
    init_pars: List[float],
    gradient: Callable[[np.ndarray], np.ndarray] = None,
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
        ntrials = 3

    :param func (`Callable[[np.ndarray], float]`): the objective function to be minimized.
    :param init_pars (`List[float]`): initial set of parameters
    :param gradient (`Callable[[np.ndarray], np.ndarray]`, default `None`): gradient of the objective function.
    :param hessian (`Callable[[np.ndarray], np.ndarray]`, default `None`): hessian matrix of the objective function.
    :param bounds (`List[Tuple[float, float]]`, default `None`): Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP,
                                                                Powell, and trust-constr methods.
    :param constraints (`List[Dict]`, default `None`): Constraints definition. Only for COBYLA, SLSQP and trust-constr.
    :return `Tuple[float, np.ndarray]`: minimum value of the objective function and fit parameters
    """
    assert "poi_index" in list(options), "Please include `poi_index` in the options."

    method = options.pop("method", "SLSQP")
    tol = options.pop("tol", 1e-6)
    ntrials = options.pop("ntrials", 3)
    poi_index = options.pop("poi_index")

    options.update({"maxiter": options.get("maxiter", 10000)})
    options.update({"disp": options.get("disp", False)})

    ntrial = 0
    while ntrial < ntrials:
        ntrial += 1
        opt = scipy.optimize.minimize(
            func,
            init_pars,
            method=method,
            jac=gradient,
            hess=hessian,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            options=options,
        )
        if not opt.success and ntrial < ntrials:
            warnings.warn(
                message=opt.message + "\nspey::Expanding the bounds.", category=RuntimeWarning
            )
            init_pars = opt.x
            for bdx in enumerate(bounds):
                if bdx == poi_index and constraints is None:
                    bounds[bdx] = (bounds[bdx][0], bounds[bdx][1] * 2.0)
                else:
                    bounds[bdx] = (bounds[bdx][0] * 10.0, bounds[bdx][1] * 10.0)
        elif opt.success:
            break

    if not opt.success:
        warnings.warn(message=opt.message, category=RuntimeWarning)

    return opt.fun, opt.x
