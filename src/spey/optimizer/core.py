import logging
import os
from importlib.util import find_spec
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from spey.base.model_config import ModelConfig

# pylint: disable=W1203

log = logging.getLogger("Spey")


def _get_minimizer(name: str):
    from .scipy_tools import minimize

    if name == "scipy":
        return minimize
    elif name == "minuit":
        if find_spec("iminuit") is not None:
            from .minuit_tools import minimize as minuit_opt

            return minuit_opt

        log.warning("minuit optimiser is not available, using scipy")
        return minimize
    raise ValueError(f"{name} is not availabe.")


def fit(
    func: Callable[[np.ndarray], np.ndarray],
    model_configuration: ModelConfig,
    do_grad: bool = False,
    hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    initial_parameters: Optional[np.ndarray] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    fixed_poi_value: Optional[float] = None,
    logpdf: Optional[Callable[[List[float]], float]] = None,
    constraints: Optional[List[Dict]] = None,
    **options,
) -> Tuple[float, np.ndarray]:
    """
    Operating the fitting of the given function.

    Args:
        func (``Callable[[np.ndarray], np.ndarray]``): Function to be optimised. If `do_grad=True`,
            function is expected to return `Tuple[np.ndarray, np.ndarray]` where first is the value
            of the function and the second is gradients with respect to the nuisance parameters.
        model_configuration (``~spey.base.model_config.ModelConfig``): Model configuration.
        do_grad (``bool``, default ``False``): boolean to set gradient on or off.
        hessian (``Optional[Callable[[np.ndarray], np.ndarray]]``, default ``None``): hessian of the function
            with respect to variational parameters. Currently not used.
        initial_parameters (``Optional[np.ndarray]``, default ``None``): Initial guess for the variational
            parameters.
        bounds (``Optional[List[Tuple[float, float]]]``, default ``None``): Bounds for the variational parameters
            each item has to have a lower and upper bound definition `(<lower>, <upper>)` in case of open boundary
            input can be `None`. Number of boundaries has to match with the number of parameters. They will match
            with respect to the index within the list.
        fixed_poi_value (``Optional[float]``, default ``None``): If `float` the poi index will be fixed.
        logpdf (``Optional[Callable[[List[float]], float]]``, default ``None``): If provided, log-probability
            distribution will be computed once the parameters are fitted. If not,  the result of the ``func`` input
            will be returned.
        constraints (``Optional[List[Dict]]``, default ``None``): Constraints of the model. see scipy for details.
        options (``Dict``): extra options for the optimiser. see scipy minimiser for details.

    Returns:
        ``Tuple[float, np.ndarray]``:
        value and the parameters. log-probability value will be returned if `logpdf` argument is provided.
    """

    init_pars = [*(initial_parameters or model_configuration.suggested_init)]
    par_bounds = [*(bounds or model_configuration.fixed_poi_bounds(fixed_poi_value))]

    minimizer_opt = options.pop(
        "minimizer", os.environ.get("SPEY_OPTIMISER", "scipy").lower()
    )
    if minimizer_opt not in ["scipy", "minuit"]:
        log.warning(f"Invalid minimizer: {minimizer_opt}, using scipy")
        minimizer_opt = "scipy"
    log.debug(f"Minimiser: {minimizer_opt}")

    fixed_vals = [False] * len(init_pars)
    if fixed_poi_value is not None:
        init_pars[model_configuration.poi_index] = fixed_poi_value
        fixed_vals[model_configuration.poi_index] = True

    if model_configuration.suggested_fixed is not None:
        for idx, isfixed in enumerate(model_configuration.suggested_fixed):
            if isfixed:
                fixed_vals[idx] = True

    options.update({"poi_index": model_configuration.poi_index})

    fun, x = _get_minimizer(minimizer_opt)(
        func=func,
        init_pars=init_pars,
        fixed_vals=fixed_vals,
        do_grad=do_grad,
        hessian=hessian,
        bounds=par_bounds,
        constraints=[] if constraints is None else constraints,
        **options,
    )

    return fun if logpdf is None else logpdf(x), x
