import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from spey.base.model_config import ModelConfig

from .scipy_tools import minimize

# pylint: disable=W1203

log = logging.getLogger("Spey")


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

    def make_constraint(index: int, value: float) -> Callable[[np.ndarray], float]:
        def func(vector: np.ndarray) -> float:
            return vector[index] - value

        return func

    constraints = [] if constraints is None else constraints
    if fixed_poi_value is not None:
        init_pars[model_configuration.poi_index] = fixed_poi_value
        log.debug(
            f"Fixing POI index at: {model_configuration.poi_index} to value {fixed_poi_value}"
        )
        constraints.append(
            {
                "type": "eq",
                "fun": make_constraint(model_configuration.poi_index, fixed_poi_value),
            }
        )

    if model_configuration.suggested_fixed is not None:
        for idx, isfixed in enumerate(model_configuration.suggested_fixed):
            if isfixed:
                log.debug(f"Constraining {idx} to value {init_pars[idx]}")
                constraints.append(
                    {
                        "type": "eq",
                        "fun": make_constraint(idx, init_pars[idx]),
                    }
                )

    options.update({"poi_index": model_configuration.poi_index})

    fun, x = minimize(
        func=func,
        init_pars=init_pars,
        do_grad=do_grad,
        hessian=hessian,
        bounds=par_bounds,
        constraints=constraints,
        **options,
    )

    return fun if logpdf is None else logpdf(x), x
