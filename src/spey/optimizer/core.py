import logging
import os
from importlib.util import find_spec
from typing import Callable, Dict, List, Optional, Tuple, Union

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
    fixed_poi_value: Optional[Union[float, Dict[int, float]]] = None,
    logpdf: Optional[Callable[[List[float]], float]] = None,
    constraints: Optional[List[Dict]] = None,
    **options,
) -> Tuple[float, np.ndarray]:
    """
    Dispatch a fit to either the scipy or minuit minimiser.

    The minimiser is selected by the ``minimizer`` key in ``options`` (defaults
    to the ``SPEY_OPTIMISER`` environment variable, or ``"scipy"``).
    ``fixed_poi_value`` may be a ``float`` (pins the primary POI) or a ``dict``
    of ``{index: value}`` (pins multiple parameters); pinned parameters are
    appended to the optimiser's fixed-mask via the underlying minimiser.

    Args:
        func (``Callable[[np.ndarray], np.ndarray]``): Function to be optimised.
            If ``do_grad=True``, the function must return
            ``Tuple[float, np.ndarray]`` containing the value and its gradient
            with respect to the variational parameters.
        model_configuration (:obj:`~spey.base.model_config.ModelConfig`): Model
            configuration providing POI index, suggested initialisation, and
            suggested bounds.
        do_grad (``bool``, default ``False``): Whether ``func`` returns its
            gradient alongside the value.
        hessian (``Optional[Callable[[np.ndarray], np.ndarray]]``, default ``None``):
            Hessian forwarded to scipy as the ``hess`` keyword. Ignored by
            minuit, which estimates the Hessian internally.
        initial_parameters (``Optional[np.ndarray]``, default ``None``): Initial
            guess. When ``None``, falls back to
            :attr:`~spey.base.model_config.ModelConfig.suggested_init`.
        bounds (``Optional[List[Tuple[float, float]]]``, default ``None``): Per-
            parameter ``(lower, upper)`` bounds (use ``None`` for an open side).
            When ``None``, bounds are derived from the model configuration plus
            the optional POI-fixing logic.
        fixed_poi_value (``Optional[Union[float, Dict[int, float]]]``, default ``None``):
            If a ``float``, pins the model's primary POI. If a ``dict``, pins
            every listed ``{index: value}``.
        logpdf (``Optional[Callable[[List[float]], float]]``, default ``None``):
            If provided, the function value returned is replaced by
            ``logpdf(fit_parameters)`` (typical use: return the log-likelihood
            rather than the twice-NLL passed in as ``func``).
        constraints (``Optional[List[Dict]]``, default ``None``): Extra
            scipy-style constraint dicts forwarded to the underlying minimiser.
        **options: Forwarded to the selected minimiser; see
            :func:`spey.optimizer.scipy_tools.minimize` and
            :func:`spey.optimizer.minuit_tools.minimize` for accepted keys.
            ``minimizer`` selects the backend.

    Returns:
        ``Tuple[float, np.ndarray]``:
        The objective value (or ``logpdf(fit_parameters)`` when ``logpdf`` is
        provided) and the corresponding fit parameter vector.
    """

    init_pars = (
        initial_parameters
        if isinstance(initial_parameters, (list, tuple, np.ndarray))
        else model_configuration.suggested_init
    )

    if bounds is not None:
        par_bounds = [*bounds]
    elif isinstance(fixed_poi_value, dict):
        par_bounds = model_configuration.fixed_poi_bounds_multi(fixed_poi_value)
    else:
        par_bounds = model_configuration.fixed_poi_bounds(fixed_poi_value)

    minimizer_opt = options.pop(
        "minimizer", os.environ.get("SPEY_OPTIMISER", "scipy").lower()
    )
    if minimizer_opt not in ["scipy", "minuit"]:
        log.warning(f"Invalid minimizer: {minimizer_opt}, using scipy")
        minimizer_opt = "scipy"
    log.debug(f"Minimiser: {minimizer_opt}")

    fixed_vals = [False] * len(init_pars)
    if fixed_poi_value is not None:
        if isinstance(fixed_poi_value, dict):
            for idx, val in fixed_poi_value.items():
                init_pars[idx] = val
                fixed_vals[idx] = True
        else:
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
