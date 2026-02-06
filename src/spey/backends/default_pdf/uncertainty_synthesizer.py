import warnings
from functools import partial, reduce
from typing import Dict, List, Tuple, Union

import autograd.numpy as np

from spey.system.exceptions import InvalidUncertaintyDefinition

# pylint: disable=E1101,E1120


def nonlinear_interp(
    alpha: np.ndarray, delta_up: np.ndarray, delta_dn: np.ndarray
) -> np.ndarray:
    """
    nonlinear interpolator

    Args:
        alpha (``np.ndarray``): nuisance parameter
        delta_up (``np.ndarray``): upper deviation
        delta_dn (``np.ndarray``): lower deviation
    """
    if alpha >= 0:
        return delta_up**alpha
    return delta_dn ** (-alpha)


def signal_uncertainty_synthesizer(
    signal_yields: List[float],
    modifiers: List[Union[List[float], List[Tuple[float, float]]]],
) -> Dict[str, np.ndarray]:
    """
    Synthesize signal uncertainties

    Args:
        signal_yields (`List[float]`): List of signal yields per bin
        modifiers (`List[Union[List[float], List[Tuple[float, float]]]]`):
            List of uncertainty modifiers, each can be either:
            - A list of floats representing symmetric uncertainties per bin
            - A list of tuples representing asymmetric uncertainties (up, down) per bin

    Raises:
        InvalidUncertaintyDefinition: If the number of bins in modifiers does not match
            `signal_yields`
        AssertionError: If the number of bins in modifiers is inconsistent

    Returns:
        A dictionary with:
            - "lambda": A callable that computes the modified signal yields given parameters
            - "constraint": A list of constraint dictionaries for each uncertainty modifier
    """

    nnui = len(modifiers)
    lambdas = []
    constraints = []
    domain = np.r_[len(signal_yields) + 1 : len(signal_yields) + 1 + nnui]

    for idx, values in enumerate(modifiers):
        values = np.array(values)

        assert len(values) == len(
            signal_yields
        ), f"Inconsistent number of bins, expected {len(signal_yields)}, got {len(values)}"

        constraints.append(
            {
                "distribution_type": "normal",
                "args": [np.zeros(1), np.ones(1)],
                "kwargs": {"domain": np.r_[domain[idx]]},
            }
        )

        with warnings.catch_warnings(record=True):
            if values.ndim == 1:
                delta_up = (signal_yields + values) / signal_yields
                delta_dn = (signal_yields - values) / signal_yields
            elif values.ndim == 2:
                delta_up = (signal_yields + values[:, 0]) / signal_yields
                delta_dn = (signal_yields - values[:, 1]) / signal_yields
            else:
                raise InvalidUncertaintyDefinition(
                    f"Unsupported number of uncertainty modifiers: {values.ndim}, "
                    "expected 1 for symmetric or 2 for asymmetric uncertainties"
                )

        delta_up = np.where(np.isnan(delta_up), 1.0, delta_up)
        delta_dn = np.where(np.isnan(delta_dn), 1.0, delta_dn)

        def lam_signal(
            param: np.ndarray, up: np.ndarray, dn: np.ndarray, idx: int
        ) -> np.ndarray:
            return nonlinear_interp(param[domain[idx]], up, dn)

        lambdas.append(partial(lam_signal, up=delta_up, dn=delta_dn, idx=idx))

    if nnui == 1:
        return {"lambda": lambdas[0], "constraint": constraints}

    def lam_signal_total(param: np.ndarray) -> np.ndarray:
        return reduce(lambda x, y: x * y, (lam(param) for lam in lambdas))

    return {"lambda": lam_signal_total, "constraint": constraints}
