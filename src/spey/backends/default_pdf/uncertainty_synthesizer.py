from functools import partial, reduce
from typing import Dict, List, Tuple, Union

import autograd.numpy as np
from spey.system.exceptions import InvalidUncertaintyDefinition

# pylint: disable=E1101,E1120


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
        InvalidUncertaintyDefinition: If the number of bins in modifiers does not match signal_yields
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

        if values.ndim == 1:

            def lam_signal(param: np.ndarray, values: np.ndarray, idx: int) -> np.ndarray:
                return 1.0 + values * param[domain[idx]]

        elif values.ndim == 2:

            def lam_signal(param: np.ndarray, values: np.ndarray, idx: int) -> np.ndarray:
                return 1.0 + (
                    values[:, 0 if param[domain[idx]] >= 0.0 else 1] * param[domain[idx]]
                )

        else:
            raise InvalidUncertaintyDefinition(
                f"Unsupported number of uncertainty modifiers: {values.ndim}, "
                "expected 1 for symmetric or 2 for asymmetric uncertainties"
            )

        lambdas.append(partial(lam_signal, values=values, idx=idx))

    if nnui == 1:
        return {"lambda": lambdas[0], "constraint": constraints}

    def lam_signal_total(param: np.ndarray) -> np.ndarray:
        return (
            reduce(lambda x, y: x * y, (lam(param) for lam in lambdas)) - 1.0
        ) * param[0]

    return {"lambda": lam_signal_total, "constraint": constraints}
