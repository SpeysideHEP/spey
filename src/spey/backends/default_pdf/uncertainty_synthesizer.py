from functools import partial, reduce
from typing import Dict, List, Tuple, Union

import autograd.numpy as np

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
        ValueError: If the number of bins in modifiers does not match signal_yields
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
        assert len(values) == len(
            signal_yields
        ), f"Inconsistent number of bins, expected {len(signal_yields)}, got {len(values)}"

        if values.ndim == 1:
            values = np.array(values) / np.clip(np.array(signal_yields), 1e-5, None)

            def lam_signal(param: np.ndarray, values: np.ndarray, idx: int) -> np.ndarray:
                return 1.0 + values * param[domain[idx]]

            lambdas.append(partial(lam_signal, values=values, idx=idx))
            constraints.append(
                {
                    "distribution_type": "normal",
                    "args": [np.zeros(1), np.ones(1)],
                    "kwargs": {"domain": np.r_[domain[idx]]},
                }
            )

        elif values.ndim == 2:
            values = (
                np.array(values) / np.clip(np.array(signal_yields), 1e-5, None)[:, None]
            )

            def lam_signal(param: np.ndarray, values: np.ndarray, idx: int) -> np.ndarray:
                if np.all(np.greater_equal(domain[idx], 0.0)):
                    return 1.0 + values[:, 0] * param[domain[idx]]
                return 1.0 + values[:, 1] * param[domain[idx]]

            lambdas.append(partial(lam_signal, values=values, idx=idx))
            constraints.append(
                {
                    "distribution_type": "normal",
                    "args": [np.zeros(1), np.ones(1)],
                    "kwargs": {"domain": np.r_[domain[idx]]},
                }
            )

        else:
            raise ValueError(
                f"Unsupported number of uncertainty modifiers: {values.ndim}, expected 1 or 2"
            )

    if len(lambdas) == 1:
        return {"lambda": lambdas[0], "constraint": constraints}

    def lam_signal_total(param: np.ndarray) -> np.ndarray:
        return reduce(lambda x, y: x * y, (lam(param) for lam in lambdas))

    return {"lambda": lam_signal_total, "constraint": constraints}
