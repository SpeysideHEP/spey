import warnings
from functools import partial, reduce
from typing import Any, Dict, List, Tuple, Union

import autograd.numpy as np

from spey.system.exceptions import InvalidUncertaintyDefinition

# pylint: disable=E1101,E1120


def signal_uncertainty_synthesizer(
    signal_yields: List[float],
    modifiers: List[Union[List[float], List[Tuple[float, float]]]],
) -> Dict[str, Any]:
    r"""
    Synthesize signal uncertainties

    .. versionchanged:: 0.2.7

    This function applies per-bin signal uncertainty modifiers through a
    multiplicative morphing factor defined as

    .. math::

        f_{i,k}(\theta_{i,k}) = \exp\!\left[
            \theta_{i,k} \log\!\bigl(1+\Delta_{i,k}(\theta_{i,k})\bigr)
        \right].

    Here :math:`i` indexes histogram bins and :math:`k` labels the uncertainty
    sources defined in the ``modifiers`` argument. The fractional signal variation
    is defined as

    .. math::

        \Delta_{i,k} = \frac{\sigma^{(s)}_{i,k}}{n^{(s)}_i},

    where :math:`n^{(s)}_i` is the nominal signal yield and
    :math:`\sigma^{(s)}_{i,k}` is the absolute uncertainty associated with source
    :math:`k` in bin :math:`i`.

    For asymmetric uncertainties the variation depends on the sign of the nuisance
    parameter:

    .. math::

        \Delta_{i,k}(\theta_{i,k}) =
        \begin{cases}
            \Delta_{i,k}^{+}, & \theta_{i,k} \ge 0 \\
            \Delta_{i,k}^{-}, & \theta_{i,k} < 0
        \end{cases}

    Each nuisance parameter is constrained by a standard normal prior,

    .. math::

        \mathcal{N}(\theta_{i,k}\mid 0,1).

    .. note::

        Symmetric uncertainties must be provided as ``list[float]`` entries in
        ``modifiers``. Asymmetric uncertainties must be provided as
        ``list[tuple[float, float]]``, corresponding to ``(down, up)`` variations.

    **Example:**

    Consider a single-bin signal yield

    .. math::

        n^{(s)} = 3 \pm 2^{+1}_{-1.5},

    which contains two independent uncertainty sources:
    one symmetric (:math:`\pm2`) and one asymmetric (:math:`^{+1}_{-1.5}`).

    The corresponding ``modifiers`` input should therefore contain
    one float entry and one tuple entry representing these two sources.

    .. code:: python3

        >>> signal_uncertainty_synthesizer([3.0], [[2.0], [(1.0, 1.5)]])
        >>> # {'constraint': [{'args': [array([0.]), array([1.])],
        ... #                  'distribution_type': 'normal',
        ... #                  'kwargs': {'domain': array([2])}},
        ... #                 {'args': [array([0.]), array([1.])],
        ... #                  'distribution_type': 'normal',
        ... #                  'kwargs': {'domain': array([3])}}],
        ... #  'lambda': <function signal_uncertainty_synthesizer.<locals>.lam_signal_total at 0x14f530c10>}

    Args:
        signal_yields (`List[float]`): List of signal yields per bin
        modifiers (`List[Union[List[float], List[Tuple[float, float]]]]`):
            List of uncertainty modifiers, each can be either:

            * A list of floats representing symmetric uncertainties per bin
            * A list of tuples representing asymmetric uncertainties (up, down) per bin

    Raises:
        InvalidUncertaintyDefinition: If the number of bins in modifiers does not match
            `signal_yields`
        AssertionError: If the number of bins in modifiers is inconsistent

    Returns:
        A dictionary with

        * "lambda" A callable that computes the modified signal yields given parameters
        * "constraint" A list of constraint dictionaries for each uncertainty modifier.

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
                delta_up = 1.0 + values / signal_yields
                delta_dn = delta_up
            elif values.ndim == 2:
                delta_up = 1.0 + values[:, 0] / signal_yields
                delta_dn = 1.0 + values[:, 1] / signal_yields
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
            alpha = param[domain[idx]]
            return alpha * np.log(up if alpha > 0 else dn)

        lambdas.append(partial(lam_signal, up=delta_up, dn=delta_dn, idx=idx))

    if nnui == 1:
        return {
            "lambda": lambda param: np.exp(lambdas[0](param)),
            "constraint": constraints,
        }

    def lam_signal_total(param: np.ndarray) -> np.ndarray:
        return np.exp(reduce(lambda x, y: x + y, (lam(param) for lam in lambdas)))

    return {"lambda": lam_signal_total, "constraint": constraints}
