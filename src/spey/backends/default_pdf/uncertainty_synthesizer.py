r"""
Signal Uncertainty Synthesizer
================================

This module implements the **log-normal morphing** scheme used to propagate signal
systematic uncertainties through the simplified-likelihood backends.

Motivation
----------

Simplified likelihood backends parametrise background uncertainties through nuisance
parameters :math:`\boldsymbol{\theta}` that enter the Poisson mean :math:`\lambda_i`.
Signal yields, however, are often treated as fixed.  When the signal itself carries
non-negligible uncertainties (e.g. from theoretical scale variations, PDF
uncertainties, or detector modelling), those uncertainties must also be profiled.

The synthesizer introduces one additional nuisance parameter :math:`\theta_k` per
uncertainty *source* :math:`k` and applies a multiplicative correction to the signal
yields.  This keeps the signal positive for all nuisance values and is consistent
with the log-normal morphing conventions widely used in HEP (e.g. HistFactory).

Log-normal morphing
-------------------

For each uncertainty source :math:`k` and bin :math:`i`, a fractional variation
:math:`\Delta_{i,k}` is defined as

.. math::

    \Delta_{i,k} = \frac{\sigma^{(s)}_{i,k}}{n^{(s)}_i},

where :math:`n^{(s)}_i` is the nominal signal yield and :math:`\sigma^{(s)}_{i,k}`
is the absolute uncertainty from source :math:`k` in bin :math:`i`.

The morphing factor for source :math:`k` is then

.. math::

    f_{i,k}(\theta_k)
    = \exp\!\left[\theta_k \ln\!\bigl(1 + \Delta_{i,k}(\theta_k)\bigr)\right],

where the effective variation depends on the sign of :math:`\theta_k`:

.. math::

    \Delta_{i,k}(\theta_k) =
    \begin{cases}
        \Delta_{i,k}^{+}, & \theta_k \geq 0 \\
        \Delta_{i,k}^{-}, & \theta_k < 0
    \end{cases}

(symmetric uncertainties use :math:`\Delta^+ = \Delta^-`).

When multiple sources are present the total signal modifier is the product of the
individual morphing factors:

.. math::

    f_i(\boldsymbol{\theta}_{\rm sig})
    = \prod_k f_{i,k}(\theta_k)
    = \exp\!\left[\sum_k \theta_k \ln\!\bigl(1 + \Delta_{i,k}(\theta_k)\bigr)\right].

The effective signal yield in bin :math:`i` thus becomes
:math:`\mu\, n^{(s)}_i \cdot f_i(\boldsymbol{\theta}_{\rm sig})`.

Each nuisance parameter :math:`\theta_k` is constrained by a standard normal prior
:math:`\mathcal{N}(\theta_k \mid 0, 1)`, which is added to the constraint model
of the calling backend.

Parameter layout
----------------

Signal uncertainty parameters are appended to the end of the parameter vector after
the background nuisance parameters::

    pars = [μ, θ₁, …, θ_N,  θ_{sig,1}, …, θ_{sig,K}]

The ``domain`` field in each constraint dictionary records the index of the
corresponding parameter in this vector.

Bins with zero nominal signal yield (:math:`n^{(s)}_i = 0`) produce
:math:`\Delta_{i,k} = \mathrm{NaN}`, which is replaced by 1 (no modification)
so the exponential evaluates to 1 for those bins.
"""

import warnings
from functools import partial, reduce
from typing import Any, Dict, List, Tuple, Union

import autograd.numpy as np

from spey.system.exceptions import InvalidUncertaintyDefinition

# pylint: disable=E1101,E1120


def signal_uncertainty_synthesizer(
    signal_yields: List[float],
    modifiers: List[Union[List[float], List[Tuple[float, float]]]],
    n_signal_parameters: int = 0,
    domain: np.ndarray = None,
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

    **Parameter layout with** ``n_signal_parameters``

    When the calling backend uses a callable ``signal_yields`` that accepts
    ``n_signal_parameters`` additional free parameters, those parameters occupy
    indices ``1 … n_signal_parameters`` in the full parameter vector and push
    the background nuisance parameters (and therefore the signal-uncertainty
    parameters) to higher indices::

        pars = [μ, sig_par_0, …, sig_par_{n-1}, θ_bkg_1, …, θ_bkg_N, θ_sig_1, …]

    Pass ``n_signal_parameters`` so that the domain indices are shifted
    accordingly.  Alternatively, supply ``domain`` directly to override the
    auto-computed indices.

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
        ... #  'lambda': <function ...lam_signal_total at 0x...>}

    Args:
        signal_yields (`List[float]`): List of nominal signal yields per bin.  Used
            to compute the fractional variation :math:`\Delta_{i,k}`.
        modifiers (`List[Union[List[float], List[Tuple[float, float]]]]`):
            List of uncertainty modifiers, each can be either:

            * A list of floats representing symmetric uncertainties per bin
            * A list of tuples representing asymmetric uncertainties (up, down) per bin

        n_signal_parameters (`int`, default ``0``): Number of additional free
            parameters that a callable ``signal_yields`` function accepts.  These
            parameters are placed *before* the background nuisance parameters in the
            parameter vector (immediately after :math:`\mu`), so the domain indices
            of the signal-uncertainty parameters are shifted by this amount.
            Has no effect when ``domain`` is supplied explicitly.
        domain (`np.ndarray`, default ``None``): Explicit array of parameter-vector
            indices at which the signal-uncertainty nuisance parameters live.  When
            ``None`` the indices are computed automatically as
            ``[1 + n_signal_parameters + N, …, 1 + n_signal_parameters + N + K - 1]``
            where :math:`N` is the number of bins and :math:`K` is the number of
            modifiers.

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
    if domain is None:
        start = 1 + n_signal_parameters + len(signal_yields)
        domain = np.r_[start : start + nnui]
    else:
        domain = np.asarray(domain)

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
