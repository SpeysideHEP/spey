r"""
Theoretical Uncertainties
=========================

This module implements the **log-normal morphing** scheme used to propagate signal
systematic uncertainties through the simplified-likelihood backends.

Motivation
----------

Simplified likelihood backends parametrise background uncertainties through nuisance
parameters :math:`\boldsymbol{\theta}` that enter the Poisson mean :math:`\lambda_i`.
Signal yields, however, are often treated as fixed.  When the signal itself carries
non-negligible uncertainties (e.g. from theoretical scale variations, PDF
uncertainties, or detector modelling), those uncertainties must also be profiled.

The synthesizer introduces additional nuisance parameters per uncertainty *source*
and applies a multiplicative correction to the signal yields.  This keeps the signal
positive for all nuisance values and is consistent with the log-normal morphing
conventions widely used in HEP (e.g. HistFactory).

Two morphing modes are supported, selected via the ``"type"`` key of each modifier
configuration dictionary:

* **normalization** — one nuisance parameter :math:`\theta_k` shared across all bins.
  The factor :math:`f_{i,k}(\theta_k)` is the same for every bin modulo the
  per-bin fractional variation :math:`\Delta_{i,k}`.  Use this for *correlated*
  uncertainties such as PDF variations or luminosity, where a single degree of
  freedom simultaneously shifts all bins.

* **shape** — one independent nuisance parameter :math:`\alpha_{i,k}` per bin.
  Each bin is shifted independently.  Use this for *uncorrelated* uncertainties
  such as statistical uncertainties on theory predictions or scale uncertainties
  that affect each bin differently.

Log-normal morphing
-------------------

For each uncertainty source :math:`k` and bin :math:`i`, a fractional variation
:math:`\Delta_{i,k}` is defined as

.. math::

    \Delta_{i,k} = \frac{\sigma^{(s)}_{i,k}}{n^{(s)}_i},

where :math:`n^{(s)}_i` is the nominal signal yield and :math:`\sigma^{(s)}_{i,k}`
is the absolute uncertainty from source :math:`k` in bin :math:`i`.

**Normalization modifier** (``"type": "normalization"``)

A single nuisance parameter :math:`\theta_k` is shared across all bins.  The morphing
factor for source :math:`k` is

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

**Shape modifier** (``"type": "shape"``)

Each bin :math:`i` has its own independent nuisance parameter :math:`\alpha_{i,k}`,
so the morphing factor is

.. math::

    f_{i,k}(\alpha_{i,k})
    = \exp\!\left[\alpha_{i,k}
      \ln\!\bigl(1 + \Delta_{i,k}(\alpha_{i,k})\bigr)\right],

with :math:`\Delta_{i,k}(\alpha_{i,k})` chosen by the sign of :math:`\alpha_{i,k}` as
above.  The per-bin parameters are uncorrelated; each is constrained by an independent
standard normal prior.

**Combined modifier**

When multiple sources are present the total signal modifier is the product of the
individual morphing factors:

.. math::

    f_i(\boldsymbol{\theta}_{\rm sig})
    = \prod_k f_{i,k}
    = \exp\!\left[\sum_k \ln f_{i,k}\right].

The effective signal yield in bin :math:`i` thus becomes
:math:`\mu\, n^{(s)}_i \cdot f_i(\boldsymbol{\theta}_{\rm sig})`.

Each nuisance parameter is constrained by a standard normal prior
:math:`\mathcal{N}(\theta \mid 0, 1)`, which is added to the constraint model
of the calling backend.

Parameter layout
----------------

Signal uncertainty parameters are appended to the end of the parameter vector after
the background nuisance parameters::

    pars = [μ, θ₁, …, θ_N,  θ_{sig,1}, …]

For a **normalization** modifier, one parameter is appended; for a **shape** modifier,
:math:`N_{\rm bins}` parameters are appended (one per bin).  The ``domain`` field in
each constraint dictionary records the index of the corresponding parameter in this
vector.

Bins with zero nominal signal yield (:math:`n^{(s)}_i = 0`) produce
:math:`\Delta_{i,k} = \mathrm{NaN}`, which is replaced by 1 (no modification)
so the exponential evaluates to 1 for those bins.

.. currentmodule:: spey.backends.default_pdf.uncertainty_synthesizer

.. autosummary::
    :toctree: ../_generated/

    signal_uncertainty_synthesizer

Why there are two choices of modifiers?
---------------------------------------

This module is designed with PDF and scale uncertainties in mind. Neither PDF
uncertainties nor QCD scale uncertainties are typically pure normalisation uncertainties.
Both originate from the theoretical calculation of the signal cross-section and can alter
the shape of any differential distribution used as a discriminating observable.

**PDF uncertainties** come from the parametric uncertainty in the parton distribution functions.
Because different parton flavours and momentum fractions dominate different kinematic regimes,
varying PDF eigenvectors or replicas (following the PDF4LHC prescription in :xref:`1510.03865`)
changes both the inclusive cross-section and the relative contributions of different kinematic
regions. In a multi-bin fit over, say, invariant mass or MET, PDF variations therefore produce
correlated, bin-dependent rate changes that affect the shape of the template.

**QCD scale uncertainties** (:math:`\mu_R` and :math:`\mu_F` variations by conventional factors
of 2) are proxies for missing higher-order corrections. Because fixed-order corrections are
themselves kinematic-dependent, e.g., they grow with the hardness of the event, scale variations
tend to affect the high-end tail of a distribution differently from the bulk. This is well
documented: for example, ATLAS measurements of t-tbar differential distributions
:xref:`1601.03413` and inclusive cross-section analyses consistently show bin-dependent
scale-variation envelopes.

In both cases, the dominant effect is often a normalisation shift, but the shape change is
physically real and in a multi-bin analysis, it is generally not safe to discard it.

References
----------

* J. Butterworth, S. Carrazza, et al. *PDF4LHC recommendations for LHC Run II*, J. Phys. G **43**
  (2016), 023001 :xref:`1510.03865`
* L. A. Harland-Lang, A. D. Martin, P. Motylinski and R. S. Thorne, *The impact of the final HERA
  combined data on PDFs obtained from a global fit*, Eur. Phys. J. C **76** (2016) no.4, 186,
  :xref:`1601.03413`

"""

import warnings
from functools import partial, reduce
from typing import Any, Dict, List, Optional

import autograd.numpy as np

from spey.system.exceptions import InvalidUncertaintyDefinition

# pylint: disable=E1101,E1120

#: Allowed modifier type strings.
MODIFIER_TYPES = ("normalization", "shape")


def signal_uncertainty_synthesizer(
    signal_yields: List[float],
    modifiers: List[Dict[str, Any]],
    n_signal_parameters: int = 0,
    domain_start: Optional[int] = None,
) -> Dict[str, Any]:
    r"""
    Synthesize signal uncertainties from a list of modifier configuration dictionaries.

    .. versionchanged:: 0.2.7
        ``modifiers`` is now a list of configuration dictionaries.  Each dictionary
        specifies the modifier type (``"normalization"`` or ``"shape"``), the
        per-bin uncertainty values, and an optional name.  The old list-of-arrays
        format is no longer accepted.

    Each modifier applies a multiplicative correction to the signal yields via
    log-normal morphing.  Two morphing modes are supported:

    **Normalization modifier** (``"type": "normalization"``)

    A *single* nuisance parameter :math:`\theta_k` is shared across all bins.
    Suitable for uncertainties that rescale the entire distribution coherently,
    such as PDF variations, luminosity uncertainties, or cross-section
    normalisation errors.

    The morphing factor is

    .. math::

        f_{i,k}(\theta_k)
        = \exp\!\left[\theta_k \ln\!\bigl(1 + \Delta_{i,k}(\theta_k)\bigr)\right],

    where :math:`\Delta_{i,k} = \sigma^{(s)}_{i,k} / n^{(s)}_i` and the sign
    of :math:`\theta_k` selects the up or down variation.

    **Shape modifier** (``"type": "shape"``)

    One *independent* nuisance parameter :math:`\alpha_{i,k}` per bin.  Suitable
    for uncertainties whose effect differs bin-by-bin in an unknown way, such as
    statistical uncertainties on theory predictions or bin-by-bin scale
    uncertainties.

    The morphing factor is

    .. math::

        f_{i,k}(\alpha_{i,k})
        = \exp\!\left[\alpha_{i,k}
            \ln\!\bigl(1 + \Delta_{i,k}(\alpha_{i,k})\bigr)\right],

    with each :math:`\alpha_{i,k}` constrained by an independent standard normal
    prior :math:`\mathcal{N}(\alpha_{i,k} \mid 0, 1)`.

    For asymmetric uncertainties the variation depends on the sign of the nuisance
    parameter:

    .. math::

        \Delta_{i,k}(\theta) =
        \begin{cases}
            \Delta_{i,k}^{+}, & \theta \ge 0 \\
            \Delta_{i,k}^{-}, & \theta < 0
        \end{cases}

    All nuisance parameters are constrained by standard normal priors.

    .. note::

        Symmetric uncertainties must be provided as ``list[float]`` in
        ``"uncertainties"``.  Asymmetric uncertainties must be provided as
        ``list[tuple[float, float]]``, where each tuple is ``(up, down)``
        giving the absolute up and down variations.

    **Modifier configuration dictionary**

    Each element of ``modifiers`` is a :class:`dict` with the following keys:

    .. list-table::
        :header-rows: 1
        :widths: 20 15 65

        * - Key
          - Required
          - Description
        * - ``"type"``
          - yes
          - Morphing mode: ``"normalization"`` (one shared nuisance) or
            ``"shape"`` (one nuisance per bin).
        * - ``"uncertainties"``
          - yes
          - Per-bin absolute uncertainties.  Either a ``list[float]`` for
            symmetric uncertainties or a ``list[tuple[float, float]]`` for
            asymmetric ``(up, down)`` variations.  Must have one entry per bin.
        * - ``"name"``
          - no
          - Human-readable label used to construct parameter names
            (e.g. ``"pdf"`` → ``theta_sig_pdf``).  Defaults to
            ``"mod0"``, ``"mod1"``, … when omitted.

    **Parameter layout with** ``n_signal_parameters``

    When the calling backend uses a callable ``signal_yields`` that accepts
    ``n_signal_parameters`` additional free parameters, those parameters occupy
    indices ``1 … n_signal_parameters`` in the full parameter vector and push
    the background nuisance parameters (and therefore the signal-uncertainty
    parameters) to higher indices::

        pars = [μ, sig_par_0, …, sig_par_{n-1}, θ_bkg_1, …, θ_bkg_N, θ_sig_1, …]

    Pass ``n_signal_parameters`` so that the domain indices are shifted
    accordingly.  Alternatively, supply ``domain_start`` to override the
    auto-computed starting index.

    **Examples**

    *Normalization modifier* — a single nuisance scales the whole distribution:

    .. code:: python3

        >>> signal_uncertainty_synthesizer(
        ...     signal_yields=[3.0, 5.0],
        ...     modifiers=[
        ...         {"type": "normalization", "name": "pdf", "uncertainties": [0.6, 1.0]},
        ...     ],
        ... )
        # one nuisance parameter theta_sig_pdf at domain index 3

    *Shape modifier* — independent nuisance per bin:

    .. code:: python3

        >>> signal_uncertainty_synthesizer(
        ...     signal_yields=[3.0, 5.0],
        ...     modifiers=[
        ...         {"type": "shape", "name": "scale", "uncertainties": [0.3, 0.5]},
        ...     ],
        ... )
        # two nuisance parameters theta_sig_scale_0, theta_sig_scale_1 at domain
        # indices 3 and 4

    *Mixed modifiers* — normalization + shape in one model:

    .. code:: python3

        >>> signal_uncertainty_synthesizer(
        ...     signal_yields=[3.0, 5.0],
        ...     modifiers=[
        ...         {"type": "normalization", "name": "pdf",
        ...          "uncertainties": [0.6, 1.0]},
        ...         {"type": "shape", "name": "scale",
        ...          "uncertainties": [(0.3, 0.4), (0.5, 0.6)]},
        ...     ],
        ... )
        # 3 nuisance parameters total: theta_sig_pdf (index 3),
        # theta_sig_scale_0 (index 4), theta_sig_scale_1 (index 5)

    Args:
        signal_yields (``List[float]``): List of nominal signal yields per bin.  Used
            to compute the fractional variation :math:`\Delta_{i,k}`.
        modifiers (``List[Dict[str, Any]]``): List of modifier configuration
            dictionaries.  Each dictionary must contain ``"type"`` and
            ``"uncertainties"`` keys; ``"name"`` is optional.  See table above.
        n_signal_parameters (``int``, default ``0``): Number of additional free
            parameters that a callable ``signal_yields`` function accepts.  These
            parameters are placed *before* the background nuisance parameters in the
            parameter vector (immediately after :math:`\mu`), so the domain indices
            of the signal-uncertainty parameters are shifted by this amount.
            Has no effect when ``domain_start`` is supplied explicitly.
        domain_start (``int``, default ``None``): Explicit starting index in the
            parameter vector from which signal-uncertainty nuisance parameters are
            assigned.  When ``None`` the index is computed automatically as
            ``1 + n_signal_parameters + N_bins``.

    Raises:
        :exc:`~spey.system.exceptions.InvalidUncertaintyDefinition`: If a modifier
            has an unknown ``"type"``, if the number of bins in ``"uncertainties"``
            does not match ``signal_yields``, or if the uncertainty array has an
            unsupported number of dimensions.

    Returns:
        A dictionary with

        * ``"lambda"`` — callable ``(pars: np.ndarray) -> np.ndarray`` that computes
          the per-bin signal modifier :math:`f_i(\boldsymbol{\theta}_{\rm sig})`.
        * ``"constraint"`` — list of constraint dictionaries (one per nuisance
          parameter) for the constraint model.
        * ``"n_parameters"`` — total number of signal-uncertainty nuisance parameters
          introduced.  This is 1 per normalization modifier and :math:`N_{\rm bins}`
          per shape modifier.
        * ``"parameter_names"`` — list of parameter name strings corresponding to
          the signal-uncertainty nuisance parameters, in parameter-vector order.
    """
    n_bins = len(signal_yields)
    signal_yields_arr = np.array(signal_yields, dtype=float)

    lambdas = []
    constraints = []
    parameter_names = []

    if domain_start is None:
        current_idx = 1 + n_signal_parameters + n_bins
    else:
        current_idx = int(domain_start)

    initial_idx = current_idx

    for mod_idx, modifier in enumerate(modifiers):
        mod_type = modifier.get("type")
        if mod_type not in MODIFIER_TYPES:
            raise InvalidUncertaintyDefinition(
                f"Unknown modifier type '{mod_type}'. "
                f"Expected one of {MODIFIER_TYPES}."
            )

        values = np.array(modifier["uncertainties"])
        mod_name = modifier.get("name", f"mod{mod_idx}")

        if len(values) != n_bins:
            raise InvalidUncertaintyDefinition(
                f"Modifier '{mod_name}': expected {n_bins} bins, got {len(values)}"
            )

        with warnings.catch_warnings(record=True):
            if values.ndim == 1:
                delta_up = 1.0 + values / signal_yields_arr
                delta_dn = delta_up
            elif values.ndim == 2:
                delta_up = 1.0 + values[:, 0] / signal_yields_arr
                delta_dn = 1.0 + values[:, 1] / signal_yields_arr
            else:
                raise InvalidUncertaintyDefinition(
                    f"Modifier '{mod_name}': unsupported uncertainty shape "
                    f"({values.ndim}D array). "
                    "Expected 1D for symmetric or 2D for asymmetric (up, down) uncertainties."
                )

        delta_up = np.where(np.isnan(delta_up), 1.0, delta_up)
        delta_dn = np.where(np.isnan(delta_dn), 1.0, delta_dn)
        log_up = np.log(delta_up)
        log_dn = np.log(delta_dn)

        if mod_type == "normalization":
            # One nuisance parameter shared across all bins
            par_idx = current_idx
            current_idx += 1
            parameter_names.append(f"theta_sig_{mod_name}")

            constraints.append(
                {
                    "distribution_type": "normal",
                    "args": [np.zeros(1), np.ones(1)],
                    "kwargs": {"domain": np.r_[par_idx]},
                }
            )

            def _lam_norm(
                param: np.ndarray, lu: np.ndarray, ld: np.ndarray, pidx: int
            ) -> np.ndarray:
                alpha = param[pidx]
                return alpha * (lu if alpha > 0 else ld)

            lambdas.append(partial(_lam_norm, lu=log_up, ld=log_dn, pidx=par_idx))

        else:  # mod_type == "shape"
            # One independent nuisance parameter per bin
            bin_indices = np.arange(current_idx, current_idx + n_bins)
            current_idx += n_bins
            parameter_names.extend([f"theta_sig_{mod_name}_{i}" for i in range(n_bins)])

            for par_idx in bin_indices:
                constraints.append(
                    {
                        "distribution_type": "normal",
                        "args": [np.zeros(1), np.ones(1)],
                        "kwargs": {"domain": np.r_[par_idx]},
                    }
                )

            def _lam_shape(
                param: np.ndarray,
                lu: np.ndarray,
                ld: np.ndarray,
                indices: np.ndarray,
            ) -> np.ndarray:
                alphas = param[indices]
                chosen_log = np.where(alphas >= 0, lu, ld)
                return alphas * chosen_log

            lambdas.append(partial(_lam_shape, lu=log_up, ld=log_dn, indices=bin_indices))

    n_parameters = current_idx - initial_idx

    if not lambdas:
        return {
            "lambda": lambda _: 1.0,
            "constraint": [],
            "n_parameters": 0,
            "parameter_names": [],
        }

    if len(lambdas) == 1:
        lam_fn = lambda param: np.exp(lambdas[0](param))  # noqa: E731
    else:

        def lam_fn(param: np.ndarray) -> np.ndarray:
            return np.exp(reduce(lambda x, y: x + y, (lam(param) for lam in lambdas)))

    return {
        "lambda": lam_fn,
        "constraint": constraints,
        "n_parameters": n_parameters,
        "parameter_names": parameter_names,
    }
