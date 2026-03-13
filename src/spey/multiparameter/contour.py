r"""
This module implements a two-stage algorithm for mapping the boundary of the
:math:`(1-\alpha)` confidence region in the full parameter space of a
:class:`~spey.StatisticalModel` whose backend is
``"default.multivariate_normal"``.

Mathematical Framework
======================

Let :math:`\theta \in \mathbb{R}^k` be the model parameter vector,
:math:`\hat\theta` the maximum-likelihood estimate (MLE), and

.. math::

    \mathrm{NLL}(\theta) = -\log\mathcal{L}(\theta)

the negative log-likelihood.  Under Wilks' theorem, the test statistic

.. math::

    \chi^2(\theta) = 2\bigl[\mathrm{NLL}(\theta) - \mathrm{NLL}(\hat\theta)\bigr]

follows a :math:`\chi^2_k` distribution asymptotically.  The
:math:`(1-\alpha)` **confidence region** is

.. math::

    \mathcal{C}_\alpha = \bigl\{\theta : \chi^2(\theta) \le \Delta_\alpha\bigr\},
    \qquad \Delta_\alpha = F^{-1}_{\chi^2_k}(1-\alpha),

and its :math:`(k-1)`-dimensional boundary — the **contour** — satisfies

.. math::

    \mathrm{NLL}(\theta) = T,
    \qquad T = \mathrm{NLL}(\hat\theta) + \tfrac{\Delta_\alpha}{2}.

Algorithm
=========

Stage 1 — Pre-whitening
-----------------------

The Hessian of the NLL at the MLE equals the observed Fisher information:

.. math::

    G = \nabla^2 \mathrm{NLL}(\hat\theta)
      = -\nabla^2 \log\mathcal{L}(\hat\theta).

:math:`G` is positive (semi-)definite at a proper minimum.  After Cholesky
factorisation :math:`G = LL^T`, the **whitened coordinate**

.. math::

    \varphi = L(\theta - \hat\theta),
    \qquad \theta = \hat\theta + L^{-T}\varphi

makes the contour approximately a :math:`(k-1)`-sphere of radius
:math:`\sqrt{\Delta_\alpha}`:

.. math::

    \mathrm{NLL}\!\bigl(\hat\theta + L^{-T}\varphi\bigr)
    \approx \mathrm{NLL}(\hat\theta) + \tfrac{1}{2}|\varphi|^2 + O(|\varphi|^3).

Sampling uniform random directions on :math:`S^{k-1}` in :math:`\varphi`-space
therefore gives approximately uniform coverage of the contour, even when the
original parameter space is strongly anisotropic.

Stage 2 — Radial search
------------------------

For each unit vector :math:`\hat{e} \in S^{k-1}` (drawn by normalising a
standard Gaussian), define the one-dimensional profile

.. math::

    f(r) = \mathrm{NLL}\!\bigl(\hat\theta + L^{-T}(r\hat{e})\bigr) - T.

:math:`f` is negative at :math:`r=0` (since :math:`\mathrm{NLL}(\hat\theta) < T`)
and positive for large :math:`r`.  The root :math:`r^*` — found via Brent's
method — yields the contour point

.. math::

    \theta^* = \hat\theta + L^{-T}(r^*\hat{e}).

Stage 3 — Gap detection
------------------------

After the radial search, :math:`M \gg N` candidate unit vectors are sampled
uniformly on :math:`S^{k-1}`.  For each candidate the maximum cosine
similarity to any radially-found direction is computed.  Candidates with the
smallest maximum similarity correspond to the largest angular gaps.

For each gap direction a **dedicated radial search** is run to locate the
exact contour crossing along that direction.  This ensures every RATTLE chain
starts directly on the contour in the sparse region, rather than at the
nearest dense radial point which may be far away geodesically.

Stage 4 — Constrained Hamiltonian Monte Carlo (RATTLE)
-------------------------------------------------------

Let :math:`C(\theta) = \mathrm{NLL}(\theta) - T` be the constraint function.
Starting from a radial contour point, the RATTLE integrator
:cite:t:`ANDERSEN198324` walks along :math:`\partial\mathcal{C}_\alpha`
while preserving the constraint at each step.  One leapfrog step with
step size :math:`\varepsilon` reads

.. math::

    p_{1/2} &= p_0
              - \tfrac{\varepsilon}{2}\,\nabla\mathrm{NLL}(\theta_0), \\[3pt]
    \theta'  &= \theta_0 + \varepsilon\, p_{1/2}, \\[3pt]
    \theta_1 &= \theta' - \lambda\,\nabla\mathrm{NLL}(\theta'),
    \quad \lambda = \frac{\mathrm{NLL}(\theta')-T}
                        {|\nabla\mathrm{NLL}(\theta')|^2}, \\[3pt]
    p'       &= p_{1/2} - \tfrac{\varepsilon}{2}\,\nabla\mathrm{NLL}(\theta_1), \\[3pt]
    p_1      &= p' - \frac{p' \cdot \nabla\mathrm{NLL}(\theta_1)}
                          {|\nabla\mathrm{NLL}(\theta_1)|^2}
               \,\nabla\mathrm{NLL}(\theta_1).

The third equation is the SHAKE projection onto the constraint surface,
iterated via Newton's method until
:math:`|\mathrm{NLL}(\theta_1) - T| < \varepsilon_\text{tol}`.
The fifth equation projects the momentum onto the tangent space of the
constraint, ensuring :math:`p_1 \perp \nabla C(\theta_1)`.

Stage 5 — Adaptive coverage improvement
-----------------------------------------

After Stage 4, all collected contour points (radial + RATTLE) are gathered
into a single point cloud.  A k-d tree is used to compute the
:math:`k_\text{NN}`-th nearest-neighbour distance for every point.  The
:math:`n_\text{chains}` most isolated points (those with the largest
:math:`k_\text{NN}` distance) seed a new round of RATTLE chains that explore
the sparsely covered neighbourhood.  This pass is repeated
:math:`n_\text{passes} - 1` times, providing bias-free coverage of regions
that are geometrically sparse on the contour even when they are not easily
detected as angular gaps from the MLE.

.. currentmodule:: spey.multiparameter

Functions
---------

.. autosummary::
    :toctree: ../_generated/

    contour.find_contour
    contour.ContourResult

"""

import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import autograd
import autograd.numpy as anp
import numpy as np
from joblib import Parallel, cpu_count, delayed
from scipy.optimize import minimize as _sp_minimize
from scipy.optimize import root_scalar
from scipy.stats import chi2 as chi2_dist
from tqdm import tqdm

log = logging.getLogger("Spey")

__all__ = ["ContourResult", "find_contour"]


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------


@dataclass
class ContourResult:
    r"""
    Container for the output of :func:`find_contour`.

    Attributes:
        theta_mle (``np.ndarray``): Maximum-likelihood estimate
            :math:`\hat\theta`, shape ``(k,)``.
        nll_min (``float``): Minimum negative log-likelihood
            :math:`\mathrm{NLL}(\hat\theta)`.
        threshold (``float``): NLL value that defines the contour boundary,
            :math:`T = \mathrm{NLL}(\hat\theta) + \Delta_\alpha / 2`.
        delta (``float``): Chi-squared quantile
            :math:`\Delta_\alpha = F^{-1}_{\chi^2_k}(1-\alpha)`.
        contour_points (``np.ndarray``): Points on the contour boundary,
            shape ``(n_points, k)``.  Every row :math:`\theta^*` satisfies
            :math:`|\mathrm{NLL}(\theta^*) - T| \lesssim \varepsilon_\text{tol}`.
        from_radial (``np.ndarray``): Boolean mask of shape ``(n_points,)``.
            ``True`` for points produced by the radial search; ``False`` for
            points added by the constrained RATTLE walk.
        parameter_names (``Optional[List[str]]``): Names of the :math:`k`
            parameters in the same order as the columns of
            :attr:`contour_points`, or ``None`` when the model does not
            provide names.
        confidence_level (``float``): The confidence level :math:`1-\alpha`,
            e.g. ``0.95``.
        dof (``int``): Degrees of freedom :math:`k` (number of model
            parameters).
    """

    theta_mle: np.ndarray
    nll_min: float
    threshold: float
    delta: float
    contour_points: np.ndarray
    from_radial: np.ndarray
    parameter_names: Optional[List[str]]
    confidence_level: float
    dof: int


def find_contour(
    stat_model,
    confidence_level: float = 0.95,
    poi_indices: list[int] = None,
    n_radial: int = 300,
    n_hmc_chains: int = 10,
    n_hmc_steps: int = 500,
    hmc_step_size: float = 0.05,
    n_gap_candidates: int = 3000,
    max_radial_bracket: float = 30.0,
    newton_tol: float = 1e-9,
    newton_max_iter: int = 100,
    whitener_regularisation: float = 1e-8,
    random_seed: Optional[int] = None,
    n_jobs: int = cpu_count() - 1,
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    n_coverage_passes: int = 2,
    coverage_knn: int = 5,
) -> ContourResult:
    r"""
    Find the :math:`(1-\alpha)` chi-squared confidence contour of a
    :class:`~spey.StatisticalModel` in its full parameter space.

    The algorithm operates in four stages (see the module docstring for the
    full mathematical derivation):

    1. **Pre-whitening** — compute the Fisher information :math:`G` at the
       MLE via the Hessian of the log-likelihood and Cholesky-factor it so
       that :math:`\varphi = L(\theta-\hat\theta)` maps the contour to an
       approximate sphere.

    2. **Radial search** — shoot :math:`N` random rays from :math:`\hat\theta`
       in whitened space and locate each ray's crossing of the NLL threshold
       :math:`T` with Brent's method.

    3. **Gap detection** — sample :math:`M \gg N` candidate directions and
       select the :math:`n_\text{chains}` directions least covered by the
       radial set as seeds for RATTLE chains.

    4. **Constrained RATTLE HMC** — for each seed direction, start a
       Hamiltonian Monte Carlo chain that walks along the constraint manifold
       :math:`\mathrm{NLL}(\theta) = T`, projecting both position and momentum
       at every step.

    .. versionadded:: 0.2.7

    Args:
        stat_model: A :class:`~spey.StatisticalModel` whose backend is
            ``"default.multivariate_normal"``.  The backend must expose
            ``get_logpdf_func()`` and ``get_hessian_logpdf_func()``; these are
            used to build the NLL, its gradient (via ``autograd``), and the
            Fisher information for whitening.
        confidence_level (``float``, default ``0.95``): Confidence level
            :math:`1-\alpha`.  The chi-squared threshold is
            :math:`\Delta_\alpha = F^{-1}_{\chi^2_k}(1-\alpha)`.
        poi_indices (``list[int]``, default ``None``): Indices of the model
            parameters (in the full parameter space, excluding :math:`\mu`)
            to include in the contour search.  :math:`\mu` itself is always
            excluded and held fixed at **1**.  Parameters *not* listed here
            — and not equal to :math:`\mu` — are treated as **profile
            parameters**: they are numerically minimised at every NLL
            evaluation so that the algorithm maps the *marginal* confidence
            boundary in the ``poi_indices`` subspace.  When ``None`` all
            non-:math:`\mu` parameters form the contour space and no
            profiling is performed.
        n_radial (``int``, default ``300``): Number of random radial
            directions to explore in Stage 2.
        n_hmc_chains (``int``, default ``10``): Number of RATTLE chains to
            run in Stage 4, one per detected gap direction.
        n_hmc_steps (``int``, default ``500``): Number of leapfrog steps per
            RATTLE chain.  Each step produces one contour point (after the
            constraint projection).
        hmc_step_size (``float``, default ``0.05``): Leapfrog step size
            :math:`\varepsilon` in whitened-coordinate units.
        n_gap_candidates (``int``, default ``3000``): Number of candidate
            directions sampled for gap detection.  Larger values give a more
            thorough search at modest extra cost.
        max_radial_bracket (``float``, default ``30.0``): Upper bracket for
            the radial root-finder in whitened-space units.  If the contour
            lies further than this from the MLE in some direction, that
            direction is skipped.
        newton_tol (``float``, default ``1e-9``): Convergence tolerance
            :math:`\varepsilon_\text{tol}` for the Newton projection step in
            RATTLE (i.e. for :math:`|\mathrm{NLL}(\theta)-T|`).
        newton_max_iter (``int``, default ``100``): Maximum Newton iterations
            in the constraint projection step.
        whitener_regularisation (``float``, default ``1e-8``): Minimum
            eigenvalue enforced on the Fisher information matrix before
            Cholesky factorisation, preventing singular whitening transforms
            when the likelihood is locally flat.
        random_seed (``Optional[int]``, default ``None``): Seed for the
            :class:`numpy.random.Generator` used throughout, enabling
            reproducible results.
        n_jobs (``int``, default ``cpu_count() - 1``): Number of parallel
            worker processes for the RATTLE HMC chains (Stage 4).  Each
            chain is independent and runs in a separate process via
            ``joblib.Parallel``.  Pass ``1`` to disable parallelism.
            Values :math:`\le 0` are clamped to ``1``.  Note: parallel execution
            requires the NLL and gradient closures to be picklable; if
            pickling fails, fall back to ``n_jobs=1``.
        bounds (default ``None``): Parameter bounds for each signal parameter
            (i.e. every parameter accepted by ``signal_yields``, in the same
            order).  Each element is a ``(lower, upper)`` pair where either
            value may be ``None`` to indicate no bound on that side.  For
            example, ``[(0.0, None), (-1.0, 1.0)]`` constrains the first
            parameter to be non-negative and the second to ``[-1, 1]``.
            When ``None`` the whole list is passed, no boundaries are applied.
            The radial search caps its bracket so that roots beyond a bound
            are never sought; the RATTLE walk terminates a chain the moment
            it steps outside the bounded region.
        n_coverage_passes (``int``, default ``2``): Total number of HMC
            exploration passes.  Pass 1 seeds chains from the angular gap
            directions (Stage 3/4).  Each subsequent pass (up to
            ``n_coverage_passes - 1`` additional rounds) uses
            :math:`k`-nearest-neighbour distances among all collected
            contour points to identify the most isolated regions and seeds
            new RATTLE chains there (Stage 5).  Set to ``1`` to disable
            the adaptive coverage stage and reproduce the original
            single-pass behaviour.
        coverage_knn (``int``, default ``5``): The :math:`k` in the
            :math:`k`-NN distance used by Stage 5 to measure local point
            density.  Larger values smooth the density estimate over a wider
            neighbourhood; smaller values focus on very localised gaps.
            Must satisfy ``coverage_knn >= 1``.


    Returns:
        :class:`~spey.multiparameter.contour.ContourResult`
            A dataclass containing
            :attr:`~spey.multiparameter.contour.ContourResult.theta_mle`,
            :attr:`~spey.multiparameter.contour.ContourResult.nll_min`,
            :attr:`~spey.multiparameter.contour.ContourResult.threshold`,
            :attr:`~spey.multiparameter.contour.ContourResult.contour_points`,
            and associated metadata.


    Raises:
        ValueError: If the model has fewer than one parameter.
        RuntimeError: If the MLE optimisation fails to converge.


    Examples:
        .. code:: python3

            import numpy as np
            import spey
            from spey.multiparameter.contour import find_contour

            base_signal = np.array([12.0, 15.0])
            bkg   = np.array([50.0, 48.0])
            data  = np.array([36.0, 33.0])
            cov   = np.array([[144.0, 13.0], [25.0, 256.0]])

            def signal(pars):
                return base_signal * (1.0 + pars[0])

            pdf_wrapper = spey.get_backend('default.multivariate_normal')
            stat_model = pdf_wrapper(
                signal_yields=signal,
                background_yields=bkg,
                data=data,
                covariance_matrix=cov,
                n_signal_parameters=1,
                analysis="demo",
            )

            result = find_contour(
                stat_model,
                confidence_level=0.95,
                n_radial=200,
                n_hmc_chains=5,
                random_seed=42
            )

            print("MLE :", result.theta_mle)
            print("NLL min :", result.nll_min)
            print("Contour points:", result.contour_points.shape)

    """
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError(f"confidence_level must be in (0, 1); got {confidence_level}.")

    n_jobs = max(1, n_jobs)
    rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Introspect the model
    # ------------------------------------------------------------------
    cfg = stat_model.config()
    k_total: int = cfg.npar  # total number of model parameters
    poi_idx: int = cfg.poi_index  # index of the signal-strength parameter mu
    cfg_bounds: list[tuple[float, float]] = cfg.suggested_bounds

    # All non-mu parameter indices, sorted ascending.
    all_signal_indices: List[int] = [i for i in range(k_total) if i != poi_idx]

    # Contour parameter indices: the subspace whose boundary we map.
    # Parameters listed in poi_indices (minus mu) are the contour axes.
    if poi_indices is None:
        contour_indices: List[int] = all_signal_indices
    else:
        contour_indices: List[int] = sorted(i for i in poi_indices if i != poi_idx)

    # Profile (nuisance) parameter indices: non-mu parameters NOT in the
    # contour subspace.  These are minimised at every NLL evaluation so
    # that the algorithm maps the *marginal* confidence boundary.
    _contour_set = set(contour_indices)
    profile_indices: List[int] = [i for i in all_signal_indices if i not in _contour_set]

    k: int = len(contour_indices)  # dimension of the contour search space
    k_profile: int = len(profile_indices)

    if k < 1:
        raise ValueError(
            "Model must have at least one non-POI (signal shape) parameter "
            "for contour finding."
        )

    if bounds is not None and len(bounds) != k:
        raise ValueError(
            f"bounds must have one entry per signal parameter (expected {k}, "
            f"got {len(bounds)})."
        )
    if bounds is None:
        bounds = [cfg_bounds[idx] for idx in contour_indices]

    # Parameter names for the contour subspace only (mu excluded)
    _all_names: Optional[List[str]] = cfg.parameter_names
    param_names: Optional[List[str]] = (
        [_all_names[i] for i in contour_indices] if _all_names is not None else None
    )

    # ------------------------------------------------------------------
    # Stage 1: find MLE over ALL signal parameters with mu fixed at 1.
    # This gives the correct nll_min (threshold base) and the initial
    # values for profile parameters.
    # ------------------------------------------------------------------
    logpdf_fn = stat_model.backend.get_logpdf_func()

    theta_mle_all, nll_min = _find_mle(stat_model, all_signal_indices)
    log.debug("MLE all signal params = %s, NLL_min = %.6f", theta_mle_all, nll_min)

    # Map global indices → positions within all_signal_indices
    _sig_pos = {idx: pos for pos, idx in enumerate(all_signal_indices)}
    _contour_pos = [_sig_pos[i] for i in contour_indices]
    _profile_pos = [_sig_pos[i] for i in profile_indices]

    theta_mle = theta_mle_all[_contour_pos]  # (k,)
    theta_mle_profile = theta_mle_all[_profile_pos] if profile_indices else np.empty(0)

    # ------------------------------------------------------------------
    # Build NLL callables in the contour-parameter subspace (mu = 1).
    #
    # Two regimes:
    #   k_profile == 0  — all signal params are contour params; use
    #                     autograd-differentiable NLL directly.
    #   k_profile  > 0  — profile parameters are minimised at every NLL
    #                     evaluation; gradient via the envelope theorem.
    # ------------------------------------------------------------------
    if k_profile == 0:
        # All signal parameters are contour parameters (current behaviour).
        # anp.concatenate keeps the computation graph autograd-differentiable.
        def _nll(signal_pars: np.ndarray):
            r"""
            Autograd-compatible NLL as a function of all signal parameters.

            :math:`\mu` is held fixed at **1** by inserting it at position
            *poi_idx* of the full parameter vector.
            """
            full_pars = anp.concatenate(
                [
                    signal_pars[:poi_idx],
                    anp.array([1.0]),
                    signal_pars[poi_idx:],
                ]
            )
            return -logpdf_fn(full_pars)

        _grad_nll_auto = autograd.grad(_nll)  # pylint: disable=no-value-for-parameter

        def nll_scalar(contour_pars: np.ndarray) -> float:
            """Plain-float NLL (mu=1 fixed) for scipy root-finders."""
            return float(_nll(np.asarray(contour_pars, dtype=float)))

        def grad_nll(contour_pars: np.ndarray) -> np.ndarray:
            r"""Gradient :math:`\nabla_{\theta_c}\mathrm{NLL}`, mu=1 fixed."""
            return np.array(
                _grad_nll_auto(np.asarray(contour_pars, dtype=float)), dtype=float
            )

    else:
        # Profile regime: minimise over profile_indices at each evaluation.
        # Gradient uses the envelope theorem:
        #   ∇_{θ_c} NLL_profiled = [∇_{full} NLL]_{contour columns}
        # evaluated at the profile-optimal point.
        #
        # The inner minimisation uses the profile-parameter subspace directly
        # with bounds and constraints from prepare_for_fit, matching the
        # SLSQP defaults used by the rest of spey.
        _contour_arr = np.array(contour_indices, dtype=int)
        _profile_arr = np.array(profile_indices, dtype=int)
        _theta_profile_init = theta_mle_profile.copy()

        def _assemble(cp: np.ndarray, pp: np.ndarray) -> np.ndarray:
            """Build full parameter vector: mu=1, contour=cp, profile=pp."""
            full = np.empty(k_total, dtype=float)
            full[poi_idx] = 1.0
            full[_contour_arr] = cp
            full[_profile_arr] = pp
            return full

        # Prepare the fit once to access the model configuration (bounds,
        # suggested_init) and any backend constraints.

        _spey_fit_args = stat_model.prepare_for_fit()
        _model_cfg = _spey_fit_args["model_configuration"]
        _backend_constraints = _spey_fit_args.get("constraints", [])

        # Bounds for the profile parameters only (taken from model config).
        _all_bounds = list(_model_cfg.suggested_bounds)
        _profile_bounds = [_all_bounds[i] for i in profile_indices]

        def _profile_opt(cp: np.ndarray) -> np.ndarray:
            r"""
            Minimise the NLL over profile parameters with :math:`\mu` and
            the contour parameters fixed.

            The optimisation is performed directly in the reduced
            :math:`k_\text{profile}`-dimensional profile subspace — not
            over the full parameter vector — so no equality constraints are
            needed and SLSQP never encounters an over-constrained problem.

            Bounds are taken from ``model_configuration.suggested_bounds``
            for the profile-parameter slots.  Any backend constraints (which
            are typically defined on the full parameter vector) are adapted
            by embedding the current contour values into the constraint
            function via a closure.

            The optimisation uses SLSQP with ``tol=1e-6`` and
            ``maxiter=10000``, matching the defaults in
            :func:`~spey.optimizer.scipy_tools.minimize`.  The initial
            point for the profile parameters is the joint signal-MLE value
            :math:`\hat\theta_p`.
            """

            def obj(pp: np.ndarray) -> float:
                return float(-logpdf_fn(_assemble(cp, pp)))

            # Adapt backend constraints (full-param) to profile-param space.
            constraints = []
            for c in _backend_constraints:
                orig_fun = c["fun"]

                def _wrapped(pp, _f=orig_fun, _cp=cp):
                    return _f(_assemble(_cp, pp))

                constraints.append({**c, "fun": _wrapped})

            res = _sp_minimize(
                obj,
                x0=_theta_profile_init,
                method="SLSQP",
                bounds=_profile_bounds,
                constraints=constraints,
                tol=1e-6,
                options={"maxiter": 10000, "disp": False},
            )
            return res.x if res.success else _theta_profile_init

        _grad_logpdf_auto = autograd.grad(  # pylint: disable = no-value-for-parameter
            logpdf_fn
        )

        def nll_scalar(contour_pars: np.ndarray) -> float:
            """Profiled NLL: min over profile params with contour fixed."""
            cp = np.asarray(contour_pars, dtype=float)
            pp = _profile_opt(cp)
            return float(-logpdf_fn(_assemble(cp, pp)))

        def grad_nll(contour_pars: np.ndarray) -> np.ndarray:
            r"""
            Gradient of profiled NLL via the envelope theorem.

            At the profile optimum :math:`\theta_p^*(\theta_c)`, the
            gradient of :math:`\mathrm{NLL}` w.r.t. :math:`\theta_p` is
            zero, so:

            .. math::

                \nabla_{\theta_c}\mathrm{NLL}_\text{prof}(\theta_c)
                = \bigl[\nabla_\text{full}\,\mathrm{NLL}\bigr]_{\text{contour cols}}
                  \Big|_{\theta_p = \theta_p^*(\theta_c)}.
            """
            cp = np.asarray(contour_pars, dtype=float)
            pp = _profile_opt(cp)
            full = _assemble(cp, pp)
            full_grad_logpdf = np.array(_grad_logpdf_auto(full), dtype=float)
            return -full_grad_logpdf[_contour_arr]

    # ------------------------------------------------------------------
    # Compute chi-squared threshold
    # DoF = k = number of contour parameters (profiled params are not DoF)
    # ------------------------------------------------------------------
    delta = float(chi2_dist.ppf(confidence_level, df=k))
    threshold = nll_min + delta / 2.0
    log.debug("delta(chi2, df=%d) = %.4f, NLL threshold = %.6f", k, delta, threshold)

    # ------------------------------------------------------------------
    # Stage 1 (continued): build whitening transform from Fisher information
    # Hessian is evaluated at the full parameter vector with mu=1.
    # When profile parameters exist the Schur complement gives the marginal
    # precision for the contour subspace; otherwise the contour block is used.
    # ------------------------------------------------------------------
    theta_mle_full = np.empty(k_total, dtype=float)
    theta_mle_full[poi_idx] = 1.0
    theta_mle_full[np.array(contour_indices)] = theta_mle
    if profile_indices:
        theta_mle_full[np.array(profile_indices)] = theta_mle_profile

    hessian_fn = stat_model.backend.get_hessian_logpdf_func()
    L, L_inv_T = _build_whitener(
        hessian_fn,
        theta_mle_full,
        contour_indices,
        profile_indices,
        k,
        whitener_regularisation,
    )

    def to_theta(phi: np.ndarray) -> np.ndarray:
        r"""
        Inverse whitening in signal-parameter space:
        :math:`\theta_s = \hat\theta_s + L^{-T}\varphi`.
        """
        return theta_mle + L_inv_T @ phi

    def _max_r_for_dir(e: np.ndarray) -> float:
        r"""
        Maximum :math:`r \ge 0` such that
        :math:`\theta_s(r) = \hat\theta_s + L^{-T}(r\hat{e})`
        stays within all finite parameter bounds.

        For each constrained index :math:`i`, the step in original
        parameter space is :math:`s_i = (L^{-T}\hat{e})_i`.  A positive
        step hits the upper bound at :math:`r = (u_i - \hat\theta_i)/s_i`;
        a negative step hits the lower bound at
        :math:`r = (l_i - \hat\theta_i)/s_i`.  The minimum over all such
        constraints gives the first boundary encountered along the ray.
        Returns ``np.inf`` when no bounds are active.
        """
        if bounds is None:
            return np.inf
        step = L_inv_T @ e
        r_max = np.inf
        for i, (lo, hi) in enumerate(bounds):
            if hi is not None and step[i] > 1e-30:
                r_max = min(r_max, (hi - theta_mle[i]) / step[i])
            if lo is not None and step[i] < -1e-30:
                r_max = min(r_max, (lo - theta_mle[i]) / step[i])
        return float(r_max)

    # ------------------------------------------------------------------
    # Stage 2: radial search in whitened space
    # ------------------------------------------------------------------
    radial_points, radial_dirs = _radial_search(
        nll_scalar,
        theta_mle,
        threshold,
        to_theta,
        k,
        n_radial,
        max_radial_bracket,
        rng,
        _max_r_for_dir,
    )
    log.debug(
        "Radial search: %d / %d directions converged.", len(radial_points), n_radial
    )

    # ------------------------------------------------------------------
    # Stage 3: gap detection — angular gaps in whitened direction space
    # ------------------------------------------------------------------
    gap_dirs = _detect_gap_directions(radial_dirs, k, n_gap_candidates, n_hmc_chains, rng)

    # ------------------------------------------------------------------
    # Stage 4: constrained RATTLE HMC from gap seeds
    #
    # For each gap direction we first attempt a targeted radial search to
    # find the exact contour crossing along that direction.  This places
    # the chain starting point directly in the sparse region rather than
    # at the nearest (potentially dense) radial point.
    # ------------------------------------------------------------------
    hmc_points: np.ndarray
    if k == 1:
        # Tangent space is 0-dimensional — RATTLE cannot move; radial search
        # already found both contour points (±direction).
        hmc_points = np.empty((0, k), dtype=float)
        log.debug("k=1: skipping RATTLE (tangent space is trivial).")
    else:
        gap_starts = _find_gap_starts(
            gap_dirs,
            nll_scalar,
            theta_mle,
            threshold,
            to_theta,
            max_radial_bracket,
            _max_r_for_dir,
            radial_points,
            radial_dirs,
        )

        if len(gap_starts) == 0:
            hmc_pts = np.empty((0, k), dtype=float)
        else:
            hmc_pts = _run_hmc_chains(
                nll_scalar,
                grad_nll,
                threshold,
                np.array(gap_starts, dtype=float),
                n_hmc_steps,
                hmc_step_size,
                newton_tol,
                newton_max_iter,
                rng,
                n_jobs,
                bounds,
            )
        log.debug(
            "Stage 4: %d gap seeds → %d RATTLE points.",
            len(gap_starts),
            len(hmc_pts),
        )

        # ------------------------------------------------------------------
        # Stage 5: adaptive coverage — iterative kNN-based reseeding
        #
        # Pool all contour points found so far; use a k-d tree to identify
        # the most isolated (largest kNN distance) regions and seed new
        # chains there.  Repeat for n_coverage_passes − 1 additional passes.
        # ------------------------------------------------------------------
        knn_k = max(1, coverage_knn)
        _all_pts: List[np.ndarray] = []
        if len(radial_points) > 0:
            _all_pts.append(radial_points)
        if len(hmc_pts) > 0:
            _all_pts.append(hmc_pts)
        collected = np.vstack(_all_pts) if _all_pts else np.empty((0, k), dtype=float)

        for pass_idx in range(n_coverage_passes - 1):
            sparse_starts = _find_sparse_seeds(collected, n_hmc_chains, knn_k, rng)
            if sparse_starts is None or len(sparse_starts) == 0:
                break
            log.debug(
                "Stage 5 pass %d/%d: %d sparse seeds.",
                pass_idx + 2,
                n_coverage_passes,
                len(sparse_starts),
            )
            new_pts = _run_hmc_chains(
                nll_scalar,
                grad_nll,
                threshold,
                sparse_starts,
                n_hmc_steps,
                hmc_step_size,
                newton_tol,
                newton_max_iter,
                rng,
                n_jobs,
                bounds,
            )
            if len(new_pts) > 0:
                hmc_pts = np.vstack([hmc_pts, new_pts]) if len(hmc_pts) > 0 else new_pts
                collected = np.vstack([collected, new_pts])

        hmc_points = hmc_pts

    log.debug("RATTLE+adaptive produced %d additional contour points.", len(hmc_points))

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    if len(radial_points) > 0 and len(hmc_points) > 0:
        all_points = np.vstack([radial_points, hmc_points])
        from_radial = np.concatenate(
            [
                np.ones(len(radial_points), dtype=bool),
                np.zeros(len(hmc_points), dtype=bool),
            ]
        )
    elif len(radial_points) > 0:
        all_points = radial_points
        from_radial = np.ones(len(radial_points), dtype=bool)
    elif len(hmc_points) > 0:
        all_points = hmc_points
        from_radial = np.zeros(len(hmc_points), dtype=bool)
    else:
        warnings.warn(
            "No contour points were found.  The confidence region may be "
            "unbounded or the NLL landscape is numerically ill-conditioned.",
            RuntimeWarning,
            stacklevel=2,
        )
        all_points = np.empty((0, k), dtype=float)
        from_radial = np.empty(0, dtype=bool)

    return ContourResult(
        theta_mle=theta_mle,
        nll_min=nll_min,
        threshold=threshold,
        delta=delta,
        contour_points=all_points,
        from_radial=from_radial,
        parameter_names=param_names,
        confidence_level=confidence_level,
        dof=k,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_mle(
    stat_model,
    signal_indices: List[int],
) -> Tuple[np.ndarray, float]:
    r"""
    Find the MLE of the signal parameters with :math:`\mu` fixed at **1**.

    Calls :meth:`~spey.StatisticalModel.maximize_likelihood` with
    ``fixed_poi_value=1.0`` (which fixes :math:`\mu` via the ``fit``
    optimizer) and ``poi_indices=signal_indices`` to retrieve the fitted
    values for the requested parameters.

    Should be called with ``all_signal_indices`` (all non-:math:`\mu`
    parameters) so that the returned ``nll_min`` is the global minimum
    over the full signal-parameter space and profile parameters start
    from their optimal values.

    Args:
        stat_model: A :class:`~spey.StatisticalModel` instance.
        signal_indices (``List[int]``): Indices of the parameters to
            optimise (typically all non-:math:`\mu` parameters).

    Returns:
        ``Tuple[np.ndarray, float]``:
        ``(theta_mle, nll_min)`` — the MLE signal-parameter vector of
        shape ``(k,)`` (where ``k = len(signal_indices)``) and the
        minimum NLL value :math:`\mathrm{NLL}(\mu{=}1,\,\hat\theta_s)`.

    Raises:
        RuntimeError: If ``maximize_likelihood`` returns a non-finite NLL.
    """
    signal_dict, nll_min = stat_model.maximize_likelihood(
        poi_indices=signal_indices,
        return_nll=True,
        fixed_poi_value=1.0,  # fixes pars[poi_idx]=1; flows to fit()
    )
    theta_mle = np.array([signal_dict[i] for i in signal_indices], dtype=float)
    nll_min = float(nll_min)
    if not np.isfinite(nll_min):
        raise RuntimeError(
            f"maximize_likelihood returned a non-finite NLL ({nll_min}).  "
            "The model may be ill-conditioned."
        )
    return theta_mle, nll_min


def _build_whitener(
    hessian_fn,
    theta_mle_full: np.ndarray,
    contour_indices: List[int],
    profile_indices: List[int],
    k: int,
    reg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Construct the whitening matrices :math:`L` and :math:`L^{-T}` from the
    **marginal** Fisher information for the contour parameters at the MLE.

    The full observed Fisher information is evaluated at
    :math:`(\mu{=}1,\,\hat\theta_s)`:

    .. math::

        G_\text{full} = -\nabla^2 \log\mathcal{L}(\mu{=}1,\,\hat\theta_s).

    When there are no profile parameters the contour block is used directly:

    .. math::

        G = \bigl[G_\text{full}\bigr]_{ij},
        \quad i,j \in \text{contour\_indices}.

    When profile parameters are present the **Schur complement** gives the
    marginal precision matrix that matches the curvature of the profiled NLL:

    .. math::

        G = G_{cc} - G_{cp}\,G_{pp}^{-1}\,G_{pc},

    where :math:`G_{cc}`, :math:`G_{cp}`, :math:`G_{pp}` are the
    contour–contour, contour–profile, and profile–profile blocks of
    :math:`G_\text{full}`, respectively.

    After spectral regularisation and Cholesky factorisation
    :math:`G = LL^T`, the whitened coordinate
    :math:`\varphi = L(\theta_c - \hat\theta_c)` maps the contour to
    approximately a :math:`(k-1)`-sphere.

    If any computation fails, the identity is used as a fallback.

    Args:
        hessian_fn: Callable returned by
            :meth:`~spey.base.BackendBase.get_hessian_logpdf_func`.
        theta_mle_full (``np.ndarray``): Full MLE parameter vector with
            :math:`\mu=1` inserted at the correct position.
        contour_indices (``List[int]``): Indices of the contour parameters.
        profile_indices (``List[int]``): Indices of the profile parameters
            (empty list when all signal params are contour params).
        k (``int``): Number of contour parameters (``len(contour_indices)``).
        reg (``float``): Minimum eigenvalue for spectral regularisation.

    Returns:
        ``Tuple[np.ndarray, np.ndarray]``:
        ``(L, L_inv_T)`` — lower-triangular Cholesky factor and its
        transpose-inverse, each of shape ``(k, k)``.
    """
    try:
        H_logpdf_full = np.array(hessian_fn(theta_mle_full.astype(float)), dtype=float)
        ci = np.array(contour_indices)
        G_cc = -H_logpdf_full[np.ix_(ci, ci)]

        if profile_indices:
            # Schur complement: marginal precision for contour params,
            # accounting for the profiled (free) parameters.
            pi = np.array(profile_indices)
            G_pp = -H_logpdf_full[np.ix_(pi, pi)]
            G_cp = -H_logpdf_full[np.ix_(ci, pi)]
            G_pc = -H_logpdf_full[np.ix_(pi, ci)]
            try:
                G = G_cc - G_cp @ np.linalg.solve(G_pp, G_pc)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "Schur complement solve failed; falling back to "
                    "block-diagonal approximation for whitening.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                G = G_cc
        else:
            G = G_cc
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Hessian computation failed ({exc}); falling back to identity "
            "whitening (equivalent to no pre-conditioning).",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.eye(k), np.eye(k)

    # Spectral regularisation: clip small / negative eigenvalues
    w, V = np.linalg.eigh(G)
    w = np.maximum(w, reg)
    G_reg = (V * w) @ V.T

    try:
        L = np.linalg.cholesky(G_reg)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Cholesky factorisation failed after regularisation; "
            "falling back to identity whitening.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.eye(k), np.eye(k)

    L_inv = np.linalg.solve(L, np.eye(k))  # L^{-1}
    L_inv_T = L_inv.T  # L^{-T}
    return L, L_inv_T


def _radial_search(
    nll_scalar,
    theta_mle: np.ndarray,
    threshold: float,
    to_theta,
    k: int,
    n_samples: int,
    max_bracket: float,
    rng: np.random.Generator,
    max_r_for_dir=None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Locate contour points by root-finding along random rays in whitened space.

    For each sample a unit vector :math:`\hat{e}` is drawn uniformly on
    :math:`S^{k-1}` by normalising a standard Gaussian vector.  The
    one-dimensional function

    .. math::

        f(r) = \mathrm{NLL}\!\bigl(\theta(r)\bigr) - T,
        \qquad \theta(r) = \hat\theta + L^{-T}(r\hat{e}),

    satisfies :math:`f(0) < 0` (MLE is inside the contour) and
    :math:`f(r) \to +\infty` as :math:`r \to \infty`.  The unique root
    :math:`r^*` is found with Brent's method
    (:func:`scipy.optimize.root_scalar`).

    The upper bracket is initialised at :math:`r_\text{init} = \sqrt{\Delta}`
    (the approximate whitened-space radius) and doubled until :math:`f > 0`
    or the maximum :math:`r_\text{max}` is reached.

    Args:
        nll_scalar: NLL function returning a plain ``float``.
        theta_mle (``np.ndarray``): MLE, shape ``(k,)``.
        threshold (``float``): NLL threshold :math:`T`.
        to_theta: Callable mapping whitened coordinate :math:`\varphi` to
            :math:`\theta`, i.e. :math:`\theta = \hat\theta + L^{-T}\varphi`.
        k (``int``): Parameter-space dimension.
        n_samples (``int``): Number of ray directions to try.
        max_bracket (``float``): Upper bound on the radial search in
            whitened-space units.
        rng (``np.random.Generator``): Random number generator.
        max_r_for_dir: Optional callable ``e -> float`` that returns the
            maximum :math:`r` allowed in direction *e* before a parameter
            bound is violated.  ``None`` means no bounds are active.

    Returns:
        ``Tuple[np.ndarray, np.ndarray]``:
        ``(contour_points, directions)`` — arrays of shape
        ``(n_found, k)`` and ``(n_found, k)`` respectively, where
        *n_found* ≤ *n_samples*.
    """
    nll_at_mle = nll_scalar(theta_mle)
    # Guard: if numerical NLL at MLE already exceeds threshold, something is wrong
    if nll_at_mle >= threshold:
        warnings.warn(
            f"NLL at MLE ({nll_at_mle:.6f}) ≥ threshold ({threshold:.6f}).  "
            "The MLE may not be well-converged; radial search will be skipped.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.empty((0, k), dtype=float), np.empty((0, k), dtype=float)

    # Approximate whitened-space radius for initial bracket
    r_init = float(np.sqrt(max(threshold - nll_at_mle, 1.0) * 2.0))

    contour_points: List[np.ndarray] = []
    directions: List[np.ndarray] = []

    for _ in range(n_samples):
        # Sample uniform direction on S^{k-1}
        z = rng.standard_normal(size=k)
        norm = np.linalg.norm(z)
        if norm < 1e-15:
            continue
        e = z / norm

        def f(r: float) -> float:  # noqa: ANN001
            return nll_scalar(to_theta(r * e)) - threshold

        # Effective upper bracket: limited by max_bracket AND parameter bounds
        r_bounds_max: float = max_r_for_dir(e) if max_r_for_dir is not None else np.inf
        effective_max: float = min(max_bracket, r_bounds_max)
        if effective_max < 1e-10:
            # MLE is at or past a boundary in this direction; skip
            continue

        r_lo: float = 1e-8
        r_hi: float = min(r_init, effective_max)
        try:
            f_lo = f(r_lo)
            if f_lo >= 0.0:
                # MLE not inside contour along this direction — very unusual
                continue
        except Exception:  # noqa: BLE001
            continue

        try:
            while f(r_hi) <= 0.0 and r_hi < effective_max:
                r_hi = min(r_hi * 2.0, effective_max)
            if f(r_hi) <= 0.0:
                # Contour not found within max_bracket / bounds in this direction
                continue
        except Exception:  # noqa: BLE001
            continue

        try:
            result = root_scalar(
                f,
                bracket=[r_lo, r_hi],
                method="brentq",
                xtol=1e-12,
                rtol=1e-12,
            )
        except ValueError:
            continue

        if result.converged:
            theta_star = to_theta(result.root * e)
            contour_points.append(theta_star)
            directions.append(e)

    if not contour_points:
        return np.empty((0, k), dtype=float), np.empty((0, k), dtype=float)
    return np.array(contour_points, dtype=float), np.array(directions, dtype=float)


def _detect_gap_directions(
    found_dirs: np.ndarray,
    k: int,
    n_candidates: int,
    n_gaps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""
    Identify directions in whitened space with sparse radial coverage.

    A set of :math:`M` candidate unit vectors is drawn uniformly on
    :math:`S^{k-1}`.  For each candidate :math:`c` the *coverage score*

    .. math::

        s(c) = \max_{d \in \mathcal{D}} \langle c, d \rangle

    measures its cosine similarity to the nearest already-found direction
    :math:`d` in the set :math:`\mathcal{D}`.  A small :math:`s(c)` (close
    to :math:`-1`) indicates that :math:`c` lies in an angular gap.  The
    :math:`n_\text{gaps}` candidates with the smallest :math:`s` are returned
    as RATTLE seeds.

    If no radial directions have been found, :math:`n_\text{gaps}` random
    directions are returned.

    Args:
        found_dirs (``np.ndarray``): Unit directions already explored by the
            radial search, shape ``(n_found, k)``.
        k (``int``): Parameter-space dimension.
        n_candidates (``int``): Number of candidate directions to sample
            (:math:`M` in the description above).
        n_gaps (``int``): Number of gap directions to return.
        rng (``np.random.Generator``): Random number generator.

    Returns:
        ``np.ndarray``: Gap-direction unit vectors, shape
        ``(n_gaps, k)``.
    """
    Z = rng.standard_normal(size=(n_candidates, k))
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    candidates = Z / np.maximum(norms, 1e-15)

    if len(found_dirs) == 0:
        return candidates[:n_gaps]

    # cos_sim[i, j] = dot(candidates[i], found_dirs[j])
    cos_sim = candidates @ found_dirs.T  # (n_candidates, n_found)
    max_cos = np.max(cos_sim, axis=1)  # coverage score per candidate

    # Least-covered = smallest max cosine
    gap_idx = np.argsort(max_cos)[:n_gaps]
    return candidates[gap_idx]


def _find_gap_starts(
    gap_dirs: np.ndarray,
    nll_scalar,
    theta_mle: np.ndarray,
    threshold: float,
    to_theta,
    max_bracket: float,
    max_r_for_dir,
    radial_points: np.ndarray,
    radial_dirs: np.ndarray,
) -> List[np.ndarray]:
    r"""
    Find a contour starting point for each gap direction.

    For each unit vector :math:`\hat{e}` in *gap_dirs* the function attempts
    a targeted radial search — a Brent root-finding call along the ray
    :math:`\theta(r) = \hat\theta + L^{-T}(r\hat{e})` — so that the
    returned point lies directly on the contour in the gap region.

    If the radial search fails (the contour is not crossed within the
    bracket, or the NLL evaluation raises an exception), the function falls
    back to the nearest already-found radial point measured by cosine
    similarity of directions.  If no radial points exist at all, that gap
    direction is silently skipped.

    This two-step strategy ensures chains start *inside* each detected gap
    rather than at a nearby dense region, which is critical for non-spherical
    or multi-modal confidence contours where angular proximity in whitened
    space does not imply geometric proximity on the manifold.

    Args:
        gap_dirs (``np.ndarray``): Unit gap directions in whitened space,
            shape ``(n_gaps, k)``.
        nll_scalar: NLL function returning a plain ``float``.
        theta_mle (``np.ndarray``): MLE in contour-parameter space, shape
            ``(k,)``.
        threshold (``float``): NLL threshold :math:`T`.
        to_theta: Callable mapping whitened coordinate :math:`\varphi`
            to signal-parameter space :math:`\theta`.
        max_bracket (``float``): Maximum radial bracket in whitened space.
        max_r_for_dir: Optional callable ``e -> float`` returning the
            maximum allowed :math:`r` in direction *e* before a parameter
            bound is hit.  ``None`` means no bounds.
        radial_points (``np.ndarray``): Already-found radial contour points,
            shape ``(n_radial, k)``.  Used as fallback starts.
        radial_dirs (``np.ndarray``): Corresponding whitened-space unit
            directions, shape ``(n_radial, k)``.

    Returns:
        ``List[np.ndarray]``: Starting points (one per successfully handled
        gap direction), each of shape ``(k,)``.
    """
    nll_at_mle = nll_scalar(theta_mle)
    r_init = float(np.sqrt(max(threshold - nll_at_mle, 1.0) * 2.0))
    has_radial = len(radial_points) > 0

    starts: List[np.ndarray] = []
    for e in gap_dirs:
        r_max = max_r_for_dir(e) if max_r_for_dir is not None else np.inf
        eff_max = min(max_bracket, r_max)

        found = False
        if eff_max > 1e-10:
            r_lo: float = 1e-8
            r_hi: float = min(r_init, eff_max)

            # Use a default-argument capture of e so the closure is correct
            # even if Python reuses the loop variable name.
            def f(r: float, _e: np.ndarray = e) -> float:  # noqa: ANN001
                return nll_scalar(to_theta(r * _e)) - threshold

            try:
                if f(r_lo) < 0.0:
                    while f(r_hi) <= 0.0 and r_hi < eff_max:
                        r_hi = min(r_hi * 2.0, eff_max)
                    if f(r_hi) > 0.0:
                        res = root_scalar(
                            f,
                            bracket=[r_lo, r_hi],
                            method="brentq",
                            xtol=1e-12,
                            rtol=1e-12,
                        )
                        if res.converged:
                            starts.append(to_theta(res.root * e))
                            found = True
            except Exception:  # noqa: BLE001
                pass

        if not found:
            if has_radial:
                nearest = int(np.argmax(radial_dirs @ e))
                starts.append(radial_points[nearest].copy())
            # else: no fallback available — silently skip this gap direction

    return starts


def _find_sparse_seeds(
    all_points: np.ndarray,
    n_seeds: int,
    knn_k: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    r"""
    Identify the most sparsely covered contour points via :math:`k`-NN distance.

    A k-d tree is built over *all_points* and the :math:`k`-th nearest-neighbour
    distance :math:`d_k(\theta^*)` is computed for every point
    :math:`\theta^* \in` *all_points*.  Points with the largest
    :math:`d_k` values lie in the sparsest regions of the point cloud and
    are ideal seeds for new RATTLE chains.

    This metric is unbiased with respect to the geometry of the contour:
    it detects sparse regions purely from the distribution of existing points,
    regardless of their direction from the MLE.  It therefore complements
    the angular gap detection of Stage 3, which can miss geometric gaps on
    highly curved or non-Gaussian contours.

    Args:
        all_points (``np.ndarray``): All contour points collected so far,
            shape ``(n, k)``.
        n_seeds (``int``): Number of sparse starting points to return.
        knn_k (``int``): Neighbour index used for the density estimate;
            clamped to ``min(knn_k, n - 1)``.
        rng (``np.random.Generator``): Random number generator (reserved
            for future tie-breaking; currently unused).

    Returns:
        ``Optional[np.ndarray]``: The *n_seeds* most isolated points, shape
        ``(n_seeds, k)``, or ``None`` when the point cloud is too small
        (:math:`n < 2`).
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    n = len(all_points)
    if n < 2:
        return None

    k_eff = min(knn_k, n - 1)
    tree = cKDTree(all_points)
    # query k_eff + 1 neighbours: index 0 is the point itself (distance 0)
    dists, _ = tree.query(all_points, k=k_eff + 1)
    knn_dists = dists[:, k_eff]  # distance to the k_eff-th nearest neighbour

    n_out = min(n_seeds, n)
    sparse_idx = np.argsort(knn_dists)[-n_out:]  # largest distances first
    return all_points[sparse_idx]


def _project_theta(
    nll_scalar,
    grad_nll,
    theta: np.ndarray,
    threshold: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    r"""
    Project :math:`\theta` onto the constraint surface
    :math:`\mathrm{NLL}(\theta) = T` via Newton's method (SHAKE step).

    The Newton correction is

    .. math::

        \lambda = \frac{\mathrm{NLL}(\theta) - T}{|\nabla\mathrm{NLL}(\theta)|^2},
        \qquad
        \theta \leftarrow \theta - \lambda\,\nabla\mathrm{NLL}(\theta),

    repeated until :math:`|\mathrm{NLL}(\theta) - T| < \varepsilon_\text{tol}`
    or *max_iter* iterations are exhausted.

    This is a scalar version of the SHAKE constraint-correction used in
    molecular dynamics :footcite:t:`andersen1983rattle`.

    Args:
        nll_scalar: NLL function returning a plain ``float``.
        grad_nll: Gradient of NLL, returning a ``np.ndarray``.
        theta (``np.ndarray``): Starting point, shape ``(k,)``.
        threshold (``float``): Target NLL value :math:`T`.
        tol (``float``): Convergence tolerance on the constraint residual.
        max_iter (``int``): Maximum number of Newton iterations.

    Returns:
        ``np.ndarray``: Projected point on (or very close to) the constraint
        surface, shape ``(k,)``.
    """
    for _ in range(max_iter):
        residual = nll_scalar(theta) - threshold
        if abs(residual) < tol:
            break
        g = grad_nll(theta)
        g_sq = float(np.dot(g, g))
        if g_sq < 1e-20:
            break
        theta = theta - (residual / g_sq) * g
    return theta


def _rattle_step(
    nll_scalar,
    grad_nll,
    theta: np.ndarray,
    p: np.ndarray,
    threshold: float,
    step_size: float,
    tol: float,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Perform one RATTLE leapfrog step on the constraint manifold
    :math:`\mathrm{NLL}(\theta) = T`.

    The RATTLE algorithm :footcite:t:`andersen1983rattle` integrates
    constrained Hamiltonian dynamics with the Hamiltonian

    .. math::

        H(\theta, p) = \tfrac{1}{2}|p|^2 + \mathrm{NLL}(\theta),

    subject to the holonomic constraint
    :math:`C(\theta) = \mathrm{NLL}(\theta) - T = 0`.

    The update equations (one leapfrog step, step size :math:`\varepsilon`)
    are:

    .. math::

        p_{1/2} &= p_0
                  - \tfrac{\varepsilon}{2}\,\nabla\mathrm{NLL}(\theta_0), \\[3pt]
        \theta'  &= \theta_0 + \varepsilon\, p_{1/2}, \\[3pt]
        \theta_1 &= \texttt{project}(\theta'),
                   \quad \text{(SHAKE, Newton)} \\[3pt]
        p'       &= p_{1/2}
                  - \tfrac{\varepsilon}{2}\,\nabla\mathrm{NLL}(\theta_1), \\[3pt]
        p_1      &= p' - \frac{p' \cdot \nabla\mathrm{NLL}(\theta_1)}
                              {|\nabla\mathrm{NLL}(\theta_1)|^2}
                   \,\nabla\mathrm{NLL}(\theta_1).

    The final line projects the momentum onto the tangent space of the
    constraint at :math:`\theta_1`, ensuring
    :math:`p_1 \perp \nabla C(\theta_1)` as required for the next step.

    Args:
        nll_scalar: NLL function returning a plain ``float``.
        grad_nll: Gradient :math:`\nabla\mathrm{NLL}`, returning
            ``np.ndarray``.
        theta (``np.ndarray``): Current position on the constraint, shape
            ``(k,)``.
        p (``np.ndarray``): Current momentum in the tangent space, shape
            ``(k,)``.
        threshold (``float``): NLL threshold :math:`T`.
        step_size (``float``): Leapfrog step size :math:`\varepsilon`.
        tol (``float``): Newton convergence tolerance for constraint
            projection.
        max_iter (``int``): Maximum Newton iterations in projection.

    Returns:
        ``Tuple[np.ndarray, np.ndarray]``:
        ``(theta_new, p_new)`` — updated position on the constraint and
        tangent-space momentum.
    """
    eps = step_size
    g0 = grad_nll(theta)

    # Half-kick
    p_half = p - (eps / 2.0) * g0

    # Drift
    theta_prime = theta + eps * p_half

    # SHAKE: project theta_prime onto the constraint surface
    theta_new = _project_theta(
        nll_scalar, grad_nll, theta_prime, threshold, tol, max_iter
    )

    # Half-kick at new position
    g1 = grad_nll(theta_new)
    p_prime = p_half - (eps / 2.0) * g1

    # Project momentum to tangent space of constraint at theta_new
    g1_sq = float(np.dot(g1, g1))
    if g1_sq > 1e-20:
        p_new = p_prime - (float(np.dot(p_prime, g1)) / g1_sq) * g1
    else:
        p_new = p_prime

    return theta_new, p_new


def _rattle_walk(
    nll_scalar,
    grad_nll,
    start: np.ndarray,
    threshold: float,
    n_steps: int,
    step_size: float,
    tol: float,
    max_iter: int,
    rng: np.random.Generator,
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
) -> np.ndarray:
    r"""
    Walk along the constraint manifold from *start* using RATTLE.

    The chain begins by projecting *start* onto
    :math:`\mathrm{NLL}(\theta) = T` (in case it is slightly off-surface
    due to numerical error) and drawing an initial momentum uniformly from
    the tangent space.  At each step :func:`_rattle_step` is called; failed
    steps (numerical errors) terminate the chain early.

    The momentum is periodically re-sampled (every ``max(1, n_steps // 5)``
    steps) to prevent the chain from retracing its path and to improve
    coverage of the manifold.

    Args:
        nll_scalar: NLL function returning a plain ``float``.
        grad_nll: Gradient of NLL returning ``np.ndarray``.
        start (``np.ndarray``): Starting point (near the constraint),
            shape ``(k,)``.
        threshold (``float``): NLL threshold :math:`T`.
        n_steps (``int``): Total number of leapfrog steps.
        step_size (``float``): Leapfrog step size :math:`\varepsilon`.
        tol (``float``): Newton projection tolerance.
        max_iter (``int``): Maximum Newton iterations per step.
        rng (``np.random.Generator``): Random number generator.
        bounds (``Optional[List[Tuple[Optional[float], Optional[float]]]]``,
            default ``None``): Per-parameter ``(lower, upper)`` bounds in
            signal-parameter space.  ``None`` entries indicate no bound on
            that side.  If any element of ``theta`` violates a finite
            bound after a RATTLE step the chain is terminated early and
            the points collected so far are returned.

    Returns:
        ``np.ndarray``: Trajectory of contour points, shape
        ``(n_collected, k)`` where *n_collected* ≤ *n_steps* + 1.
    """
    k = len(start)

    # Project start onto constraint (handles small numerical drift)
    theta = _project_theta(nll_scalar, grad_nll, start.copy(), threshold, tol, max_iter)

    def _tangent_momentum(theta_: np.ndarray) -> np.ndarray:
        r"""
        Draw a random unit vector from the tangent space of the constraint
        at *theta_*:

        .. math::

            p \sim \mathcal{N}(0, I_k),
            \qquad
            p \leftarrow p - \frac{p \cdot \hat{n}}{\hat{n} \cdot \hat{n}} \hat{n},
            \quad \hat{n} = \nabla\mathrm{NLL}(\theta_).
        """
        g = grad_nll(theta_)
        g_sq = float(np.dot(g, g))
        raw = rng.standard_normal(size=k)
        if g_sq > 1e-20:
            raw = raw - (float(np.dot(raw, g)) / g_sq) * g
        n = np.linalg.norm(raw)
        return raw / n if n > 1e-15 else raw

    p = _tangent_momentum(theta)
    resample_every = max(1, n_steps // 5)

    trajectory = [theta.copy()]
    for step in range(n_steps):
        # Periodic momentum re-sampling to improve manifold exploration
        if step > 0 and step % resample_every == 0:
            p = _tangent_momentum(theta)

        try:
            theta, p = _rattle_step(
                nll_scalar,
                grad_nll,
                theta,
                p,
                threshold,
                step_size,
                tol,
                max_iter,
            )
        except Exception:  # noqa: BLE001
            log.debug("RATTLE step failed at step %d; terminating chain.", step)
            break

        # Terminate if any parameter has left its allowed range
        if bounds is not None:
            if any(
                (lo is not None and theta[i] < lo) or (hi is not None and theta[i] > hi)
                for i, (lo, hi) in enumerate(bounds)
            ):
                log.debug(
                    "RATTLE chain left parameter bounds at step %d; terminating.", step
                )
                break

        trajectory.append(theta.copy())

    return np.array(trajectory, dtype=float)


def _run_hmc_chains(
    nll_scalar,
    grad_nll,
    threshold: float,
    starts: np.ndarray,
    n_steps: int,
    step_size: float,
    tol: float,
    max_iter: int,
    rng: np.random.Generator,
    n_jobs: int = 1,
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
) -> np.ndarray:
    r"""
    Launch one RATTLE chain per provided starting point.

    Chains are independent and dispatched to *n_jobs* worker processes via
    ``joblib.Parallel``.  Each worker receives its own
    :class:`numpy.random.Generator` seeded from *rng* before forking so
    that results are reproducible while avoiding shared-state race conditions.

    Args:
        nll_scalar: NLL function returning a plain ``float``.
        grad_nll: Gradient of NLL returning ``np.ndarray``.
        threshold (``float``): NLL threshold :math:`T`.
        starts (``np.ndarray``): Starting points for each chain, shape
            ``(n_chains, k)``.  Every row should lie on (or very close to)
            the contour :math:`\mathrm{NLL}(\theta) = T`; a Newton
            projection step at the beginning of :func:`_rattle_walk` corrects
            any small numerical drift.
        n_steps (``int``): RATTLE steps per chain.
        step_size (``float``): Leapfrog step size.
        tol (``float``): Newton tolerance.
        max_iter (``int``): Maximum Newton iterations.
        rng (``np.random.Generator``): Random number generator used to
            derive per-chain seeds.
        n_jobs (``int``, default ``1``): Number of parallel worker processes.
        bounds (``Optional[List[Tuple[Optional[float], Optional[float]]]]``,
            default ``None``): Per-parameter ``(lower, upper)`` bounds
            forwarded to :func:`_rattle_walk`.

    Returns:
        ``np.ndarray``: All RATTLE trajectory points concatenated, shape
        ``(n_total, k)``.  Empty ``(0, k)`` array if no chains produce
        points.
    """
    if len(starts) == 0:
        k = starts.shape[1] if starts.ndim == 2 and starts.shape[1] > 0 else 1
        return np.empty((0, k), dtype=float)

    n, k = starts.shape

    # Pre-generate per-chain seeds from the main rng *before* forking so
    # the rng state advances deterministically regardless of n_jobs.
    seeds: np.ndarray = np.array(
        [int(rng.integers(2**63)) for _ in range(n)], dtype=int
    )

    def _one_chain(starts: np.ndarray, seeds: np.ndarray, idx: int) -> np.ndarray:
        start = starts[idx]
        seed = seeds[idx]
        chain_rng = np.random.default_rng(seed)
        return _rattle_walk(
            nll_scalar,
            grad_nll,
            start,
            threshold,
            n_steps,
            step_size,
            tol,
            max_iter,
            chain_rng,
            bounds,
        )

    loop = tqdm(
        iterable=range(n),
        total=n,
        desc="Running HMC Chain",
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
    )

    try:
        trajs = Parallel(
            n_jobs=n_jobs, prefer="processes", batch_size="auto", max_nbytes="10M"
        )(delayed(_one_chain)(starts, seeds, idx) for idx in loop)
    except Exception as err:
        log.debug("An error has occured: %s", err)
        trajs = [_one_chain(starts, seeds, idx) for idx in loop]

    all_trajs = [t for t in trajs if len(t) > 0]
    if not all_trajs:
        return np.empty((0, k), dtype=float)
    return np.vstack(all_trajs)
