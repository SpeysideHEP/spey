from typing import Callable, List, Tuple, Union
import logging
import numpy as np

from .model_config import ModelConfig

log = logging.getLogger("Spey")


def resolve_parameter_index(parameter: Union[int, str], cfg: ModelConfig) -> int:
    r"""
    Validate and convert a nuisance-parameter identifier to its integer index.

    Args:
        parameter (``int`` or ``str``): Parameter index or name.
        cfg (:obj:`~spey.base.model_config.ModelConfig`): Model configuration.

    Raises:
        :obj:`ValueError`: If the model has fewer than 2 parameters, if the index
            is out of range, if the index refers to the POI, or if the name is not
            found in :attr:`~spey.base.model_config.ModelConfig.parameter_names`.

    Returns:
        ``int``: Resolved parameter index.
    """
    if cfg.npar < 2:
        raise ValueError(
            "Nuisance-parameter profiling requires at least 2 model parameters "
            f"(POI + at least one nuisance), but this model has only {cfg.npar}."
        )
    if isinstance(parameter, str):
        if cfg.parameter_names is None:
            raise ValueError(
                "Cannot resolve parameter name: parameter_names is not set in ModelConfig."
            )
        if parameter not in cfg.parameter_names:
            raise ValueError(
                f"Parameter '{parameter}' not found in parameter_names: {cfg.parameter_names}"
            )
        return cfg.parameter_names.index(parameter)
    param_idx = int(parameter)
    if not 0 <= param_idx < cfg.npar:
        raise ValueError(f"Parameter index {param_idx} is out of range [0, {cfg.npar}).")
    if param_idx == cfg.poi_index:
        raise ValueError(
            f"Parameter index {param_idx} refers to the primary POI. "
            "Leave parameter=None to profile the POI instead."
        )
    return param_idx


def _refine_global_min_1d(
    nll_fn: Callable[[float], float],
    mllhd: float,
    reference: float,
    lo: float,
    hi: float,
    n_samples: int = 9,
) -> Tuple[float, float]:
    r"""
    Re-anchor a 1D NLL minimum against a coarse multi-start scan.

    When the optimiser behind :meth:`HypothesisTestingBase.maximize_likelihood`
    converges to a strictly-local basin, ``mllhd`` is above the true global
    minimum.  Using that value as the :math:`\chi^2` anchor makes the profile
    :math:`2(\mathrm{NLL}(\theta)-\mathrm{NLL}_{\min})` negative at the other
    basin's minimum — i.e. the threshold is too low and the confidence region
    shrinks (or disappears).  A cheap evenly-spaced scan of ``nll_fn`` on
    ``[lo, hi]`` detects any lower minimum; a bounded scalar minimisation
    around the best scan point refines it.  The returned ``(mllhd, reference)``
    is guaranteed to satisfy ``mllhd <= nll_fn(x)`` for every point sampled so
    far, up to numerical noise.

    Args:
        nll_fn: 1D negative log-likelihood function.
        mllhd: NLL value at the current (possibly local) estimate.
        reference: Argument value associated with ``mllhd`` — used downstream
            to split two-sided intervals into left/right roots.
        lo, hi: Search range for the scan.
        n_samples: Number of evenly-spaced evaluations of ``nll_fn`` on
            ``[lo, hi]``.  Values below 2 are clamped.

    Returns:
        ``Tuple[float, float]`` — the re-anchored ``(mllhd, reference)``.
    """
    import scipy.optimize as _opt  # noqa: PLC0415

    if not (hi > lo) or n_samples < 2:
        return mllhd, reference

    xs = np.linspace(float(lo), float(hi), int(n_samples))
    best_x, best_f = reference, mllhd
    for x in xs:
        try:
            fv = float(nll_fn(float(x)))
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(fv):
            continue
        if fv < best_f:
            best_x, best_f = float(x), fv

    # Only re-anchor if the scan found a genuinely lower minimum.
    if best_f >= mllhd - 1e-9:
        return mllhd, reference

    # Refine around the best scan point with a bounded 1D minimiser so the
    # new anchor is accurate to machine precision, not just the scan step.
    width = (float(hi) - float(lo)) / max(1, int(n_samples) - 1)
    lo2 = max(float(lo), best_x - width)
    hi2 = min(float(hi), best_x + width)
    if hi2 > lo2:
        try:
            res = _opt.minimize_scalar(
                lambda v: float(nll_fn(float(v))),
                bounds=(lo2, hi2),
                method="bounded",
                options={"xatol": 1e-10},
            )
            if np.isfinite(res.fun) and float(res.fun) < best_f:
                best_x, best_f = float(res.x), float(res.fun)
        except Exception as exc:  # noqa: BLE001
            log.debug("_refine_global_min_1d minimize_scalar failed: %s", exc)

    log.debug(
        "_refine_global_min_1d: re-anchored NLL_min %.9g -> %.9g at arg %.9g",
        mllhd,
        best_f,
        best_x,
    )
    return best_f, best_x


def _enumerate_crossings_1d(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    n_scan: int = 121,
    xtol: float = 2e-12,
    rtol: float = 1e-4,
    maxiter: int = 10000,
) -> List[float]:
    r"""
    Return **every** root of a 1D scalar function on ``[lo, hi]``.

    A uniform grid of ``n_scan`` points is evaluated; each interval that shows
    a sign change is refined with :func:`scipy.optimize.toms748` (falling back
    to :func:`scipy.optimize.brentq` if TOMS-748 stumbles on near-tangent
    roots).  Results are deduplicated and returned in ascending order.

    This is the correct shape for bracketing the boundary of a non-convex
    confidence region: every endpoint of every disjoint interval shows up as
    a sign change and gets refined independently, unlike a single
    bracket-and-solve pass which can only ever return one root per side.

    Args:
        f: 1D scalar function.
        lo, hi: Search range.
        n_scan: Number of grid points (clamped to ``>= 3``).  The effective
            resolution needs to resolve the narrowest basin; if a suspected
            feature is narrower than ``(hi-lo)/(n_scan-1)`` the feature may
            be skipped.
        xtol, rtol, maxiter: Forwarded to the inner root-finder.

    Returns:
        ``List[float]`` — sorted, deduplicated root list (possibly empty).
    """
    import scipy.optimize as _opt  # noqa: PLC0415

    if not (hi > lo):
        return []
    n_scan = max(3, int(n_scan))

    xs = np.linspace(float(lo), float(hi), n_scan)
    fs = np.full(n_scan, np.nan, dtype=float)
    for i, x in enumerate(xs):
        try:
            fs[i] = float(f(float(x)))
        except Exception:  # noqa: BLE001
            fs[i] = np.nan

    roots: List[float] = []
    for i in range(n_scan - 1):
        fa, fb = fs[i], fs[i + 1]
        if not (np.isfinite(fa) and np.isfinite(fb)):
            continue
        # Exact hits land on the grid — keep the gridpoint itself.
        if fa == 0.0:
            roots.append(float(xs[i]))
            continue
        if fb == 0.0 and i == n_scan - 2:
            roots.append(float(xs[i + 1]))
            continue
        if fa * fb >= 0.0:
            continue
        # Sign change — refine.
        a, b = float(xs[i]), float(xs[i + 1])
        try:
            x0, r = _opt.toms748(
                f,
                a,
                b,
                k=2,
                xtol=xtol,
                rtol=rtol,
                full_output=True,
                maxiter=maxiter,
            )
            if r.converged:
                roots.append(float(x0))
                continue
        except Exception as exc:  # noqa: BLE001
            log.debug("toms748 failed on [%s, %s]: %s", a, b, exc)
        try:
            x0 = _opt.brentq(f, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter)
            roots.append(float(x0))
        except Exception as exc:  # noqa: BLE001
            log.debug("brentq fallback failed on [%s, %s]: %s", a, b, exc)

    # Deduplicate nearby hits (rare but possible when the grid catches a root
    # exactly at a gridpoint AND an adjacent sign-change is refined onto it).
    out: List[float] = []
    for r in sorted(roots):
        if not out or abs(r - out[-1]) > max(1e-10, 1e-8 * abs(r)):
            out.append(r)
    return out
