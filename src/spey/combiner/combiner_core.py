r"""
Combiner Core
=============

Shared machinery of the combiner plug-ins.  :class:`CombinerBase` is a
:class:`~spey.BackendBase` that joins an arbitrary collection of
:class:`~spey.StatisticalModel` objects into **one** likelihood with a single global
parameter vector.  It implements everything a combination needs — likelihood,
gradient, Hessian, Asimov data, sampling, constraint handling and introspection — and
leaves exactly one decision to its subclasses: *which local parameters are the same
global parameter* (:meth:`CombinerBase._build_parameter_map`).

* :class:`~spey.combiner.UncorrelatedStatisticsCombiner`
  (``default.uncorrelated_combiner``) identifies only the parameters of interest.
* :class:`~spey.combiner.CorrelatedStatisticsCombiner`
  (``default.correlated_combiner``) identifies any user-declared parameters.

Mathematical Background
-----------------------

**Parameter identification as a gather map**

Let the combination contain :math:`M` models.  Model :math:`m` has a local parameter
vector :math:`\boldsymbol{\theta}_m\in\mathbb{R}^{n_m}`.  The combiner constructs a
global vector :math:`\mathbf{p}\in\mathbb{R}^{N}` together with an index map
:math:`I_m\in\{0,\dots,N-1\}^{n_m}` such that

.. math::

    \theta_{m,j} = p_{I_m[j]}\ ,\qquad j = 0,\dots,n_m-1\ .

Two models share a parameter exactly when their index maps point at the same global
slot.  All parameters that are not shared receive a private slot, so
:math:`N \le \sum_m n_m` with equality when nothing is shared.

**Joint likelihood**

The observations of distinct analyses are conditionally independent *given* the
parameters, so the joint likelihood factorises over models even though the models
are correlated through the shared parameters:

.. math::

    \ln\mathcal{L}(\mathbf{p})
    = \sum_{m=1}^{M} \ln\mathcal{L}_m\!\left(\mathbf{p}[I_m]\right).

Correlations between analyses are induced *entirely* by the identification
:math:`\theta_{m,j} = p_{I_m[j]}`; the parameters of a shared slot are profiled
**once, jointly**, rather than independently per analysis.

**Gradient and Hessian — the chain rule**

The global-to-local map is a pure gather, hence its Jacobian is a selection matrix,

.. math::

    \frac{\partial \theta_{m,j}}{\partial p_a} = \delta_{I_m[j],\,a}\ ,

and the chain rule collapses into a scatter-add.  With
:math:`g_{m,j} = \partial\ln\mathcal{L}_m/\partial\theta_{m,j}` and
:math:`H^{(m)}_{jk} = \partial^2\ln\mathcal{L}_m/\partial\theta_{m,j}\partial\theta_{m,k}`,

.. math::

    \frac{\partial \ln\mathcal{L}}{\partial p_a}
    = \sum_{m}\sum_{j} g_{m,j}\,\delta_{I_m[j],\,a}\ ,
    \qquad
    \frac{\partial^2 \ln\mathcal{L}}{\partial p_a \partial p_b}
    = \sum_{m}\sum_{j,k} H^{(m)}_{jk}\,
      \delta_{I_m[j],\,a}\,\delta_{I_m[k],\,b}\ .

No Jacobian matrix is ever formed or multiplied: the gradient is accumulated with a
single :func:`numpy.bincount` over the concatenated index map, and the Hessian by
adding each :math:`n_m \times n_m` block into the ``np.ix_(I_m, I_m)`` sub-block of
the global matrix.  The cost is :math:`\mathcal{O}(\sum_m n_m)` and
:math:`\mathcal{O}(\sum_m n_m^2)` respectively, on top of the backends' own work.

Automatic differentiation
-------------------------

Constituent backends may compute their derivatives with *any* framework —
:mod:`autograd`, ``jax``, ``tensorflow``, ``pytorch``, or closed-form expressions.
The combiner never differentiates through a backend; it only consumes the
derivatives that the backend advertises through
:func:`~spey.BackendBase.get_objective_function` (with ``do_grad=True``) and
:func:`~spey.BackendBase.get_hessian_logpdf_func`, and converts the returned
objects to :obj:`numpy.ndarray` (see :func:`_to_numpy`).

So that ``spey`` components which differentiate the log-pdf *externally* keep
working — most importantly :func:`~spey.multiparameter.find_contour`, which applies
:func:`autograd.grad` to the callable returned by
:func:`~spey.BackendBase.get_logpdf_func` — the combined log-pdf is exposed as an
:func:`autograd.extend.primitive` whose vector-Jacobian products are wired to the
combined gradient and Hessian assembled above.  The primitive also acts as a
**barrier** between ``spey``'s :mod:`autograd` tracing and whichever framework a
backend uses internally: a backend is only ever called with a concrete
``float64`` :obj:`numpy.ndarray`, never with an :mod:`autograd` box.

.. attention::

    Differentiability is a property of the *whole* combination: if a single
    constituent model cannot supply a gradient (or a Hessian), the combined model
    reports the corresponding method as unavailable by raising
    :obj:`NotImplementedError`, and ``spey`` transparently falls back to its
    numerical optimiser.

Writing a new combiner
----------------------

A new combination prescription only has to decide the index maps.  Subclass
:class:`CombinerBase`, add the plug-in metadata, and implement
:meth:`~CombinerBase._build_parameter_map`; :meth:`~CombinerBase._allocate_slots`
turns a ``{(model position, local index): group}`` mapping into index maps:

.. code-block:: python3

    class ShareEverythingNamedMu(CombinerBase):
        name = "my_package.mu_combiner"

        def _build_parameter_map(self, local_names):
            groups = {
                (pos, names.index("mu")): "mu"
                for pos, names in enumerate(local_names)
            }
            maps, slot_of = self._allocate_slots(local_names, groups)
            return maps, {slot_of["mu"]: "mu"}

.. versionadded:: 0.2.8

References
----------
* G. Cowan, K. Cranmer, E. Gross, O. Vitells, *Asymptotic formulae for
  likelihood-based tests of new physics*, Eur. Phys. J. C **71** (2011) 1554,
  :xref:`1007.1727`.
"""

import logging
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from autograd.extend import defvjp, primitive

from spey._version import __version__
from spey.base.backend_base import BackendBase
from spey.base.model_config import ModelConfig
from spey.interface.statistical_model import StatisticalModel, _is_implemented
from spey.system.exceptions import AnalysisQueryError, InvalidInput
from spey.utils import ExpectationType

__all__ = ["CombinerBase"]

log = logging.getLogger("Spey")

# pylint: disable=W1203

#: Identifier of a constituent model: its :attr:`~spey.StatisticalModel.analysis`
#: name or its position within the combination.
ModelKey = Union[str, int]

#: Data accepted by the combined likelihood: a flat array laid out in model order, or
#: a mapping from model key to that model's data.
CombinedData = Union[List[float], np.ndarray, Dict[ModelKey, Any]]


def __dir__():
    return __all__


def _to_numpy(value: Any) -> np.ndarray:
    r"""
    Convert a value produced by an arbitrary autodiff framework to a ``float64`` array.

    Constituent backends may return :obj:`numpy.ndarray`, ``jax`` arrays,
    ``tensorflow`` tensors, or ``pytorch`` tensors (possibly carrying a gradient tape
    or living on an accelerator).  This helper normalises all of them:

    * ``pytorch`` tensors are detached from the autograd graph and moved to the host,
    * objects exposing ``.numpy()`` (``tensorflow``, ``pytorch``) are converted,
    * everything else is passed to :func:`numpy.asarray`, which covers ``jax``
      arrays and any object implementing ``__array__``.

    Args:
        value (``Any``): Object returned by a backend.

    Returns:
        ``np.ndarray``:
        Contiguous ``float64`` array view of ``value``.
    """
    if isinstance(value, np.ndarray):
        return value if value.dtype == np.float64 else value.astype(np.float64)

    # pytorch: detach from the tape and move off the accelerator before conversion.
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    to_host = getattr(value, "cpu", None)
    if callable(to_host):
        value = to_host()

    if not isinstance(value, np.ndarray):
        as_numpy = getattr(value, "numpy", None)
        if callable(as_numpy):
            try:
                value = as_numpy()
            except (TypeError, RuntimeError):  # pragma: no cover - framework specific
                log.debug("`.numpy()` conversion failed, falling back to np.asarray")

    return np.asarray(value, dtype=np.float64)


def _to_float(value: Any) -> float:
    """
    Convert a scalar produced by an arbitrary autodiff framework to a ``float``.

    Args:
        value (``Any``): Scalar-like object returned by a backend.

    Returns:
        ``float``:
        Plain Python float.
    """
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - framework specific
        return float(_to_numpy(value).reshape(()))


def _intersect_bounds(
    bounds: List[Optional[Tuple[Optional[float], Optional[float]]]],
    name: str,
) -> Tuple[Optional[float], Optional[float]]:
    r"""
    Intersect the optimiser bounds declared by every contributor of a shared slot.

    A shared parameter must satisfy *all* of the constraints its owners impose, so
    the joint bound is :math:`[\max_i \ell_i,\ \min_i u_i]` with ``None`` treated as
    an open side.

    Args:
        bounds (``List[Optional[Tuple[Optional[float], Optional[float]]]]``): Bounds
          declared by each contributing model for this slot.
        name (``str``): Global parameter name, used in the error message.

    Raises:
        ``~spey.system.exceptions.InvalidInput``: If the intersection is empty.

    Returns:
        ``Tuple[Optional[float], Optional[float]]``:
        Intersected ``(lower, upper)`` bound.
    """
    lows = [b[0] for b in bounds if b is not None and b[0] is not None]
    highs = [b[1] for b in bounds if b is not None and b[1] is not None]
    low = max(lows) if lows else None
    high = min(highs) if highs else None
    if low is not None and high is not None and low > high:
        raise InvalidInput(
            f"Bounds of the shared parameter '{name}' do not overlap: "
            f"the intersection of {bounds} is empty."
        )
    return (low, high)


def _has_gradient(model: StatisticalModel) -> bool:
    """
    Whether a model's backend advertises a gradient.

    Args:
        model (~spey.StatisticalModel): Model to probe.

    Returns:
        ``bool``
    """
    try:
        model.backend.get_objective_function(do_grad=True)
    except NotImplementedError:
        return False
    return True


def _has_hessian(model: StatisticalModel) -> bool:
    """
    Whether a model's backend advertises a Hessian.

    Args:
        model (~spey.StatisticalModel): Model to probe.

    Returns:
        ``bool``
    """
    try:
        model.backend.get_hessian_logpdf_func()
    except NotImplementedError:
        return False
    return True


class CombinerBase(BackendBase):
    r"""
    Base class of the combiner plug-ins: joins :class:`~spey.StatisticalModel` objects
    into a single backend with one global parameter vector.

    The joint log-likelihood is

    .. math::

        \ln\mathcal{L}(\mathbf{p})
        = \sum_{m} \ln\mathcal{L}_m\!\left(\mathbf{p}[I_m]\right),

    where :math:`I_m` is the gather map from the global parameter vector into the
    local parameter vector of model :math:`m` (see the module documentation for the
    derivation of the gradient and Hessian).  Subclasses decide the gather maps by
    implementing :meth:`_build_parameter_map`; everything else is provided here.

    Each constituent model keeps its own backend, its own data, and its own
    likelihood prescription, and the backends do not need to be of the same type.

    Args:
        statistical_models (``Sequence[~spey.StatisticalModel]``): Models to combine.
          Their :attr:`~spey.StatisticalModel.analysis` identifiers must be unique.

    Raises:
        ``TypeError``: If an entry of ``statistical_models`` is not a
          :class:`~spey.StatisticalModel`.
        ``~spey.system.exceptions.AnalysisQueryError``: If two models share the same
          :attr:`~spey.StatisticalModel.analysis` identifier.
        ``~spey.system.exceptions.InvalidInput``: If ``statistical_models`` is empty,
          or if the bounds of a shared parameter do not overlap.
    """

    version: str = __version__
    """Version of the backend"""
    author: str = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: str = __version__
    """Spey version required for the backend"""

    __slots__ = [
        "_models",
        "_configs",
        "_maps",
        "_block_index",
        "_flat_index",
        "_duplicated",
        "_npar",
        "_config",
        "_data_widths",
        "_gradient_available",
        "_hessian_available",
        "constraints",
    ]

    def __init__(self, statistical_models: Sequence[StatisticalModel]):
        self._models: Tuple[StatisticalModel, ...] = self._validate_models(
            statistical_models
        )
        self._configs: List[ModelConfig] = [
            model.backend.config() for model in self._models
        ]

        local_names = [
            self._local_parameter_names(cfg, model.analysis)
            for cfg, model in zip(self._configs, self._models)
        ]
        maps, shared_names = self._build_parameter_map(local_names)
        self._set_index_maps(maps)
        self._build_config(shared_names, local_names)

        self._data_widths: Optional[List[int]] = None
        """Number of data points each model consumes; filled on first use."""

        self._gradient_available, self._hessian_available = self._probe_derivatives()
        self._disable_unsupported_capabilities()
        self.constraints = self._lift_constraints()
        """Constraints of the constituent models, lifted to the global parameter vector"""

        log.debug(
            "Combined %d models into %d parameters: %s",
            len(self._models),
            self._npar,
            self._config.parameter_names,
        )

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_parameter_map(
        self, local_names: List[List[str]]
    ) -> Tuple[List[np.ndarray], Dict[int, str]]:
        """
        Decide which local parameters are identified with each other.

        This is the only method a combiner has to implement.  It is called during
        construction, after :attr:`statistical_models` and the constituent
        configurations are available; :meth:`_allocate_slots` builds the index maps
        from a grouping of local parameters.

        Args:
            local_names (``List[List[str]]``): Local parameter names per model, see
              :meth:`_local_parameter_names`.

        Returns:
            ``Tuple[List[np.ndarray], Dict[int, str]]``:
            The gather map :math:`I_m` of every model, in model order, and the global
            name of every shared slot keyed by slot index.
        """

    def _select_poi(self, names: List[str]) -> int:
        """
        Choose the global index of the combined parameter of interest.

        Defaults to the global slot holding the POI of the first model.

        Args:
            names (``List[str]``): Global parameter names.

        Returns:
            ``int``:
            Global index of the parameter of interest.
        """
        return int(self._maps[0][self._configs[0].poi_index])

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_models(
        statistical_models: Sequence[StatisticalModel],
    ) -> Tuple[StatisticalModel, ...]:
        """
        Check the constituent models and freeze them into a tuple.

        Args:
            statistical_models (``Sequence[~spey.StatisticalModel]``): Models to combine;
              a single model is accepted as well.

        Raises:
            ``TypeError``: If an entry is not a :class:`~spey.StatisticalModel`.
            ``~spey.system.exceptions.AnalysisQueryError``: If two analysis identifiers
              coincide.
            ``~spey.system.exceptions.InvalidInput``: If no model is given.

        Returns:
            ``Tuple[~spey.StatisticalModel, ...]``:
            The constituent models in the order they were supplied.
        """
        if isinstance(statistical_models, StatisticalModel):
            statistical_models = [statistical_models]
        models = tuple(statistical_models)
        if len(models) == 0:
            raise InvalidInput("At least one statistical model is required.")
        for model in models:
            if not isinstance(model, StatisticalModel):
                raise TypeError(
                    f"Can not combine type {type(model)}; expected spey.StatisticalModel."
                )
        analyses = [model.analysis for model in models]
        if len(set(analyses)) != len(analyses):
            raise AnalysisQueryError(
                f"Analysis identifiers have to be unique, got {analyses}."
            )
        return models

    @staticmethod
    def _local_parameter_names(cfg: ModelConfig, analysis: str) -> List[str]:
        r"""
        Return the local parameter names of a model, synthesising them when absent.

        Backends are not required to declare
        :attr:`~spey.base.model_config.ModelConfig.parameter_names`.  When they do
        not, the parameter of interest — which the backend *does* identify through
        :attr:`~spey.base.model_config.ModelConfig.poi_index` — is named ``"mu"``,
        following ``spey``'s own convention, so that a POI can be referred to
        uniformly across backends.  Every other parameter falls back to
        ``"<analysis>_par_<index>"``.

        Args:
            cfg (:obj:`~spey.base.model_config.ModelConfig`): Configuration of the model.
            analysis (``str``): Analysis identifier, used to build fallback names.

        Returns:
            ``List[str]``:
            One name per local parameter.
        """
        if cfg.parameter_names is not None and len(cfg.parameter_names) == cfg.npar:
            return list(cfg.parameter_names)
        log.debug(
            "`%s` does not declare parameter_names; only its POI ('mu') can be "
            "referred to by name, every other parameter has to be referred to by index.",
            analysis,
        )
        return [
            "mu" if idx == cfg.poi_index else f"{analysis}_par_{idx}"
            for idx in range(cfg.npar)
        ]

    @staticmethod
    def _allocate_slots(
        local_names: List[List[str]], group_of: Dict[Tuple[int, int], Hashable]
    ) -> Tuple[List[np.ndarray], Dict[Hashable, int]]:
        """
        Allocate global parameter slots and build the per-model gather maps.

        Slots are created in first-occurrence order — model ``0``'s parameters first —
        so that the global POI is normally index ``0``.  Local parameters mapped to the
        same group share one slot; every other local parameter receives a private one.

        Args:
            local_names (``List[List[str]]``): Local parameter names per model; only
              their lengths are used.
            group_of (``Dict[Tuple[int, int], Hashable]``): Group label of every shared
              ``(model position, local parameter index)`` pair.

        Returns:
            ``Tuple[List[np.ndarray], Dict[Hashable, int]]``:
            The gather map of every model and the global slot of every group.
        """
        slot_of_group: Dict[Hashable, int] = {}
        maps: List[np.ndarray] = []
        n_slots = 0

        for pos, names in enumerate(local_names):
            index_map = np.empty(len(names), dtype=int)
            for par in range(len(names)):
                group = group_of.get((pos, par))
                if group is None:
                    index_map[par] = n_slots
                    n_slots += 1
                elif group in slot_of_group:
                    index_map[par] = slot_of_group[group]
                else:
                    slot_of_group[group] = n_slots
                    index_map[par] = n_slots
                    n_slots += 1
            maps.append(index_map)

        return maps, slot_of_group

    def _set_index_maps(self, maps: List[np.ndarray]) -> None:
        """
        Store the gather maps and precompute everything the hot path needs.

        That is the maps themselves, their ``np.ix_`` form, the concatenated index used
        for the gradient scatter, and whether any map repeats a slot.

        Args:
            maps (``List[np.ndarray]``): Gather map of every model, in model order.

        Raises:
            ``~spey.system.exceptions.InvalidInput``: If the maps do not have one entry
              per local parameter, or do not cover every global slot ``0 … N-1``.
        """
        self._maps: List[np.ndarray] = [np.asarray(m, dtype=int) for m in maps]
        if len(self._maps) != len(self._models) or any(
            index_map.size != cfg.npar
            for index_map, cfg in zip(self._maps, self._configs)
        ):
            raise InvalidInput(
                "Every model needs one index-map entry per local parameter."
            )
        self._flat_index: np.ndarray = (
            np.concatenate(self._maps) if self._maps else np.empty(0, dtype=int)
        )
        self._npar: int = int(self._flat_index.max()) + 1 if self._flat_index.size else 0
        if np.unique(self._flat_index).size != self._npar or self._flat_index.min() < 0:
            raise InvalidInput(
                "Index maps have to cover every global parameter slot 0 ... N-1 "
                "without gaps."
            )
        self._block_index: List[Tuple[np.ndarray, np.ndarray]] = [
            np.ix_(index_map, index_map) for index_map in self._maps
        ]
        # A model may in principle map two of its own parameters onto one global
        # slot; that needs the (slower) accumulating scatter.
        self._duplicated: List[bool] = [
            len(np.unique(index_map)) != len(index_map) for index_map in self._maps
        ]

    def _build_config(
        self, shared_names: Dict[int, str], local_names: List[List[str]]
    ) -> None:
        """
        Assemble the global :class:`~spey.base.model_config.ModelConfig`.

        Initial values are taken from the first contributor of each slot, bounds are
        intersected over all contributors (a shared parameter must satisfy every
        owner's constraint), a slot is fixed if any contributor fixes it, and the
        minimum POI is the largest of the minima declared by the models that
        contribute their own POI to the combined POI slot.

        Args:
            shared_names (``Dict[int, str]``): Global name of every shared slot.
            local_names (``List[List[str]]``): Local parameter names per model.

        Raises:
            ``~spey.system.exceptions.InvalidInput``: If the bounds of a shared
              parameter do not overlap, or if :meth:`_select_poi` rejects the request.
        """
        contributors: List[List[Tuple[int, int]]] = [[] for _ in range(self._npar)]
        for pos, index_map in enumerate(self._maps):
            for par, slot in enumerate(index_map):
                contributors[slot].append((pos, par))

        names = self._global_names(contributors, shared_names, local_names)

        suggested_init: List[float] = []
        suggested_bounds: List[Tuple[Optional[float], Optional[float]]] = []
        suggested_fixed: List[bool] = []
        has_fixed = False
        for slot, owners in enumerate(contributors):
            first_pos, first_par = owners[0]
            suggested_init.append(
                float(self._configs[first_pos].suggested_init[first_par])
            )
            suggested_bounds.append(
                _intersect_bounds(
                    [self._configs[pos].suggested_bounds[par] for pos, par in owners],
                    names[slot],
                )
            )
            fixed = False
            for pos, par in owners:
                model_fixed = self._configs[pos].suggested_fixed
                if model_fixed is not None:
                    has_fixed = True
                    fixed = fixed or bool(model_fixed[par])
            suggested_fixed.append(fixed)

        global_poi = self._select_poi(names)
        minimum_poi = max(
            (
                self._configs[pos].minimum_poi
                for pos, par in contributors[global_poi]
                if self._configs[pos].poi_index == par
            ),
            default=-np.inf,
        )

        self._config = ModelConfig(
            poi_index=global_poi,
            minimum_poi=float(minimum_poi),
            suggested_init=suggested_init,
            suggested_bounds=suggested_bounds,
            parameter_names=names,
            suggested_fixed=suggested_fixed if has_fixed else None,
        )

    def _global_names(
        self,
        contributors: List[List[Tuple[int, int]]],
        shared_names: Dict[int, str],
        local_names: List[List[str]],
    ) -> List[str]:
        """
        Build unique global parameter names.

        A shared slot takes its group name.  A private slot keeps its local name when
        that name is unambiguous and is otherwise qualified with the analysis
        identifier (``"<analysis>::<local name>"``).

        Args:
            contributors (``List[List[Tuple[int, int]]]``): ``(model, parameter)``
              pairs feeding each slot.
            shared_names (``Dict[int, str]``): Group name for each shared slot.
            local_names (``List[List[str]]``): Local parameter names per model.

        Returns:
            ``List[str]``:
            One unique name per global parameter.
        """
        candidates: List[str] = []
        for slot, owners in enumerate(contributors):
            if slot in shared_names:
                candidates.append(shared_names[slot])
            else:
                pos, par = owners[0]
                candidates.append(local_names[pos][par])

        counts: Dict[str, int] = {}
        for candidate in candidates:
            counts[candidate] = counts.get(candidate, 0) + 1

        names: List[str] = []
        for slot, candidate in enumerate(candidates):
            if counts[candidate] == 1 or slot in shared_names:
                names.append(candidate)
            else:
                pos, par = contributors[slot][0]
                names.append(f"{self._models[pos].analysis}::{local_names[pos][par]}")

        # Extremely defensive: qualification can only clash if two models had the
        # same analysis name, which the constructor already rejects.
        seen: Dict[str, int] = {}
        for idx, candidate in enumerate(names):
            if candidate in seen:
                seen[candidate] += 1
                names[idx] = f"{candidate}_{seen[candidate]}"
            else:
                seen[candidate] = 0
        return names

    def _resolve_model_key(self, key: ModelKey) -> int:
        """
        Resolve an analysis name or position into a model position.

        Args:
            key (``Union[str, int]``): Analysis identifier or positional index.

        Raises:
            ``~spey.system.exceptions.AnalysisQueryError``: If the key matches no model.

        Returns:
            ``int``:
            Position of the model within the combination.
        """
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            if not 0 <= int(key) < len(self._models):
                raise AnalysisQueryError(
                    f"Model index {key} is out of range for {len(self._models)} models."
                )
            return int(key)
        for pos, model in enumerate(self._models):
            if model.analysis == key:
                return pos
        raise AnalysisQueryError(f"'{key}' is not among the analyses: {self.analyses}.")

    def _probe_derivatives(self) -> Tuple[bool, bool]:
        """
        Determine whether every constituent model can supply a gradient and a Hessian.

        The probe only *builds* the backend closures — it never evaluates them — so it
        is cheap and free of side effects.

        Returns:
            ``Tuple[bool, bool]``:
            Whether the gradient and the Hessian are available for the combination.
        """
        gradient, hessian = True, True
        for model in self._models:
            if not _has_gradient(model):
                log.debug("`%s` does not provide a gradient.", model.analysis)
                gradient = False
            if not _has_hessian(model):
                log.debug("`%s` does not provide a Hessian.", model.analysis)
                hessian = False
        return gradient, hessian

    def _disable_unsupported_capabilities(self) -> None:
        """
        Hide optional methods that the combination can not actually perform.

        :func:`expected_data` and :func:`get_sampler` are only meaningful when *every*
        constituent model implements them.  When one does not, the corresponding
        method is replaced on this instance by the unimplemented
        :class:`~spey.BackendBase` version, so that
        :attr:`~spey.StatisticalModel.is_asymptotic_calculator_available` and
        :attr:`~spey.StatisticalModel.is_toy_calculator_available` report the
        combination's real capabilities instead of promising a calculator that would
        fail on first use.
        """
        for method, fallback in (
            ("expected_data", BackendBase.expected_data),
            ("get_sampler", BackendBase.get_sampler),
        ):
            unsupported = [
                model.analysis
                for model in self._models
                if not _is_implemented(getattr(model.backend, method), fallback)
            ]
            if unsupported:
                log.debug(
                    "`%s` is unavailable for the combination: %s do(es) not implement it.",
                    method,
                    ", ".join(unsupported),
                )
                setattr(self, method, fallback.__get__(self, type(self)))

    def _lift_constraints(self) -> List[Any]:
        r"""
        Lift the optimiser constraints of each model into the global parameter space.

        A constraint of model :math:`m` is a function of its local parameters; the
        lifted version evaluates it on :math:`\mathbf{p}[I_m]` and scatters the
        columns of its Jacobian back into the global vector.

        Raises:
            ``~spey.system.exceptions.InvalidInput``: If a model declares a constraint
              of a type this combiner cannot remap.

        Returns:
            ``List[Any]``:
            Constraints expressed in the global parameter vector.
        """
        # Imported here: scipy.optimize is only needed when a backend declares
        # constraints, and this keeps the module import light.
        from scipy.optimize import (  # pylint: disable=import-outside-toplevel
            NonlinearConstraint,
        )

        lifted: List[Any] = []
        for model, index_map in zip(self._models, self._maps):
            for constraint in getattr(model.backend, "constraints", []):
                if not isinstance(constraint, NonlinearConstraint):
                    raise InvalidInput(
                        f"'{model.analysis}' declares a constraint of type "
                        f"{type(constraint)} which can not be mapped onto the combined "
                        "parameter vector. Only scipy NonlinearConstraint is supported."
                    )
                lifted.append(
                    NonlinearConstraint(
                        fun=self._lift_callable(constraint.fun, index_map),
                        lb=constraint.lb,
                        ub=constraint.ub,
                        jac=(
                            self._lift_jacobian(constraint.jac, index_map)
                            if callable(constraint.jac)
                            else constraint.jac
                        ),
                    )
                )
        return lifted

    @staticmethod
    def _lift_callable(
        func: Callable[[np.ndarray], Any], index_map: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Wrap a local callable so that it accepts the global parameter vector.

        Args:
            func (``Callable[[np.ndarray], Any]``): Function of the local parameters.
            index_map (``np.ndarray``): Gather map of the owning model.

        Returns:
            ``Callable[[np.ndarray], np.ndarray]``:
            Function of the global parameters.
        """

        def lifted(pars: np.ndarray) -> np.ndarray:
            """Evaluate the local function on the gathered parameters."""
            return _to_numpy(func(np.asarray(pars, dtype=np.float64)[index_map]))

        return lifted

    def _lift_jacobian(
        self, jac: Callable[[np.ndarray], Any], index_map: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Wrap a local constraint Jacobian so that it returns global-space columns.

        Args:
            jac (``Callable[[np.ndarray], Any]``): Jacobian of the local constraint,
              with shape ``(n_constraints, n_local_parameters)``.
            index_map (``np.ndarray``): Gather map of the owning model.

        Returns:
            ``Callable[[np.ndarray], np.ndarray]``:
            Jacobian with shape ``(n_constraints, n_global_parameters)``.
        """
        npar = self._npar

        def lifted(pars: np.ndarray) -> np.ndarray:
            """Scatter the local Jacobian columns into the global parameter space."""
            local = _to_numpy(jac(np.asarray(pars, dtype=np.float64)[index_map]))
            local = np.atleast_2d(local)
            out = np.zeros((local.shape[0], npar), dtype=np.float64)
            np.add.at(out, (slice(None), index_map), local)
            return out

        return lifted

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(analyses={self.analyses}, "
            f"parameters={self._config.parameter_names})"
        )

    def __len__(self) -> int:
        """Number of constituent statistical models."""
        return len(self._models)

    def __iter__(self) -> Iterator[StatisticalModel]:
        """Iterate over the constituent statistical models."""
        yield from self._models

    def __getitem__(self, item: ModelKey) -> StatisticalModel:
        """
        Retrieve a constituent model by position or analysis name.

        Args:
            item (``Union[str, int]``): Analysis identifier or positional index.

        Raises:
            ``~spey.system.exceptions.AnalysisQueryError``: If the key matches no model.

        Returns:
            ~spey.StatisticalModel:
            The requested model.
        """
        return self._models[self._resolve_model_key(item)]

    @property
    def statistical_models(self) -> Tuple[StatisticalModel, ...]:
        """
        Constituent statistical models in the order they were supplied.

        Returns:
            ``Tuple[~spey.StatisticalModel, ...]``
        """
        return self._models

    @property
    def analyses(self) -> List[str]:
        """
        Analysis identifiers of the constituent models.

        Returns:
            ``List[str]``
        """
        return [model.analysis for model in self._models]

    @property
    def npar(self) -> int:
        """
        Number of parameters of the combined model.

        Returns:
            ``int``
        """
        return self._npar

    @property
    def parameter_names(self) -> List[str]:
        """
        Names of the combined (global) parameters.

        Returns:
            ``List[str]``
        """
        return list(self._config.parameter_names)

    @property
    def index_map(self) -> Dict[str, np.ndarray]:
        r"""
        Gather map :math:`I_m` of every constituent model, keyed by analysis name.

        ``index_map["SR_A"][j]`` is the global index of the :math:`j`-th local
        parameter of ``SR_A``, i.e. ``local_pars = global_pars[index_map[analysis]]``.

        Returns:
            ``Dict[str, np.ndarray]``:
            Analysis identifier to index array.
        """
        return dict(zip(self.analyses, (idx.copy() for idx in self._maps)))

    @property
    def shared_parameter_names(self) -> List[str]:
        """
        Names of the global parameters that more than one model contributes to.

        Returns:
            ``List[str]``
        """
        counts = np.bincount(self._flat_index, minlength=self._npar)
        return [
            name
            for slot, name in enumerate(self._config.parameter_names)
            if counts[slot] > 1
        ]

    @property
    def is_differentiable(self) -> bool:
        """
        Whether every constituent model can supply a gradient.

        Returns:
            ``bool``
        """
        return self._gradient_available

    @property
    def is_hessian_available(self) -> bool:
        """
        Whether every constituent model can supply a Hessian.

        Returns:
            ``bool``
        """
        return self._hessian_available

    @property
    def is_alive(self) -> bool:
        """
        Whether at least one constituent model carries a non-zero signal yield.

        Returns:
            ``bool``
        """
        return any(model.is_alive for model in self._models)

    def config(
        self, allow_negative_signal: bool = True, poi_upper_bound: float = 10.0
    ) -> ModelConfig:
        r"""
        Model configuration of the combined parameter vector.

        Args:
            allow_negative_signal (``bool``, default ``True``): If ``True`` the lower
              bound of the POI is :attr:`~spey.base.model_config.ModelConfig.minimum_poi`,
              otherwise it is ``0.0``.
            poi_upper_bound (``float``, default ``10.0``): Upper bound of the POI
              :math:`\mu` during optimisation.

        Returns:
            ~spey.base.model_config.ModelConfig:
            Combined configuration holding the global POI index, initial values,
            bounds, and parameter names.
        """
        if allow_negative_signal and poi_upper_bound == 10.0:
            return self._config

        return ModelConfig(
            poi_index=self._config.poi_index,
            minimum_poi=self._config.minimum_poi,
            suggested_init=self._config.suggested_init,
            suggested_bounds=self._config.rescale_poi_bounds(
                allow_negative_signal=allow_negative_signal,
                poi_upper_bound=poi_upper_bound,
            ),
            parameter_names=self._config.parameter_names,
            suggested_fixed=self._config.suggested_fixed,
        )

    # ------------------------------------------------------------------
    # Data bookkeeping
    # ------------------------------------------------------------------
    def _widths(self) -> List[int]:
        """
        Number of data points each constituent model consumes.

        Computed once, by asking every backend for its expected data at the suggested
        initialisation, and reused to split a combined dataset back into per-model
        chunks.

        Raises:
            ``NotImplementedError``: If a constituent model does not implement
              :func:`~spey.BackendBase.expected_data`.

        Returns:
            ``List[int]``:
            Data length per model, in model order.
        """
        if self._data_widths is None:
            init = np.array(self._config.suggested_init, dtype=np.float64)
            self._data_widths = [
                int(np.atleast_1d(_to_numpy(model.backend.expected_data(init[idx]))).size)
                for model, idx in zip(self._models, self._maps)
            ]
            log.debug("Data widths per model: %s", self._data_widths)
        return self._data_widths

    def _split_data(self, data: Optional[CombinedData]) -> List[Optional[Any]]:
        """
        Split a combined dataset into per-model chunks.

        Args:
            data (``Union[List[float], np.ndarray, Dict[Union[str, int], Any]]``, default ``None``):
              Combined dataset.

              * ``None``: every model is handed ``None`` and falls back to its own data.
              * ``dict``: maps an analysis name or model position to that model's
                data, which is handed to the backend untouched; models that are not
                listed fall back to their own data.
              * array-like: laid out in model order exactly as :func:`expected_data`
                produces it.

        Raises:
            ``~spey.system.exceptions.InvalidInput``: If an array-like ``data`` does not
              have the length that the constituent models expect.
            ``~spey.system.exceptions.AnalysisQueryError``: If a key of a ``dict``
              ``data`` matches no model.

        Returns:
            ``List[Optional[Any]]``:
            One chunk per model.
        """
        if data is None:
            return [None] * len(self._models)

        if isinstance(data, dict):
            chunks: List[Optional[Any]] = [None] * len(self._models)
            for key, value in data.items():
                chunks[self._resolve_model_key(key)] = value
            return chunks

        data = np.asarray(data, dtype=np.float64).ravel()
        widths = self._widths()
        if data.size != sum(widths):
            raise InvalidInput(
                f"Expected a combined dataset of length {sum(widths)} "
                f"({' + '.join(map(str, widths))}), got {data.size}."
            )
        edges = np.cumsum([0] + widths)
        return [data[edges[i] : edges[i + 1]] for i in range(len(widths))]

    def split_data(self, data: Union[List[float], np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Split a flat combined dataset into per-analysis chunks.

        This is the inverse of the concatenation performed by :func:`expected_data`
        and :func:`get_sampler`, e.g. to inspect the Asimov data of every constituent
        model.

        Args:
            data (``Union[List[float], np.ndarray]``): Combined dataset laid out in
              model order.

        Raises:
            ``~spey.system.exceptions.InvalidInput``: If ``data`` has the wrong length.
            ``NotImplementedError``: If a constituent model does not implement
              :func:`~spey.BackendBase.expected_data`.

        Returns:
            ``Dict[str, np.ndarray]``:
            Analysis identifier to that model's data.
        """
        return dict(zip(self.analyses, self._split_data(np.asarray(data))))

    # ------------------------------------------------------------------
    # Likelihood interface
    # ------------------------------------------------------------------
    def get_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[CombinedData] = None,
    ) -> Callable[[np.ndarray], float]:
        r"""
        Return a callable that evaluates the combined
        :math:`\ln\mathcal{L}(\mathbf{p})`.

        The combined log-pdf is the sum of the constituent log-pdfs evaluated on the
        gathered parameter vectors,

        .. math::

            \ln\mathcal{L}(\mathbf{p}) = \sum_m \ln\mathcal{L}_m(\mathbf{p}[I_m]).

        The returned callable is registered as an :func:`autograd.extend.primitive`
        with vector-Jacobian products built from the combined gradient and Hessian, so
        that ``spey`` components which differentiate the log-pdf externally — notably
        :func:`~spey.multiparameter.find_contour` — keep working regardless of which
        autodiff framework the constituent backends use internally.  The primitive
        also guarantees that a backend only ever sees a concrete ``float64`` array.

        Args:
            expected (~spey.ExpectationType): Which dataset each constituent model uses
              when ``data`` is ``None``.

              * :obj:`~spey.ExpectationType.observed`: Observed data (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Observed data with post-fit
                nuisance treatment.
              * :obj:`~spey.ExpectationType.apriori`: Background-only (SM) expectation.

            data (``Union[List[float], np.ndarray, Dict[Union[str, int], Any]]``, default ``None``):
              Combined dataset laid out in model order, as produced by
              :func:`expected_data` or :func:`get_sampler`, or a mapping from analysis
              name (or position) to the data of that model.  When given it overrides
              ``expected`` for the models it covers.

        Raises:
            ``~spey.system.exceptions.InvalidInput``: If ``data`` has the wrong length.
            ``~spey.system.exceptions.AnalysisQueryError``: If a key of a ``dict``
              ``data`` matches no model.

        Returns:
            ``Callable[[np.ndarray], float]``:
            Function ``logpdf(pars) -> float`` of the global parameter vector.
        """
        chunks = self._split_data(data)
        funcs = [
            model.backend.get_logpdf_func(expected=expected, data=chunk)
            for model, chunk in zip(self._models, chunks)
        ]
        maps = self._maps

        @primitive
        def logpdf(pars: np.ndarray) -> float:
            """Combined log-probability at the global parameter vector."""
            pars = np.asarray(pars, dtype=np.float64)
            return sum(_to_float(func(pars[idx])) for func, idx in zip(funcs, maps))

        if self._gradient_available:
            # Built on first differentiation only: `get_logpdf_func` is called on
            # every likelihood evaluation, whereas the derivative closures are needed
            # only when something differentiates through the returned callable.
            derivatives: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}

            def _derivative(order: str) -> Callable[[np.ndarray], np.ndarray]:
                """Lazily build (and memoise) the gradient or Hessian closure."""
                if order not in derivatives:
                    derivatives[order] = (
                        self._gradient_logpdf_func(expected=expected, data=data)
                        if order == "grad"
                        else self.get_hessian_logpdf_func(expected=expected, data=data)
                    )
                return derivatives[order]

            @primitive
            def grad_logpdf(pars: np.ndarray) -> np.ndarray:
                """Gradient of the combined log-probability."""
                return _derivative("grad")(pars)

            defvjp(logpdf, lambda ans, pars: lambda g: g * grad_logpdf(pars))

            if self._hessian_available:
                # The Hessian is symmetric, but transpose explicitly so the VJP stays
                # correct for backends returning a slightly asymmetric numerical matrix.
                defvjp(
                    grad_logpdf,
                    lambda ans, pars: lambda g: _derivative("hess")(pars).T.dot(g),
                )

        return logpdf

    def _gradient_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[CombinedData] = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        r"""
        Return a callable evaluating :math:`\nabla_{\mathbf{p}}\ln\mathcal{L}`.

        Each backend supplies the gradient of its *objective*, :math:`-\ln\mathcal{L}_m`,
        so the sign is flipped before the scatter-add.

        Args:
            expected (~spey.ExpectationType): Dataset prescription, see
              :func:`get_logpdf_func`.
            data (``Union[List[float], np.ndarray, Dict[Union[str, int], Any]]``, default ``None``):
              Combined dataset, see :func:`get_logpdf_func`.

        Raises:
            ``NotImplementedError``: If any constituent model lacks a gradient.

        Returns:
            ``Callable[[np.ndarray], np.ndarray]``:
            Function returning the length-``npar`` gradient of the log-pdf.
        """
        objective = self.get_objective_function(
            expected=expected, data=data, do_grad=True
        )

        def gradient(pars: np.ndarray) -> np.ndarray:
            """Gradient of the combined log-probability (minus the objective's)."""
            return -objective(pars)[1]

        return gradient

    def get_objective_function(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[CombinedData] = None,
        do_grad: bool = True,
    ) -> Callable[[np.ndarray], Union[float, Tuple[float, np.ndarray]]]:
        r"""
        Return the combined objective :math:`-\ln\mathcal{L}(\mathbf{p})` and,
        optionally, its gradient.

        The gradient is assembled from the constituent gradients by the scatter-add

        .. math::

            \frac{\partial(-\ln\mathcal{L})}{\partial p_a}
            = \sum_m \sum_j \frac{\partial(-\ln\mathcal{L}_m)}{\partial\theta_{m,j}}
              \,\delta_{I_m[j],\,a}\ ,

        implemented as a single :func:`numpy.bincount` over the concatenated index
        maps, which handles shared slots — where several models contribute to the same
        global derivative — in one pass.

        The constituent objective closures are built **once**, when this method is
        called, so that repeated optimiser evaluations only pay for the backend
        evaluations themselves plus the gather/scatter.

        Args:
            expected (~spey.ExpectationType): Dataset prescription, see
              :func:`get_logpdf_func`.
            data (``Union[List[float], np.ndarray, Dict[Union[str, int], Any]]``, default ``None``):
              Combined dataset, see :func:`get_logpdf_func`.
            do_grad (``bool``, default ``True``): If ``True`` the callable returns
              ``(objective, gradient)``, otherwise only the scalar objective.

        Raises:
            ``NotImplementedError``: If ``do_grad=True`` and at least one constituent
              model does not implement a gradient.
            ``~spey.system.exceptions.InvalidInput``: If ``data`` has the wrong length.

        Returns:
            ``Callable[[np.ndarray], Union[float, Tuple[float, np.ndarray]]]``:
            The combined objective function.
        """
        if do_grad and not self._gradient_available:
            raise NotImplementedError(
                "Gradient is not available: "
                + ", ".join(
                    model.analysis for model in self._models if not _has_gradient(model)
                )
                + " does not implement `get_objective_function` with `do_grad=True`."
            )

        chunks = self._split_data(data)
        funcs = [
            model.backend.get_objective_function(
                expected=expected, data=chunk, do_grad=do_grad
            )
            for model, chunk in zip(self._models, chunks)
        ]
        maps = self._maps

        if not do_grad:

            def objective(pars: np.ndarray) -> float:
                """Combined negative log-likelihood."""
                pars = np.asarray(pars, dtype=np.float64)
                return sum(_to_float(func(pars[idx])) for func, idx in zip(funcs, maps))

            return objective

        flat_index = self._flat_index
        npar = self._npar
        sizes = [idx.size for idx in maps]

        def objective_and_grad(pars: np.ndarray) -> Tuple[float, np.ndarray]:
            """Combined negative log-likelihood and its gradient."""
            pars = np.asarray(pars, dtype=np.float64)
            value = 0.0
            local_grads = np.empty(flat_index.size, dtype=np.float64)
            position = 0
            for func, idx, size in zip(funcs, maps, sizes):
                current, grad = func(pars[idx])
                value += _to_float(current)
                local_grads[position : position + size] = _to_numpy(grad).ravel()
                position += size
            return value, np.bincount(flat_index, weights=local_grads, minlength=npar)

        return objective_and_grad

    def get_hessian_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[CombinedData] = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        r"""
        Return a callable evaluating the Hessian of the combined
        :math:`\ln\mathcal{L}(\mathbf{p})`.

        Each constituent Hessian is an :math:`n_m \times n_m` block that is added into
        the sub-block of the global matrix selected by the model's gather map,

        .. math::

            H_{ab} = \sum_m \sum_{j,k} H^{(m)}_{jk}\,
                     \delta_{I_m[j],\,a}\,\delta_{I_m[k],\,b}\ ,

        so shared parameters accumulate contributions from every model that owns them,
        including the off-diagonal terms that correlate a shared parameter with each
        model's private parameters.

        Args:
            expected (~spey.ExpectationType): Dataset prescription, see
              :func:`get_logpdf_func`.
            data (``Union[List[float], np.ndarray, Dict[Union[str, int], Any]]``, default ``None``):
              Combined dataset, see :func:`get_logpdf_func`.

        Raises:
            ``NotImplementedError``: If at least one constituent model does not
              implement :func:`~spey.BackendBase.get_hessian_logpdf_func`.
            ``~spey.system.exceptions.InvalidInput``: If ``data`` has the wrong length.

        Returns:
            ``Callable[[np.ndarray], np.ndarray]``:
            Function returning the ``(npar, npar)`` Hessian of the log-pdf.
        """
        if not self._hessian_available:
            raise NotImplementedError(
                "Hessian is not available: "
                + ", ".join(
                    model.analysis for model in self._models if not _has_hessian(model)
                )
                + " does not implement `get_hessian_logpdf_func`."
            )

        chunks = self._split_data(data)
        funcs = [
            model.backend.get_hessian_logpdf_func(expected=expected, data=chunk)
            for model, chunk in zip(self._models, chunks)
        ]
        maps = self._maps
        blocks = self._block_index
        duplicated = self._duplicated
        npar = self._npar

        def hessian(pars: np.ndarray) -> np.ndarray:
            """Combined Hessian of the log-probability."""
            pars = np.asarray(pars, dtype=np.float64)
            out = np.zeros((npar, npar), dtype=np.float64)
            for func, idx, block, has_duplicate in zip(funcs, maps, blocks, duplicated):
                local = _to_numpy(func(pars[idx])).reshape(idx.size, idx.size)
                if has_duplicate:
                    np.add.at(out, block, local)
                else:
                    out[block] += local
            return out

        return hessian

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    def expected_data(self, pars: List[float], **kwargs) -> np.ndarray:
        r"""
        Return the combined expected data at the given global parameter vector.

        Each model is asked for its own expected data at its gathered parameters and
        the results are concatenated in model order.  This layout is the one
        :func:`get_logpdf_func` expects for its ``data`` argument, so Asimov data
        generated here round-trips correctly.

        Args:
            pars (``List[float]``): Global parameter vector :math:`\mathbf{p}`.
            kwargs: Forwarded to each constituent
              :func:`~spey.BackendBase.expected_data`.

        Raises:
            ``NotImplementedError``: If a constituent model does not implement
              :func:`~spey.BackendBase.expected_data`.

        Returns:
            ``np.ndarray``:
            Concatenated expected data of every constituent model.
        """
        pars = np.asarray(pars, dtype=np.float64)
        return np.concatenate(
            [
                np.atleast_1d(_to_numpy(model.backend.expected_data(pars[idx], **kwargs)))
                for model, idx in zip(self._models, self._maps)
            ]
        )

    def get_sampler(self, pars: np.ndarray) -> Callable[[int], np.ndarray]:
        r"""
        Return a callable drawing pseudo-data from the combined model.

        Given the parameters, the observations of different analyses are conditionally
        independent — the models are correlated *through* the shared parameters, not
        through their observations — so each model is sampled from its own
        distribution at the gathered parameter values and the draws are concatenated
        bin-wise, in the same order as :func:`expected_data`.

        Args:
            pars (``np.ndarray``): Global parameter vector at which to condition the
              samplers.

        Raises:
            ``NotImplementedError``: If a constituent model does not implement
              :func:`~spey.BackendBase.get_sampler`.

        Returns:
            ``Callable[[int], np.ndarray]``:
            Function ``sampler(n)`` returning an array of shape
            ``(n, total number of bins)``.
        """
        pars = np.asarray(pars, dtype=np.float64)
        samplers = [
            model.backend.get_sampler(pars[idx])
            for model, idx in zip(self._models, self._maps)
        ]

        def sampler(sample_size: int, *args, **kwargs) -> np.ndarray:
            """
            Draw ``sample_size`` pseudo-experiments from every constituent model.

            Args:
                sample_size (``int``): Number of pseudo-experiments.

            Returns:
                ``np.ndarray``:
                Array of shape ``(sample_size, total number of bins)``.
            """
            return np.hstack(
                [
                    _to_numpy(sample(sample_size, *args, **kwargs)).reshape(
                        sample_size, -1
                    )
                    for sample in samplers
                ]
            )

        return sampler
