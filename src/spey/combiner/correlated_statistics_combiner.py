r"""
Correlated Statistical Model Combiner
=====================================

This module provides :class:`CorrelatedStatisticsCombiner`, a ``spey`` **plug-in**
(``default.correlated_combiner``) that joins an arbitrary collection of
:class:`~spey.StatisticalModel` objects which **share** parameters — parameters of
interest, Wilson coefficients, or ordinary nuisance parameters.

Unlike :class:`~spey.combiner.UncorrelatedStatisticsCombiner`, which identifies only
the parameters of interest of the constituent analyses, this combiner builds a single
joint parameter vector in which *any* user-declared parameters are **identified**
across models.  Because the combiner is itself a backend, the result is an ordinary
:class:`~spey.StatisticalModel` and therefore works with the entire ``spey``
toolchain — hypothesis testing, upper limits, Asimov data, and
:func:`~spey.multiparameter.find_contour`.

The joint likelihood, its gradient and its Hessian are assembled by
:class:`~spey.combiner.combiner_core.CombinerBase`; see
:mod:`spey.combiner.combiner_core` for the derivation.  This module only translates
the ``shared_parameters`` declaration into the per-model index maps :math:`I_m`,

.. math::

    \ln\mathcal{L}(\mathbf{p})
    = \sum_{m=1}^{M} \ln\mathcal{L}_m\!\left(\mathbf{p}[I_m]\right),
    \qquad \theta_{m,j} = p_{I_m[j]}\ ,

so that two models share a parameter exactly when their index maps point at the same
global slot.

.. versionadded:: 0.2.8

References
----------
* G. Cowan, K. Cranmer, E. Gross, O. Vitells, *Asymptotic formulae for
  likelihood-based tests of new physics*, Eur. Phys. J. C **71** (2011) 1554,
  :xref:`1007.1727`.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from spey.interface.statistical_model import StatisticalModel
from spey.system.exceptions import InvalidInput

from .combiner_core import CombinerBase, ModelKey

__all__ = ["CorrelatedStatisticsCombiner"]

log = logging.getLogger("Spey")

#: Identifier of a local parameter: its name or its index within the model.
ParameterKey = Union[str, int]

#: One declaration of a parameter shared between models, see
#: :class:`CorrelatedStatisticsCombiner`.
SharedParameterSpec = Union[str, Dict[str, Any], Sequence[Tuple[ModelKey, ParameterKey]]]


def __dir__():
    return __all__


class CorrelatedStatisticsCombiner(CombinerBase):
    r"""
    Combine :class:`~spey.StatisticalModel` objects that **share** parameters
    (``default.correlated_combiner``).

    Each constituent model keeps its own backend, its own data, and its own
    likelihood prescription; the combiner only decides **which local parameters are
    the same parameter**.  The joint log-likelihood is

    .. math::

        \ln\mathcal{L}(\mathbf{p})
        = \sum_{m} \ln\mathcal{L}_m\!\left(\mathbf{p}[I_m]\right),

    where :math:`I_m` is the gather map from the global parameter vector into the
    local parameter vector of model :math:`m` (see :mod:`spey.combiner.combiner_core`
    for the derivation of the gradient and Hessian).

    Shared parameters need **not** be parameters of interest: any nuisance parameter
    — a luminosity scale, a shared jet-energy-scale nuisance, a common Wilson
    coefficient — can be identified across models, and is then profiled jointly.

    Args:
        statistical_models (``Sequence[~spey.StatisticalModel]``): Models to combine.
          Their :attr:`~spey.StatisticalModel.analysis` identifiers must be unique.
        shared_parameters (``Sequence[SharedParameterSpec]``, default ``None``):
          Declaration of which local parameters are identified with each other.  Each
          entry may be

          * a ``str``: every model owning a parameter with this name joins one shared
            slot.  A parameter of interest is always reachable as ``"mu"`` when its
            backend does not declare
            :attr:`~spey.base.model_config.ModelConfig.parameter_names`.
          * a ``dict`` of the form
            ``{"name": <global name>, "members": {<model key>: <parameter key>}}``.
            The ``"name"`` key is optional and defaults to the local name of the
            first member.  A bare ``{<model key>: <parameter key>}`` mapping is also
            accepted.
          * a sequence of ``(<model key>, <parameter key>)`` pairs.

          A *model key* is either the analysis name (``str``) or the position of the
          model in ``statistical_models`` (``int``).  A *parameter key* is either the
          local parameter name (``str``) or its local index (``int``).

          When ``None``, nothing is shared and the combination reduces to an
          uncorrelated product with a block-diagonal parameter vector.
        poi_name (``str``, default ``None``): Global name of the parameter to expose
          as the combined POI.  Defaults to the global slot holding the POI of the
          first model.
        poi_index (``int``, default ``None``): Global index of the combined POI.
          Mutually exclusive with ``poi_name``.

    Raises:
        ``TypeError``: If an entry of ``statistical_models`` is not a
          :class:`~spey.StatisticalModel`.
        ``~spey.system.exceptions.AnalysisQueryError``: If two models share the same
          :attr:`~spey.StatisticalModel.analysis` identifier, or if a model key in
          ``shared_parameters`` does not match any model.
        ``~spey.system.exceptions.InvalidInput``: If ``statistical_models`` is empty,
          if a parameter key cannot be resolved, if a local parameter is declared in
          more than one shared group, if two shared groups carry the same name, if
          both ``poi_name`` and ``poi_index`` are supplied, or if the bounds of a
          shared parameter do not overlap.

    Example:

    .. code-block:: python3

        >>> import numpy as np
        >>> import spey

        >>> normal = spey.get_backend("default.normal")
        >>> model_a = normal(
        ...     signal_yields=lambda c: c[0] * np.array([12.0, 15.0]),
        ...     background_yields=[50.0, 48.0],
        ...     data=[63.0, 55.0],
        ...     absolute_uncertainties=[7.0, 6.0],
        ...     n_signal_parameters=1,
        ...     analysis="SR_A",
        ... )
        >>> model_b = normal(
        ...     signal_yields=lambda c: c[0] ** 2 * np.array([4.0, 3.0]),
        ...     background_yields=[30.0, 22.0],
        ...     data=[35.0, 20.0],
        ...     absolute_uncertainties=[5.0, 4.5],
        ...     n_signal_parameters=1,
        ...     analysis="SR_B",
        ... )

        >>> combiner = spey.get_backend("default.correlated_combiner")
        >>> combined = combiner(
        ...     statistical_models=[model_a, model_b],
        ...     # mu and the Wilson coefficient are common to both analyses
        ...     shared_parameters=["mu", "signal_par_0"],
        ...     analysis="SR_A+SR_B",
        ... )
        >>> combined.backend.parameter_names
        ['mu', 'signal_par_0']
        >>> combined.maximize_likelihood()

    .. seealso::

        :class:`~spey.combiner.UncorrelatedStatisticsCombiner` for combining models
        that share **only** their parameter of interest, and
        :func:`~spey.multiparameter.find_contour` for mapping confidence contours of
        the combined model.
    """

    name: str = "default.correlated_combiner"
    """Name of the backend"""

    __slots__ = ["_shared_spec", "_poi_request"]

    def __init__(
        self,
        statistical_models: Sequence[StatisticalModel],
        shared_parameters: Optional[Sequence[SharedParameterSpec]] = None,
        poi_name: Optional[str] = None,
        poi_index: Optional[int] = None,
    ):
        self._shared_spec = shared_parameters
        self._poi_request: Tuple[Optional[str], Optional[int]] = (poi_name, poi_index)
        super().__init__(statistical_models)

    # ------------------------------------------------------------------
    # CombinerBase hooks
    # ------------------------------------------------------------------
    def _build_parameter_map(
        self, local_names: List[List[str]]
    ) -> Tuple[List[np.ndarray], Dict[int, str]]:
        """
        Turn the ``shared_parameters`` declaration into per-model gather maps.

        Args:
            local_names (``List[List[str]]``): Local parameter names per model.

        Raises:
            ``~spey.system.exceptions.InvalidInput``: If a declaration is malformed.
            ``~spey.system.exceptions.AnalysisQueryError``: If a model key matches no
              model.

        Returns:
            ``Tuple[List[np.ndarray], Dict[int, str]]``:
            Gather map of every model and the name of every shared slot.
        """
        groups = self._resolve_groups(self._shared_spec, local_names)
        group_of = {
            (pos, par): gid
            for gid, group in enumerate(groups)
            for pos, par in group["members"].items()
        }
        maps, slot_of_group = self._allocate_slots(local_names, group_of)
        return maps, {
            slot_of_group[gid]: group["name"] for gid, group in enumerate(groups)
        }

    def _select_poi(self, names: List[str]) -> int:
        """
        Determine the index of the combined parameter of interest.

        Args:
            names (``List[str]``): Global parameter names.

        Raises:
            ``~spey.system.exceptions.InvalidInput``: If both ``poi_name`` and
              ``poi_index`` were given, or if the requested parameter does not exist.

        Returns:
            ``int``:
            Global index of the parameter of interest.
        """
        poi_name, poi_index = self._poi_request
        if poi_name is not None and poi_index is not None:
            raise InvalidInput("Please provide either `poi_name` or `poi_index`.")
        if poi_name is not None:
            if poi_name not in names:
                raise InvalidInput(
                    f"'{poi_name}' is not among the combined parameters: {names}."
                )
            return names.index(poi_name)
        if poi_index is not None:
            if not 0 <= int(poi_index) < self._npar:
                raise InvalidInput(
                    f"POI index {poi_index} is out of range for {self._npar} parameters."
                )
            return int(poi_index)
        return super()._select_poi(names)

    # ------------------------------------------------------------------
    # Declaration parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_parameter_key(key: ParameterKey, names: List[str], analysis: str) -> int:
        """
        Resolve a local parameter name or index into a local index.

        Args:
            key (``Union[str, int]``): Local parameter name or index.
            names (``List[str]``): Local parameter names of the model.
            analysis (``str``): Analysis identifier, used in the error message.

        Raises:
            ``~spey.system.exceptions.InvalidInput``: If the key cannot be resolved.

        Returns:
            ``int``:
            Index of the parameter within the model's local parameter vector.
        """
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            if not 0 <= int(key) < len(names):
                raise InvalidInput(
                    f"Parameter index {key} is out of range for '{analysis}' "
                    f"which has {len(names)} parameters."
                )
            return int(key)
        if key not in names:
            raise InvalidInput(
                f"Parameter '{key}' is not among the parameters of '{analysis}': {names}."
            )
        return names.index(key)

    def _resolve_member(
        self, member: Tuple[ModelKey, ParameterKey], local_names: List[List[str]]
    ) -> Tuple[int, int]:
        """
        Resolve a ``(model key, parameter key)`` pair into positions.

        Args:
            member (``Tuple[Union[str, int], Union[str, int]]``): Model and parameter key.
            local_names (``List[List[str]]``): Local parameter names per model.

        Raises:
            ``~spey.system.exceptions.AnalysisQueryError``: If the model key matches no
              model.
            ``~spey.system.exceptions.InvalidInput``: If the parameter key cannot be
              resolved.

        Returns:
            ``Tuple[int, int]``:
            Model position and local parameter index.
        """
        pos = self._resolve_model_key(member[0])
        par = self._resolve_parameter_key(
            member[1], local_names[pos], self._models[pos].analysis
        )
        return pos, par

    def _resolve_groups(
        self,
        shared_parameters: Optional[Sequence[SharedParameterSpec]],
        local_names: List[List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Normalise the ``shared_parameters`` declarations into resolved groups.

        Args:
            shared_parameters (``Optional[Sequence[SharedParameterSpec]]``): User
              declarations, see :class:`CorrelatedStatisticsCombiner`.
            local_names (``List[List[str]]``): Local parameter names per model.

        Raises:
            ``~spey.system.exceptions.InvalidInput``: If a declaration is malformed,
              if a local parameter belongs to more than one group, if a group ends up
              empty, or if two groups carry the same name.

        Returns:
            ``List[Dict[str, Any]]``:
            One entry per group with keys ``"name"`` (``str``) and ``"members"``
            (``Dict[int, int]`` mapping model position to local parameter index).
        """
        if shared_parameters is None:
            return []
        if isinstance(shared_parameters, (str, dict)):
            shared_parameters = [shared_parameters]

        groups: List[Dict[str, Any]] = []
        owner: Dict[Tuple[int, int], str] = {}

        for spec in shared_parameters:
            name, raw_members = self._parse_spec(spec, local_names)
            members: Dict[int, int] = {}
            for member in raw_members:
                pos, par = self._resolve_member(member, local_names)
                if (pos, par) in owner:
                    raise InvalidInput(
                        f"Parameter '{local_names[pos][par]}' of "
                        f"'{self._models[pos].analysis}' is declared in more than one "
                        f"shared group ('{owner[(pos, par)]}' and '{name}')."
                    )
                owner[(pos, par)] = name
                members[pos] = par
            if not members:
                raise InvalidInput(f"Shared parameter group '{name}' has no members.")
            if len(members) == 1:
                log.warning(
                    "Shared parameter group '%s' has a single member; it is not "
                    "correlating anything.",
                    name,
                )
            if any(group["name"] == name for group in groups):
                raise InvalidInput(f"Shared parameter group '{name}' is declared twice.")
            groups.append({"name": name, "members": members})

        return groups

    def _parse_spec(
        self, spec: SharedParameterSpec, local_names: List[List[str]]
    ) -> Tuple[str, List[Tuple[ModelKey, ParameterKey]]]:
        """
        Split a single declaration into its global name and its ``(model, parameter)`` pairs.

        Args:
            spec (``SharedParameterSpec``): One declaration.
            local_names (``List[List[str]]``): Local parameter names per model.

        Raises:
            ``~spey.system.exceptions.InvalidInput``: If ``spec`` has an unsupported
              type, or if a plain-``str`` declaration matches no model.

        Returns:
            ``Tuple[str, List[Tuple[ModelKey, ParameterKey]]]``:
            Global parameter name and the list of members.
        """
        if isinstance(spec, str):
            members = [
                (pos, spec) for pos, names in enumerate(local_names) if spec in names
            ]
            if not members:
                raise InvalidInput(
                    f"No model declares a parameter named '{spec}'. Available "
                    f"parameter names: {local_names}."
                )
            return spec, members

        if isinstance(spec, dict):
            if "members" in spec:
                name = spec.get("name")
                raw = spec["members"]
            elif "name" in spec:
                raise InvalidInput(
                    "A shared parameter declaration carrying a 'name' key must also "
                    "carry a 'members' key, e.g. "
                    "{'name': 'c1', 'members': {'SR_A': 1, 'SR_B': 'signal_par_0'}}."
                )
            else:
                name = None
                raw = spec
            members = list(raw.items()) if isinstance(raw, dict) else list(raw)
        elif isinstance(spec, (list, tuple)):
            name = None
            members = [tuple(item) for item in spec]
        else:
            raise InvalidInput(
                f"Can not interpret shared parameter declaration of type {type(spec)}."
            )

        if name is None:
            pos, par = self._resolve_member(members[0], local_names)
            name = local_names[pos][par]
        return str(name), members
