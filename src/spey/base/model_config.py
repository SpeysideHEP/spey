"""Configuration class for Statistical Models"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class ModelConfig:
    r"""
    Container to hold certain properties of the backend and statistical model.
    This will ensure the consistency of the computation through out the package.

    Args:
        poi_index (:obj:`int`): index of the parameter of interest within the parameter list.
        minimum_poi (:obj:`float`): minimum value parameter of interest can take to ensure
          :math:`N^{\rm bkg}+\mu N^{\rm sig}\geq0`. This value can be set to :math:`-\infty`
          but note that optimiser will take it as a lower bound so such low value might effect
          the convergence of the optimisation algorithm especially for relatively flat objective
          surfaceses. Hence we suggest setting the minimum value to

            .. math::

                \mu_{\rm min} = - \min\left( \frac{N^{\rm bkg}_i}{N^{\rm sig}_i} \right)\ ,\ i\in {\rm bins}

        suggested_init (:obj:`List[float]`): suggested initial parameters for the optimiser.
        suggested_bounds (:obj:`List[Tuple[float, float]]`): suggested parameter bounds for the
          optimiser.

    Returns:
        :obj:~spey.base.model_config.ModelConfig:
        Model configuration container for optimiser.
    """

    poi_index: int
    """Index of the parameter of interest wihin the parameter list"""
    minimum_poi: float
    r"""
    minimum value parameter of interest can take to ensure :math:`N^{\rm bkg}+\mu N^{\rm sig}\geq0`.
    This value can be set to :math:`-\infty` but note that optimiser will take it as a lower bound so
    such low value might effect the convergence of the optimisation algorithm especially for relatively
    flat objective surfaceses. Hence we suggest setting the minimum value to

    .. math::

        \mu_{\rm min} = - \min\left( \frac{N^{\rm bkg}_i}{N^{\rm sig}_i} \right)\ ,\ i\in {\rm bins}
    """
    suggested_init: List[float]
    """Suggested initialisation for parameters"""
    suggested_bounds: List[Tuple[float, float]]
    """Suggested upper and lower bounds for parameters"""
    parameter_names: Optional[List[str]] = None
    """Names of the parameters"""
    suggested_fixed: Optional[List[bool]] = None
    """Suggested fixed values"""

    @property
    def npar(self) -> int:
        """Number of parameters"""
        return len(self.suggested_init)

    def fixed_poi_bounds(
        self, poi_value: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        r"""
        Adjust the bounds for the parameter of interest for fixed POI fit.

        Args:
            poi_value (:obj:`Optional[float]`, default :obj:`None`): parameter of interest,
              :math:`\mu`.

        Returns:
            :obj:`List[Tuple[float, float]]`:
            Updated bounds.
        """
        if poi_value is None:
            return self.suggested_bounds
        bounds = copy.deepcopy(self.suggested_bounds)
        if any(b is None for b in bounds[self.poi_index]):
            return bounds
        if not bounds[self.poi_index][0] < poi_value < bounds[self.poi_index][1]:
            bounds[self.poi_index] = (
                self.minimum_poi if poi_value < 0.0 else 0.0,
                poi_value + 1,
            )
        return bounds

    def resolve_poi_indices(
        self,
        poi_test: Union[float, Dict[Union[int, str], float]],
    ) -> Dict[int, float]:
        r"""
        Resolve ``poi_test`` to a mapping of ``{parameter_index: fixed_value}``.

        Args:
            poi_test (:obj:`Union[float, Dict[Union[int, str], float]]`): parameter of interest
              value, or a dictionary mapping POI indices (``int``) or names (``str``) to their
              fixed values.

        Raises:
            :obj:`ValueError`: If a string key cannot be found in :attr:`parameter_names`.

        Returns:
            :obj:`Dict[int, float]`:
            Mapping from parameter index to the value it should be fixed at.
        """
        if not isinstance(poi_test, dict):
            return {self.poi_index: float(poi_test)}
        resolved: Dict[int, float] = {}
        for key, val in poi_test.items():
            if isinstance(key, str):
                if self.parameter_names is None:
                    raise ValueError(
                        "Cannot resolve POI name: parameter_names not set in ModelConfig."
                    )
                if key not in self.parameter_names:
                    raise ValueError(
                        f"POI name '{key}' not found in parameter_names: "
                        f"{self.parameter_names}"
                    )
                resolved[self.parameter_names.index(key)] = float(val)
            else:
                resolved[int(key)] = float(val)
        return resolved

    def fixed_poi_bounds_multi(
        self, poi_values: Optional[Dict[int, float]] = None
    ) -> List[Tuple[float, float]]:
        r"""
        Adjust bounds for multiple fixed parameters.

        Args:
            poi_values (:obj:`Optional[Dict[int, float]]`, default :obj:`None`): mapping of
              parameter index to fixed value.  If ``None`` the suggested bounds are returned
              unchanged.

        Returns:
            :obj:`List[Tuple[float, float]]`:
            Updated bounds.
        """
        if not poi_values:
            return self.suggested_bounds
        bounds = copy.deepcopy(self.suggested_bounds)
        for idx, val in poi_values.items():
            if any(b is None for b in bounds[idx]):
                continue
            if not bounds[idx][0] < val < bounds[idx][1]:
                bounds[idx] = (
                    self.minimum_poi if val < 0.0 else 0.0,
                    val + 1,
                )
        return bounds

    def rescale_poi_bounds(
        self, allow_negative_signal: bool = True, poi_upper_bound: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        r"""
        Rescale bounds for POI.

        Args:
            allow_negative_signal (:obj:`bool`, default :obj:`True`): If :obj:`True` :math:`\hat\mu`
              value will be allowed to be negative.
            poi_upper_bound (:obj:`float`, default :obj:`None`): Maximum value POI can take during
              optimisation.

        Returns:
            :obj:`List[Tuple[float, float]]`:
            Updated bounds.
        """
        bounds = copy.deepcopy(self.suggested_bounds)
        if poi_upper_bound:
            bounds[self.poi_index] = (
                self.minimum_poi if allow_negative_signal else 0.0,
                poi_upper_bound,
            )
        else:
            bounds[self.poi_index] = (
                self.minimum_poi if allow_negative_signal else 0.0,
                bounds[self.poi_index],
            )
        return bounds
