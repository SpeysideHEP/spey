"""Configuration class for Statistical Models"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import copy


@dataclass
class ModelConfig:
    """
    Container to hold certain properties of the backend and statistical model.
    This will ensure the consistency of the computation through out the package.

    Args:
        poi_index (:obj:`int`): index of the parameter of interest within the parameter list.
        minimum_poi (:obj:`float`): minimum value parameter of interest can take,
          see :attr:`~spey.DataBase.minimum_poi`.
        suggested_init (:obj:`List[float]`): suggested initial parameters for the optimiser.
        suggested_bounds (:obj:`List[Tuple[float, float]]`): suggested parameter bounds for the
          optimiser.

    Returns:
        :obj:~spey.base.model_config.ModelConfig:
        Model configuration container for optimiser.
    """

    poi_index: int
    minimum_poi: float
    suggested_init: List[float]
    suggested_bounds: List[Tuple[float, float]]

    def fixed_poi_bounds(self, poi_value: Optional[float] = None) -> List[Tuple[float, float]]:
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
        if not bounds[self.poi_index][0] < poi_value < bounds[self.poi_index][1]:
            bounds[self.poi_index] = (self.minimum_poi if poi_value < 0.0 else 0.0, poi_value + 1)
        return bounds

    def rescale_poi_bounds(
        self, allow_negative_signal: bool = True, poi_upper_bound: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """
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
