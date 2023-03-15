"""Configuration class for Statistical Models"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import copy


@dataclass
class ModelConfig:
    """
    Configuration class for statistical model data. This class contains information
    regarding how the fit should evolve.

    :param poi_index (`int`): index of the parameter of interest within parameter list
    :param minimum_poi (`float`): minimum value that parameter of interest can take
    :param suggested_init (`List[float]`): suggested initialisation for parameters
    :param suggested_bounds (`List[Tuple[float, float]]`): suggested boundaries for parameters
    """

    poi_index: int
    minimum_poi: float
    suggested_init: List[float]
    suggested_bounds: List[Tuple[float, float]]

    def fixed_poi_bounds(self, poi_value: Optional[float] = None) -> List[Tuple[float, float]]:
        """
        Adjust bounds with respect to fixed POI value

        :param poi_value (`Optional[float]`, default `None`): desired POI value
        :return `List[Tuple[float, float]]`: updated bounds
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
        Rescale bounds for POI

        :param allow_negative_signal (`bool`, default `True`): if true, POI can take negative values.
        :param poi_upper_bound (`Optional[float]`, default `None`): sets new upper bound for POI.
        :return `List[Tuple[float, float]]`: rescaled bounds
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
