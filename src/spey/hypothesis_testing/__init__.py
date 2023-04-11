from typing import Callable, Text, Tuple, List

from spey.system.exceptions import UnknownComputer
from .asymptotic_calculator import compute_asymptotic_confidence_level
from .toy_calculator import compute_toy_confidence_level

__all__ = ["get_confidence_level_computer"]


def get_confidence_level_computer(
    name: str,
) -> Callable[[float, float, Text], Tuple[List[float], List[float]]]:
    r"""
    Retreive confidence level computer

    Args:
        name (``str``): Name of the computer

          * ``"asymptotic"``: uses asymptotic confidence level computer
          * ``"toy"``: uses toy confidence level computer

    Raises:
        ``UnknownComputer``: If ``name`` input does not correspond any of the above.

    Returns:
        ``Callable[[float, float, Text], Tuple[List[float], List[float]]]``:
        Confidence level computer.
    """
    try:
        return {
            "asymptotic": compute_asymptotic_confidence_level,
            "toy": compute_toy_confidence_level,
        }[name]
    except KeyError as excl:
        raise UnknownComputer(
            f"{name} is unknown. Available confidence level computers are 'toy' or 'asymptotic'"
        ) from excl
