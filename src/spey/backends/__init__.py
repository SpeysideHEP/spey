from enum import Enum, auto

from .pyhf_backend.interface import PyhfInterface
from .pyhf_backend.data import Data as PyhfData
from .simplifiedlikelihood_backend.interface import SimplifiedLikelihoodInterface
from .simplifiedlikelihood_backend.data import Data as SLData

__all__ = [
    "AvailableBackends",
    "PyhfInterface",
    "PyhfData",
    "SimplifiedLikelihoodInterface",
    "SLData",
]


class AvailableBackends(Enum):
    pyhf = auto()
    simplified_likelihoods = auto()

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
