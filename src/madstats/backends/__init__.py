from enum import Enum, auto

class available_backends(Enum):
    pyhf = auto()
    simplified_likelihoods = auto()