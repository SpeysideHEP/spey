from enum import Enum, auto

__all__ = [
    "AvailableBackends",
    "pyhf_backend",
    "simplifiedlikelihood_backend",
]


class AvailableBackends(Enum):
    pyhf = auto()
    simplified_likelihoods = auto()

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
