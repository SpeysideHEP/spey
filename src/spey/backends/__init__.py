from enum import Enum, auto

__all__ = ["AvailableBackends", "simplifiedlikelihood_backend", "pyhf_backend"]

class AvailableBackends(Enum):
    pyhf = auto()
    simplified_likelihoods = auto()

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
