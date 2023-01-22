from enum import Enum, auto

class AvailableBackends(Enum):
    pyhf = auto()
    simplified_likelihoods = auto()

    def __repr__(self):
        return str(self).split(".")[-1]