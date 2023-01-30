from enum import Enum


class AvailableBackends(Enum):
    pyhf = "pyhf"
    simplified_likelihoods = "simplified_likelihoods"

    def __repr__(self):
        return self.value

    def __str__(self):
        return str(repr(self))
