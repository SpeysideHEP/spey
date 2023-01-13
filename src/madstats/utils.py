from enum import Enum, auto
from dataclasses import dataclass, field
import numpy as np
from typing import Text, List


class ExpectationType(Enum):
    apriori = auto()
    aposteriori = auto()
    observed = auto()

    def __repr__(self):
        if self == self.apriori:
            return str(self) + ": Set observed events to expected background events."
        elif self == self.aposteriori:
            return str(self) + ": Compute likelihood fit with observed events."
        return str(self)

    def __eq__(self, other):
        current = str(self).split(".")[1]
        if isinstance(other, ExpectationType):
            other = str(other).split(".")[1]
            return current == other
        elif isinstance(other, str):
            current = str(self).split(".")[1]
            return other == current
        elif isinstance(other, bool):
            return self == (ExpectationType.apriori if other else ExpectationType.observed)
        else:
            raise ValueError(f"Unknown comparison: type({other}) == {type(other)}")

    @staticmethod
    def as_expectationtype(other):
        if isinstance(other, ExpectationType):
            return other
        elif isinstance(other, (str, bool)):
            if other == "aposteriori":
                return ExpectationType.aposteriori
            elif other == "apriori" or other == True:
                return ExpectationType.apriori
            else:
                return ExpectationType.observed


class Units(Enum):
    pb = "picobarn"
    fb = "femtobarn"
    GeV = "GeV"
    TeV = "TeV"

    def __repr__(self):
        return str(self).split(".")[1]

    def __float__(self):
        if self in [Units.pb, Units.GeV]:
            return 1.0
        elif self in [Units.fb, Units.TeV]:
            return 1000.0

    def __int__(self):
        if self in [Units.pb, Units.GeV]:
            return 1
        elif self in [Units.fb, Units.TeV]:
            return 1000

    def __mul__(self, other):
        if isinstance(other, (float, int, np.ndarray)):
            return float(self) * other
        else:
            raise ValueError(f"Multiplication request with unknown type: {type(other)}")


@dataclass(frozen=True)
class Region:
    nobs: int
    nb: float
    delta_nb: float
    signal_eff: float
    name: Text = field(default="__unknown_region__")


@dataclass(frozen=True)
class Analysis:
    name: Text = field(default="__unknown_analysis__")
    sqrts: float = field(default=13.0 * Units.GeV)
    regiondata: List[Region] = field(default_factory=list)
    luminosity: float = field(default=1.0 * Units.fb)

    def __repr__(self):
        txt = (
            f"Analysis(\n    name = '{self.name}',"
            f"\n    sqrts = {self.sqrts:.1f} [GeV],"
            f"\n    luminosity = {self.luminosity:.1f} [1/fb]"
            f"\n    number of regions = {len(self)}\n"
        )
        for reg in self:
            txt += "        " + str(repr(reg)) + "\n"
        txt += "\n)"
        return txt

    def __len__(self):
        return len(self.regiondata)

    @property
    def regions(self):
        return [reg.name for reg in self.regiondata]

    def __iter__(self):
        for reg in self.regiondata:
            yield reg

    def __getitem__(self, item):
        for reg in self.regiondata:
            if reg.name == item:
                return reg
