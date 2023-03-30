from typing import Text, List
from enum import Enum, auto
from dataclasses import dataclass, field
import numpy as np

__all__ = ["ExpectationType"]


class ExpectationType(Enum):
    """
    Expectation type has been used to determine the nature of the statistical model through out the package.
    It consists of three main arguments:

        * :obj:`observed` : indicates that the fit of the statistical model will be done over experimental data
        * :obj:`aposteriori`: as in :obj:`observed` the fit will be done over data where the likelihood results will
          be identical to :obj:`observed`, computation of :math:`CL_s` values will be done for by centralising the test
          statistics around background.
        * :obj:`apriori`: theorists are generatly interested in difference of their model from the SM simulation. Hence
          this option will overwrite the observed data in the statistical model with simulated background values and performs
          the computation with respect to prefit values, meaning prior to the experimental observation. :math:`CL_s` values
          are again computed by centralising the test statistics around the background i.e. SM background.

    User can simply set the value of :obj:`expected` to a desired :obj:`ExpectationType`:

    .. code-block:: python3

        >>> expected = spey.ExpectationType.aposteriori

    This will trigger appropriate action to be taken through out the package.
    """

    apriori = auto()
    aposteriori = auto()
    observed = auto()

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        current = str(self)
        if isinstance(other, ExpectationType):
            other = str(other)
            return current == other
        if isinstance(other, str):
            current = str(self)
            return other == current
        if isinstance(other, bool):
            return self == (ExpectationType.apriori if other else ExpectationType.observed)
        if other is None:
            return False

        raise ValueError(f"Unknown comparison: type({other}) = {type(other)}")


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
class Dataset:
    xsection: field(default=1.0, repr=True)
    name: Text = field(default="__unknown_dataset__", repr=True)

    def __repr__(self):
        return f"Dataset(name = '{self.name}', xsection = {self.xsection:.5f} [pb])"


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
    sqrts: float = field(default=13.0)
    regiondata: List[Region] = field(default_factory=list)
    luminosity: float = field(default=1.0)

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
