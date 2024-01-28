from enum import Enum, auto

__all__ = ["ExpectationType"]

# pylint: disable=C0103


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
            return self == (
                ExpectationType.apriori if other else ExpectationType.observed
            )
        if other is None:
            return False

        raise ValueError(f"Unknown comparison: type({other}) = {type(other)}")
