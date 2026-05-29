from enum import Enum, auto

__all__ = ["ExpectationType"]

# pylint: disable=C0103


class ExpectationType(Enum):
    """
    Expectation type has been used to determine the nature of the statistical model through out the package.
    It consists of three main arguments:

        * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
          prescription which means that the experimental data will be assumed to be the truth
          (default).
        * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
          post-fit prescription which means that the experimental data will be assumed to be
          the truth.
        * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
          prescription which means that the SM will be assumed to be the truth.

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

    __hash__ = Enum.__hash__
