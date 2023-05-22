import numpy as np


def solve_bifurcation_for_gamma(
    lower_bounds: np.ndarray, upper_bounds: np.ndarray, number_of_iterations: int = 10000
) -> np.ndarray:
    """
    Compute Gamma via bifurcation see arXiv:pyhsics/0406120 eq. 11

    Args:
        lower_bounds (``np.ndarray``): lower uncertainties
        upper_bounds (``np.ndarray``): upper uncertainties
        number_of_iterations (``int``, default ``10000``): number of iterations

    Returns:
        ``np.ndarray``:
        Gamma value for each asymmetric uncertainty
    """

    def func(low, up):
        a = 0.0
        b = 1.0 / low
        for _ in range(number_of_iterations):
            x = (a + b) / 2.0
            if np.exp(-x * (low + up)) <= (1.0 - low * x) / (1.0 + up * x):
                a = x
            else:
                b = x
        return a

    new_bound = []
    for lower, upper in zip(lower_bounds, upper_bounds):
        new_bound.append(func(lower, upper))

    return np.array(new_bound)
