from typing import Callable
import scipy
import numpy as np

from .sldata import expansion_output


def sample_generator(
    pars: np.ndarray,
    signal: np.ndarray,
    background: np.ndarray,
    third_moment_expansion: expansion_output,
) -> Callable[[int], np.ndarray]:
    """
    Function to generate samples

    :param pars (`np.ndarray`): nuisance parameters
    :param signal (`np.ndarray`): signal yields
    :param background (`np.ndarray`): expected background yields
    :param third_moment_expansion (`expansion_output`): third moment expansion
    :return `Callable[[int], np.ndarray]`: Function to draw samples from
    """

    if third_moment_expansion.A is None:
        lmbda = background + pars[1:] + pars[0] * signal
    else:
        lmbda = (
            pars[0] * signal
            + third_moment_expansion.A
            + pars[1:]
            + third_moment_expansion.C
            * np.square(pars[1:])
            / np.square(third_moment_expansion.B)
        )
    lmbda = np.clip(lmbda, 1e-5, None)

    cov = (
        third_moment_expansion.V(pars[1:])
        if callable(third_moment_expansion.V)
        else third_moment_expansion.V
    )

    distributions = [
        {"type": np.random.poisson, "kwargs": {"lam": lmbda}},
        {
            "type": np.random.multivariate_normal,
            "kwargs": {"mean": np.zeros(len(background)), "cov": cov},
        },
    ]

    def sampler(number_of_samples: int) -> np.ndarray:
        """
        Sample generator for the statistical model

        :param number_of_samples (`int`): number of samples to be drawn from the model
        :return `np.ndarray`: Sampled observations
        """
        data = np.zeros((number_of_samples, len(distributions)))
        for idx, dist in enumerate(distributions):
            data[:, idx] = dist["type"](
                size=(number_of_samples, len(background)), **dist["kwargs"]
            )
        random_idx = np.random.choice(
            np.arange(2), size=(number_of_samples,), p=[0.5] * 2
        )

        return data[np.arange(number_of_samples), random_idx]

    return sampler
