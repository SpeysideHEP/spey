import numpy as np
import scipy

from typing import Callable, Union


def compute_confidence_level(
    mu: Union[float, np.ndarray],
    negloglikelihood_asimov: Union[Callable[[np.ndarray], float], float],
    min_negloglikelihood_asimov: float,
    negloglikelihood: Union[Callable[[np.ndarray], float], float],
    min_negloglikelihood: float,
) -> float:
    """
    Compute confidence limit with respect to a given mu

    :param mu: POI (signal strength)
    :param negloglikelihood_asimov: POI dependent negative log-likelihood function
                                    based on asimov data
    :param min_negloglikelihood_asimov: minimum negative log-likelihood for asimov data
    :param negloglikelihood: POI dependent negative log-likelihood function
    :param min_negloglikelihood: minimum negative log-likelihood
    :return: confidence limit
    """
    if isinstance(mu, (float, int)):
        mu = np.array([float(mu)])
    elif len(mu) == 0:
        mu = np.array([mu])

    if callable(negloglikelihood_asimov):
        nllA = negloglikelihood_asimov(mu)
    else:
        nllA = negloglikelihood_asimov

    if callable(negloglikelihood):
        nll = negloglikelihood(mu)
    else:
        nll = negloglikelihood

    qmu = 0.0 if 2.0 * (nll - min_negloglikelihood) < 0.0 else 2.0 * (nll - min_negloglikelihood)
    sqmu = np.sqrt(qmu)
    qA = max(2.0 * (nllA - min_negloglikelihood_asimov), 0.0)
    sqA = np.sqrt(qA)
    if qA >= qmu:
        CLsb = 1.0 - scipy.stats.multivariate_normal.cdf(sqmu)
        CLb = scipy.stats.multivariate_normal.cdf(sqA - sqmu)
    else:
        if qA == 0.0:
            CLsb, CLb = 1.0, 1.0
        else:
            CLsb = 1.0 - scipy.stats.multivariate_normal.cdf((qmu + qA) / (2.0 * sqA))
            CLb = 1.0 - scipy.stats.multivariate_normal.cdf((qmu - qA) / (2.0 * sqA))

    if CLb == 0.0:
        return 0.0

    return CLsb / CLb
