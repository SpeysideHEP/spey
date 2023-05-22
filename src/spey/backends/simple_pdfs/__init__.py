"""Simple PDF interface"""

from typing import Optional, Text, Callable, List, Union, Tuple, Any
from autograd import numpy as np
from autograd import grad, hessian, jacobian

from scipy.optimize import NonlinearConstraint

from spey.optimizer import fit
from spey.base import BackendBase
from spey.base import ModelConfig
from spey.utils import ExpectationType
from spey._version import __version__
from spey.helper_functions import covariance_to_correlation
from spey.backends.distributions import MainModel, ConstraintModel
from .utils import solve_bifurcation_for_gamma


def __dir__():
    return []


# pylint: disable=E1101,E1120


class UnknownModelDeffinition(Exception):
    """Unknown statistical model definition exception"""


class SimplePDFBase(BackendBase):

    name: Text = "simple_pdfs.base"
    """Name of the backend"""
    version: Text = __version__
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = __version__
    """Spey version required for the backend"""

    __slots__ = ["_model", "_main_model", "constraints"]

    def __init__(
        self,
        signal_yields: List[float],
        background_yields: List[float],
        data: List[float],
        absolute_uncertainties: List[float] = None,
        absolute_uncertainty_envelops: List[Tuple[float, float]] = None,
    ):
        assert (
            len(signal_yields) == len(background_yields) == len(data)
        ), "Input dimensionality does not match."

        self.signal_yields = np.array(signal_yields)
        self.background_yields = np.array(background_yields)
        self.data = np.array(data)
        if absolute_uncertainties is None:
            self.absolute_uncertainties = None
        else:
            assert len(absolute_uncertainties) == len(
                data
            ), "Dimensionality of the uncertainties does not match"
            self.absolute_uncertainties = np.array(absolute_uncertainties)
        if absolute_uncertainty_envelops is None:
            self.upper_envelop = None
            self.lower_envelop = None
        else:
            assert len(absolute_uncertainty_envelops) == len(
                data
            ), "Dimensionality of the uncertainties does not match"
            up, dn = [], []
            for upper, lower in absolute_uncertainty_envelops:
                up.append(upper)
                dn.append(lower)
            self.upper_envelop = np.array(up)
            self.lower_envelop = np.array(dn)

        minimum_poi = -np.min(
            self.background_yields[self.signal_yields > 0.0]
            / self.signal_yields[self.signal_yields > 0.0]
        )

        self._config = ModelConfig(
            0,
            minimum_poi,
            [1.0] * (len(data) + 1),
            [(minimum_poi, 10)] + [(None, None)] * len(data),
        )

        self._main_model = None
        self._constraint_model = None
        self.constraints = []

    @property
    def main_model(self) -> MainModel:
        """
        retreive the main model distribution

        Raises:
            ``UnknownModelDeffinition``: If statistical model is not defined

        Returns:
            ``MainModel``:
            Main model distribution.
        """
        if self._main_model is None:

            def lam(pars: np.ndarray) -> np.ndarray:
                """Compute lambda for Main model"""
                return pars[0] * self.signal_yields + self.background_yields

            self._main_model = MainModel(lam)

        return self._main_model

    @property
    def constraint_model(self) -> ConstraintModel:
        """
        retreive the constraint model distribution

        Returns:
            ``MainModel``:
            Main model distribution.
        """
        return self._constraint_model

    @property
    def is_alive(self) -> bool:
        """Returns True if at least one bin has non-zero signal yield."""
        return np.any(self.signal_yields > 0.0)

    def config(
        self, allow_negative_signal: bool = True, poi_upper_bound: float = 10.0
    ) -> ModelConfig:
        r"""
        Model configuration.

        Args:
            allow_negative_signal (``bool``, default ``True``): If ``True`` :math:`\hat\mu`
              value will be allowed to be negative.
            poi_upper_bound (``float``, default ``40.0``): upper bound for parameter
              of interest, :math:`\mu`.

        Returns:
            ~spey.base.ModelConfig:
            Model configuration. Information regarding the position of POI in
            parameter list, suggested input and bounds.
        """
        if allow_negative_signal and poi_upper_bound == 10.0:
            return self._config

        return ModelConfig(
            self._config.poi_index,
            self._config.minimum_poi,
            self._config.suggested_init,
            [(0, poi_upper_bound)] + self._config.suggested_bounds[1:],
        )

    def get_objective_function(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.ndarray] = None,
        do_grad: bool = True,
    ) -> Callable[[np.ndarray], Union[Tuple[float, np.ndarray], float]]:
        r"""
        Objective function i.e. twice negative log-likelihood, :math:`-2\log\mathcal{L}(\mu, \theta)`

        Args:
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            data (``np.ndarray``, default ``None``): input data that to fit
            do_grad (``bool``, default ``True``): If ``True`` return objective and its gradient
              as ``tuple`` if ``False`` only returns objective function.

        Returns:
            ``Callable[[np.ndarray], Union[float, Tuple[float, np.ndarray]]]``:
            Function which takes fit parameters (:math:`\mu` and :math:`\theta`) and returns either
            objective or objective and its gradient.
        """

        data = self.data if data is None else data

        if self.constraint_model is None:

            def negative_loglikelihood(pars: np.ndarray) -> np.ndarray:
                """Compute twice negative log-likelihood"""
                return -self.main_model.log_prob(pars, data[: len(self.data)])

        else:

            def negative_loglikelihood(pars: np.ndarray) -> np.ndarray:
                """Compute twice negative log-likelihood"""
                return -self.main_model.log_prob(
                    pars, data[: len(self.data)]
                ) - self.constraint_model.log_prob(pars[1:])

        if do_grad:
            grad_negative_loglikelihood = grad(negative_loglikelihood, argnum=0)
            return lambda pars: (
                negative_loglikelihood(pars),
                grad_negative_loglikelihood(pars),
            )

        return negative_loglikelihood

    def get_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.array] = None,
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        r"""
        Generate function to compute :math:`\log\mathcal{L}(\mu, \theta)` where :math:`\mu` is the
        parameter of interest and :math:`\theta` are nuisance parameters.

        Args:
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            data (``np.array``, default ``None``): input data that to fit

        Returns:
            ``Callable[[np.ndarray], float]``:
            Function that takes fit parameters (:math:`\mu` and :math:`\theta`) and computes
            :math:`\log\mathcal{L}(\mu, \theta)`.
        """
        data = self.data if data is None else data

        if self.constraint_model is None:
            return lambda pars: self.main_model.log_prob(pars, data[: len(self.data)])

        return lambda pars: self.main_model.log_prob(
            pars, data[: len(self.data)]
        ) + self.constraint_model.log_prob(pars[1:])

    def get_hessian_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.ndarray] = None,
    ) -> Callable[[np.ndarray], float]:
        r"""
        Currently Hessian of :math:`\log\mathcal{L}(\mu, \theta)` is only used to compute
        variance on :math:`\mu`. This method returns a callable function which takes fit
        parameters (:math:`\mu` and :math:`\theta`) and returns Hessian.

        Args:
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
              p-values to be computed.

              * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
              * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            data (``np.ndarray``, default ``None``): input data that to fit

        Returns:
            ``Callable[[np.ndarray], float]``:
            Function that takes fit parameters (:math:`\mu` and :math:`\theta`) and
            returns Hessian of :math:`\log\mathcal{L}(\mu, \theta)`.
        """
        log_prob = self.get_logpdf_func(expected, data)
        return hessian(log_prob, argnum=0)

    def get_sampler(self, pars: np.ndarray) -> Callable[[int], np.ndarray]:
        r"""
        Retreives the function to sample from.

        Args:
            pars (``np.ndarray``): fit parameters (:math:`\mu` and :math:`\theta`)
            include_auxiliary (``bool``): wether or not to include auxiliary data
              coming from the constraint model.

        Returns:
            ``Callable[[int, bool], np.ndarray]``:
            Function that takes ``number_of_samples`` as input and draws as many samples
            from the statistical model.
        """

        def sampler(sample_size: int, include_auxiliary: bool = True) -> np.ndarray:
            """
            Fucntion to generate samples.

            Args:
                sample_size (``int``): number of samples to be generated.
                include_auxiliary (``bool``): wether or not to include auxiliary data
                    coming from the constraint model.

            Returns:
                ``np.ndarray``:
                generated samples
            """
            sample = self.main_model.sample(pars, sample_size)

            if include_auxiliary and self.constraint_model is not None:
                constraint_sample = self.constraint_model.sample(pars[1:], sample_size)
                sample = np.hstack([sample, constraint_sample])

            return sample

        return sampler

    def expected_data(
        self, pars: List[float], include_auxiliary: bool = True
    ) -> List[float]:
        r"""
        Compute the expected value of the statistical model

        Args:
            pars (``List[float]``): nuisance, :math:`\theta` and parameter of interest,
              :math:`\mu`.
            include_auxiliary (``bool``): wether or not to include auxiliary data
              coming from the constraint model.

        Returns:
            ``List[float]``:
            Expected data of the statistical model
        """
        data = self.main_model.expected_data(pars)

        if include_auxiliary and self.constraint_model is not None:
            data = np.hstack([data, self.constraint_model.expected_data()])

        return data


class GeneralisedPoisson(SimplePDFBase):
    """
    Experimental

    Args:
        signal_yields (``List[float]``): _description_
        background_yields (``List[float]``): _description_
        data (``List[float]``): _description_
        absolute_uncertainty_envelops (``List[Tuple[float, float]]``, default ``None``): _description_
        niterations (``int``, default ``10000``): _description_

    Returns:
        ``_type_``:
        _description_
    """

    name: Text = "simple_pdfs.generalised_poisson"
    """Name of the backend"""
    version: Text = SimplePDFBase.version
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = SimplePDFBase.spey_requires
    """Spey version required for the backend"""
    doi: List[Text] = ["10.48550/arXiv.physics/0406120"]
    """Citable DOI for the backend"""
    arXiv: List[Text] = ["physics/0406120"]
    """arXiv reference for the backend"""

    def __init__(
        self,
        signal_yields: List[float],
        background_yields: List[float],
        data: List[float],
        absolute_uncertainty_envelops: List[Tuple[float, float]] = None,
        niterations: int = 10000,
    ):
        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            absolute_uncertainty_envelops=absolute_uncertainty_envelops,
        )

        gamma = solve_bifurcation_for_gamma(
            self.lower_envelop, self.upper_envelop, niterations
        )

        nu = 0.5 / (gamma * self.upper_envelop - np.log(1.0 + gamma * self.upper_envelop))
        alpha = nu * gamma

        self._constraint_model = ConstraintModel(
            "generalisedpoisson", self.background_yields, alpha, nu
        )

        self.constraints = [
            NonlinearConstraint(lambda pars: pars[1:], self.background_yields, np.inf)
        ]


class VariableGaussian(SimplePDFBase):
    r"""
    Variable Gaussian has been inspired by :xref:`physics/0406120` sec 3.5. Assuming
    that the upper and lower uncertainty envelops are Gaussain, this method modifies
    the covariance matrix of the multivariate normal distribution as follows

    .. math::

        \sigma(x) &= \frac{2\sigma_+\sigma_-}{\sigma_++\sigma_-} +
        \frac{\sigma_+-\sigma_-}{\sigma_++\sigma_-}(x-\hat{x})

        \Sigma(x) &= \sigma(x) \rho \sigma(x)

    where :math:`\sigma_\pm` are upper and lower uncertainty envelops and :math:`\hat{x}` are
    the best fit values i.e. background yields. Thus the likelihood distribution can be written
    as

    .. math::

        \mathcal{L}(\mu, \theta) = {\rm Poiss}(n_{obs}| \mu n_s + n_{bkg})
        \mathcal{N}(\theta| n_{bkg}, \Sigma(x))

    .. note::

        :xref:`physics/0406120` presents two forms for Variable Gaussian in sections 3.5 and
        3.6, we observed that both forms result in the same outcome. Thus this implementation
        only includes form 1.

    Args:
        signal_yields (``np.ndarray``): signal yields
        background_yields (``np.ndarray``): background yields
        data (``np.ndarray``): observations
        correlation_matrix (``np.ndarray``): correlations between regions
        absolute_uncertainty_envelops (``List[Tuple[float, float]]``): upper and lower uncertainty
          envelops for each background yield.
    """

    name: Text = "simple_pdfs.variable_gaussian"
    """Name of the backend"""
    version: Text = SimplePDFBase.version
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = SimplePDFBase.spey_requires
    """Spey version required for the backend"""
    doi: List[Text] = ["10.48550/arXiv.physics/0406120"]
    """Citable DOI for the backend"""
    arXiv: List[Text] = ["physics/0406120"]
    """arXiv reference for the backend"""

    def __init__(
        self,
        signal_yields: np.ndarray,
        background_yields: np.ndarray,
        data: np.ndarray,
        correlation_matrix: np.ndarray,
        absolute_uncertainty_envelops: List[Tuple[float, float]],
    ):
        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            absolute_uncertainty_envelops=absolute_uncertainty_envelops,
        )

        self.correlation_matrix = np.array(correlation_matrix)
        assert self.correlation_matrix.shape[0] == len(
            self.upper_envelop
        ), "Dimensionality of the correlation matrix does not match the backgroun."

        # arXiv:physics/0406120 eq 16
        sigma = (
            2.0
            * self.upper_envelop
            * self.lower_envelop
            / (self.upper_envelop + self.lower_envelop)
        )
        sigma_prime = (self.upper_envelop - self.lower_envelop) / (
            self.upper_envelop + self.lower_envelop
        )

        def VariableCov(pars: np.ndarray) -> np.ndarray:
            """Compute covariance matrix for multivariate normal distribution"""
            effective_sigma = np.sqrt(
                sigma + sigma_prime * (pars - self.background_yields)
            ) * np.eye(len(self.background_yields))
            return effective_sigma @ self.correlation_matrix @ effective_sigma

        self._constraint_model = ConstraintModel(
            "multivariatenormal", self.background_yields, VariableCov
        )

        assert self.main_model is not None, "Main model has not been constructed properly"
