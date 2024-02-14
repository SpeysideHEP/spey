"""Interface for default PDF sets"""

import logging
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union

from autograd import value_and_grad, hessian, jacobian
from autograd import numpy as np
from scipy.optimize import NonlinearConstraint

from spey._version import __version__
from spey.backends.distributions import ConstraintModel, MainModel
from spey.base import BackendBase, ModelConfig
from spey.helper_functions import covariance_to_correlation
from spey.optimizer import fit
from spey.utils import ExpectationType

from .third_moment import third_moment_expansion
from .uncertainty_synthesizer import signal_uncertainty_synthesizer

# pylint: disable=E1101,E1120
log = logging.getLogger("Spey")

# pylint: disable=W1203


class DefaultPDFBase(BackendBase):
    """
    Default PDF backend base

    Args:
        signal_yields (``np.ndarray``): signal yields
        background_yields (``np.ndarray``): background yields
        data (``np.ndarray``): observed yields
        covariance_matrix (``np.ndarray``): covariance matrix. The dimensionality of each axis has
          to match with ``background_yields``, ``signal_yields``, and ``data`` inputs.

          .. warning::

            The diagonal terms of the covariance matrix involves squared absolute background
            uncertainties. In case of uncorralated bins user should provide a diagonal matrix
            with squared background uncertainties.

        signal_uncertainty_configuration (``Dict[Text, Any]]``, default ``None``): Configuration
          input for signal uncertainties

          * absolute_uncertainties (``List[float]``): Absolute uncertainties for the signal
          * absolute_uncertainty_envelops (``List[Tuple[float, float]]``): upper and lower
              uncertainty envelops
          * correlation_matrix (``List[List[float]]``): Correlation matrix
          * third_moments (``List[float]``): diagonal elemetns of the third moment

    .. note::

        To enable a differentiable statistical model, all inputs are wrapped with
        :func:`autograd.numpy.array` function.
    """

    name: Text = "default_pdf.base"
    """Name of the backend"""
    version: Text = __version__
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = __version__
    """Spey version required for the backend"""

    __slots__ = [
        "_model",
        "_main_model",
        "_constraint_model",
        "constraints",
        "signal_uncertainty_configuration",
    ]

    def __init__(
        self,
        signal_yields: np.ndarray,
        background_yields: np.ndarray,
        data: np.ndarray,
        covariance_matrix: Optional[
            Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]
        ] = None,
        signal_uncertainty_configuration: Optional[Dict[Text, Any]] = None,
    ):
        self.data = np.array(data, dtype=np.float64)
        self.signal_yields = np.array(signal_yields, dtype=np.float64)
        self.background_yields = np.array(background_yields, dtype=np.float64)
        self.covariance_matrix = (
            np.array(covariance_matrix, dtype=np.float64)
            if not callable(covariance_matrix) and covariance_matrix is not None
            else covariance_matrix
        )
        if signal_uncertainty_configuration is None:
            self.signal_uncertainty_configuration = {}
        else:
            self.signal_uncertainty_configuration = signal_uncertainty_synthesizer(
                signal_yields=self.signal_yields,
                **signal_uncertainty_configuration,
                domain=slice(len(background_yields) + 1, None),
            )

        minimum_poi = -np.inf
        if self.is_alive:
            minimum_poi = -np.min(
                self.background_yields[self.signal_yields > 0.0]
                / self.signal_yields[self.signal_yields > 0.0]
            )
        log.debug(f"Min POI set to : {minimum_poi}")

        self._main_model = None
        self._constraint_model = None
        self.constraints = []
        """Constraints to be used during optimisation process"""

        self._config = ModelConfig(
            poi_index=0,
            minimum_poi=minimum_poi,
            suggested_init=[1.0] * (len(data) + 1)
            + (signal_uncertainty_configuration is not None)
            * ([1.0] * len(signal_yields)),
            suggested_bounds=[(minimum_poi, 10)]
            + [(None, None)] * len(data)
            + (signal_uncertainty_configuration is not None)
            * ([(None, None)] * len(signal_yields)),
        )

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

    @property
    def constraint_model(self) -> ConstraintModel:
        """retreive constraint model distribution"""
        if self._constraint_model is None:
            corr = covariance_to_correlation(self.covariance_matrix)
            self._constraint_model = ConstraintModel(
                [
                    {
                        "distribution_type": "multivariatenormal",
                        "args": [np.zeros(len(self.data)), corr],
                        "kwargs": {"domain": slice(1, None)},
                    }
                ]
                + self.signal_uncertainty_configuration.get("constraint", [])
            )
        return self._constraint_model

    @property
    def main_model(self) -> MainModel:
        """retreive the main model distribution"""
        if self._main_model is None:
            A = self.background_yields
            B = np.sqrt(np.diag(self.covariance_matrix))

            def lam(pars: np.ndarray) -> np.ndarray:
                """
                Compute lambda for Main model with third moment expansion.
                For details see above eq 2.6 in :xref:`1809.05548`

                Args:
                    pars (``np.ndarray``): nuisance parameters

                Returns:
                    ``np.ndarray``:
                    expectation value of the poisson distribution with respect to
                    nuisance parameters.
                """
                return pars[0] * self.signal_yields + A + B * pars[slice(1, len(B) + 1)]

            def constraint(pars: np.ndarray) -> np.ndarray:
                """Compute constraint term"""
                return A + B * pars[slice(1, len(B) + 1)]

            jac_constr = jacobian(constraint)

            self.constraints.append(
                NonlinearConstraint(constraint, 0.0, np.inf, jac=jac_constr)
            )

            if self.signal_uncertainty_configuration.get("lambda", None) is not None:
                signal_lambda = self.signal_uncertainty_configuration["lambda"]

                def poiss_lamb(pars: np.ndarray) -> np.ndarray:
                    """combined lambda expression"""
                    return lam(pars) + signal_lambda(pars)

            else:
                poiss_lamb = lam

            self._main_model = MainModel(poiss_lamb)

        return self._main_model

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
        current_data = (
            self.background_yields if expected == ExpectationType.apriori else self.data
        )
        data = current_data if data is None else data
        log.debug(f"Data: {data}")

        def negative_loglikelihood(pars: np.ndarray) -> np.ndarray:
            """Compute twice negative log-likelihood"""
            return -self.main_model.log_prob(
                pars, data[: len(self.data)]
            ) - self.constraint_model.log_prob(pars)

        if do_grad:
            return value_and_grad(negative_loglikelihood, argnum=0)

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
        current_data = (
            self.background_yields if expected == ExpectationType.apriori else self.data
        )
        data = current_data if data is None else data
        log.debug(f"Data: {data}")

        return lambda pars: self.main_model.log_prob(
            pars, data[: len(self.data)]
        ) + self.constraint_model.log_prob(pars)

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
        current_data = (
            self.background_yields if expected == ExpectationType.apriori else self.data
        )
        data = current_data if data is None else data
        log.debug(f"Data: {data}")

        def log_prob(pars: np.ndarray) -> np.ndarray:
            """Compute log-probability"""
            return self.main_model.log_prob(
                pars, data[: len(self.data)]
            ) + self.constraint_model.log_prob(pars)

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

            if include_auxiliary:
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

        if include_auxiliary:
            data = np.hstack([data, self.constraint_model.expected_data()])
        return data


class UncorrelatedBackground(DefaultPDFBase):
    r"""
    Interface for uncorrelated background uncertainties.
    This simple backend is designed to handle single or multi-bin statistical models
    with uncorrelated uncertainties. Inputs has to be given as list of ``NumPy`` array where
    each input should include same number of regions. It assumes absolute uncertainties
    on the background sample e.g. for a background sample yield reported as
    :math:`3.1\pm0.5` the background yield is ``3.1`` and the absolute uncertainty is
    ``0.5``. It forms a combination of normal and poisson distributions to from the input
    data where the log-probability is computed as sum of all normal and poisson distributions.

    Args:
        signal_yields (``List[float]``): signal yields
        background_yields (``List[float]``): background yields
        data (``List[int]``): observations
        absolute_uncertainties (``List[float]``): absolute uncertainties on the background
        signal_uncertainty_configuration (``Dict[Text, Any]]``, default ``None``): Configuration
          input for signal uncertainties

          * absolute_uncertainties (``List[float]``): Absolute uncertainties for the signal
          * absolute_uncertainty_envelops (``List[Tuple[float, float]]``): upper and lower
              uncertainty envelops
          * correlation_matrix (``List[List[float]]``): Correlation matrix
          * third_moments (``List[float]``): diagonal elemetns of the third moment

    .. note::

        Each input should have the same dimensionality, i.e. if ``data`` has three regions,
        ``signal_yields``, ``background_yields`` and ``absolute_uncertainties`` inputs should
        have three regions as well.

    Example:

    .. code:: python3

        >>> import spey
        >>> stat_wrapper = spey.get_backend('default_pdf.uncorrelated_background')

        >>> data = [1, 3]
        >>> signal = [0.5, 2.0]
        >>> background = [2.0, 2.8]
        >>> background_unc = [1.1, 0.8]

        >>> stat_model = stat_wrapper(
        ...     signal, background, data, background_unc, analysis="multi-bin", xsection=0.123
        ... )
        >>> print("1-CLs : %.3f" % tuple(stat_model.exclusion_confidence_level()))
    """

    name: Text = "default_pdf.uncorrelated_background"
    """Name of the backend"""
    version: Text = DefaultPDFBase.version
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = DefaultPDFBase.spey_requires
    """Spey version required for the backend"""

    def __init__(
        self,
        signal_yields: List[float],
        background_yields: List[float],
        data: List[int],
        absolute_uncertainties: List[float],
        signal_uncertainty_configuration: Optional[Dict[Text, Any]] = None,
    ):
        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            covariance_matrix=None,
            signal_uncertainty_configuration=signal_uncertainty_configuration,
        )

        B = np.array(absolute_uncertainties)

        self._constraint_model: ConstraintModel = ConstraintModel(
            [
                {
                    "distribution_type": "normal",
                    "args": [np.zeros(len(self.data)), np.ones(len(B))],
                    "kwargs": {"domain": slice(1, len(B) + 1)},
                }
            ]
            + self.signal_uncertainty_configuration.get("constraint", [])
        )

        def lam(pars: np.ndarray) -> np.ndarray:
            """Compute lambda for Main model"""
            return (
                self.background_yields
                + pars[slice(1, len(B) + 1)] * B
                + pars[0] * self.signal_yields
            )

        def constraint(pars: np.ndarray) -> np.ndarray:
            """Compute the constraint term"""
            return self.background_yields + pars[slice(1, len(B) + 1)] * B

        jac_constr = jacobian(constraint)

        self.constraints.append(
            NonlinearConstraint(constraint, 0.0, np.inf, jac=jac_constr)
        )

        if self.signal_uncertainty_configuration.get("lambda", None) is not None:
            signal_lambda = self.signal_uncertainty_configuration["lambda"]

            def poiss_lamb(pars: np.ndarray) -> np.ndarray:
                """combined lambda expression"""
                return lam(pars) + signal_lambda(pars)

        else:
            poiss_lamb = lam

        self._main_model = MainModel(poiss_lamb)


class CorrelatedBackground(DefaultPDFBase):
    r"""
    Correlated multi-region statistical model.
    The correlation between each nuisance parameter has been captured via
    Multivariate Normal distribution and the log-probability distribution is
    combination of Multivariate Normal along with Poisson distribution.
    The Multivariate Normal distribution is constructed by the help
    of a covariance matrix provided by the user which captures the
    uncertainties and background correlations between each histogram bin.
    The probability distribution of a simplified likelihood can be formed as follows;

    .. math::

        \mathcal{L}_{SL}(\mu,\theta) = \underbrace{\left[\prod_i^N {\rm Poiss}\left(n^i_{obs}
        | \lambda_i(\mu, \theta)\right) \right]}_{\rm main\ model}
        \cdot \underbrace{\mathcal{N}(\theta | 0, \rho)}_{\rm constraint\ model}

    Here the first term is the so-called main model based on Poisson distribution centred around
    :math:`\lambda_i(\mu, \theta) = \mu n^i_{sig} + \theta + n^i_{bkg}` and the second term is the
    multivariate normal distribution centred around zero with the correlation matrix
    :math:`\rho`.

    Args:
        signal_yields (``np.ndarray``): signal yields
        background_yields (``np.ndarray``): background yields
        data (``np.ndarray``): observations
        covariance_matrix (``np.ndarray``): covariance matrix (square matrix)
        signal_uncertainty_configuration (``Dict[Text, Any]]``, default ``None``): Configuration
          input for signal uncertainties

          * absolute_uncertainties (``List[float]``): Absolute uncertainties for the signal
          * absolute_uncertainty_envelops (``List[Tuple[float, float]]``): upper and lower
              uncertainty envelops
          * correlation_matrix (``List[List[float]]``): Correlation matrix
          * third_moments (``List[float]``): diagonal elemetns of the third moment

    .. note::

        Each input should have the same dimensionality, i.e. if ``data`` has three regions,
        ``signal_yields`` and ``background_yields`` inputs should have three regions as well.
        Additionally ``covariance_matrix`` is expected to be square matrix, thus for a three
        region statistical model it is expected to be 3x3 matrix.

    Example:

    .. code:: python3

        >>> import spey

        >>> stat_wrapper = spey.get_backend('default_pdf.correlated_background')

        >>> signal_yields = [12.0, 11.0]
        >>> background_yields = [50.0, 52.0]
        >>> data = [51, 48]
        >>> covariance_matrix = [[3.,0.5], [0.6,7.]]

        >>> statistical_model = stat_wrapper(signal_yields,background_yields,data,covariance_matrix)
        >>> print(statistical_model.exclusion_confidence_level())
    """

    name: Text = "default_pdf.correlated_background"
    """Name of the backend"""
    version: Text = DefaultPDFBase.version
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = DefaultPDFBase.spey_requires
    """Spey version required for the backend"""

    def __init__(
        self,
        signal_yields: np.ndarray,
        background_yields: np.ndarray,
        data: np.ndarray,
        covariance_matrix: np.ndarray,
        signal_uncertainty_configuration: Optional[Dict[Text, Any]] = None,
    ):
        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            covariance_matrix=covariance_matrix,
            signal_uncertainty_configuration=signal_uncertainty_configuration,
        )

        assert self.main_model is not None, "Unable to build the main model"
        assert self.constraint_model is not None, "Unable to build the constraint model"


class ThirdMomentExpansion(DefaultPDFBase):
    r"""
    Simplified likelihood interface with third moment expansion.
    Third moment expansion follows simplified likelihood construction
    and modifies the :math:`\lambda` and :math:`\Sigma`. Using the expected
    background yields, :math:`m^{(1)}_i`, diagonal elements of the third moments,
    :math:`m^{(3)}_i` and the covariance matrix, :math:`m^{(2)}_{ij}`, one
    can write a modified correlation matrix and :math:`\lambda` function as follows

    .. math::

        C_i &= -sign(m^{(3)}_i) \sqrt{2 m^{(2)}_{ii}} \cos\left( \frac{4\pi}{3} +
        \frac{1}{3}\arctan\left(\sqrt{ \frac{8(m^{(2)}_{ii})^3}{(m^{(3)}_i)^2} - 1}\right) \right)

        B_i &= \sqrt{m^{(2)}_{ii} - 2 C_i^2}

        A_i &=  m^{(1)}_i - C_i

        \rho_{ij} &= \frac{1}{4C_iC_j} \left( \sqrt{(B_iB_j)^2 + 8C_iC_jm^{(2)}_{ij}} - B_iB_j \right)

    which further modifies :math:`\lambda_i(\mu, \theta) = \mu n^i_{sig} + A_i + B_i \theta_i + C_i \theta_i^2`
    and the multivariate normal has been modified via the inverse of the correlation matrix,
    :math:`\mathcal{N}(\theta | 0, \rho^{-1})`. See :xref:`1809.05548` Sec. 2 for details.

    Args:
        signal_yields (``np.ndarray``): signal yields
        background_yields (``np.ndarray``): background yields
        data (``np.ndarray``): observations
        covariance_matrix (``np.ndarray``): covariance matrix (square matrix)
        third_moment (``np.ndarray``): third moment for each region.
        signal_uncertainty_configuration (``Dict[Text, Any]]``, default ``None``): Configuration
          input for signal uncertainties

          * absolute_uncertainties (``List[float]``): Absolute uncertainties for the signal
          * absolute_uncertainty_envelops (``List[Tuple[float, float]]``): upper and lower
              uncertainty envelops
          * correlation_matrix (``List[List[float]]``): Correlation matrix
          * third_moments (``List[float]``): diagonal elemetns of the third moment

    .. note::

        Each input should have the same dimensionality, i.e. if ``data`` has three regions,
        ``signal_yields`` and ``background_yields`` inputs should have three regions as well.
        Additionally ``covariance_matrix`` is expected to be square matrix, thus for a three
        region statistical model it is expected to be 3x3 matrix. Following these,
        ``third_moment`` should also have three inputs.
    """

    name: Text = "default_pdf.third_moment_expansion"
    """Name of the backend"""
    version: Text = DefaultPDFBase.version
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = DefaultPDFBase.spey_requires
    """Spey version required for the backend"""
    doi: List[Text] = ["10.1007/JHEP04(2019)064"]
    """Citable DOI for the backend"""
    arXiv: List[Text] = ["1809.05548"]
    """arXiv reference for the backend"""

    def __init__(
        self,
        signal_yields: np.ndarray,
        background_yields: np.ndarray,
        data: np.ndarray,
        covariance_matrix: np.ndarray,
        third_moment: np.ndarray,
        signal_uncertainty_configuration: Optional[Dict[Text, Any]] = None,
    ):
        third_moments = np.array(third_moment)

        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            covariance_matrix=covariance_matrix,
            signal_uncertainty_configuration=signal_uncertainty_configuration,
        )

        A, B, C, corr = third_moment_expansion(
            self.background_yields, self.covariance_matrix, third_moments, True
        )

        def lam(pars: np.ndarray) -> np.ndarray:
            """
            Compute lambda for Main model with third moment expansion.
            For details see above eq 2.6 in :xref:`1809.05548`

            Args:
                pars (``np.ndarray``): nuisance parameters

            Returns:
                ``np.ndarray``:
                expectation value of the poisson distribution with respect to
                nuisance parameters.
            """
            nI = (
                A
                + B * pars[slice(1, len(B) + 1)]
                + C * np.square(pars[slice(1, len(B) + 1)])
            )
            return pars[0] * self.signal_yields + nI

        def constraint(pars: np.ndarray) -> np.ndarray:
            """Compute constraint term"""
            return (
                A
                + B * pars[slice(1, len(B) + 1)]
                + C * np.square(pars[slice(1, len(B) + 1)])
            )

        jac_constr = jacobian(constraint)

        self.constraints.append(
            NonlinearConstraint(constraint, 0.0, np.inf, jac=jac_constr)
        )

        if self.signal_uncertainty_configuration.get("lambda", None) is not None:
            signal_lambda = self.signal_uncertainty_configuration["lambda"]

            def poiss_lamb(pars: np.ndarray) -> np.ndarray:
                """combined lambda expression"""
                return lam(pars) + signal_lambda(pars)

        else:
            poiss_lamb = lam

        self._main_model = MainModel(poiss_lamb)
        self._constraint_model = ConstraintModel(
            [
                {
                    "distribution_type": "multivariatenormal",
                    "args": [np.zeros(len(self.data)), corr],
                    "kwargs": {"domain": slice(1, len(B) + 1)},
                }
            ]
            + self.signal_uncertainty_configuration.get("constraint", [])
        )


class EffectiveSigma(DefaultPDFBase):
    r"""
    Simplified likelihood interface with variable Gaussian.
    Variable Gaussian has been inspired by :xref:`physics/0406120` sec. 3.6. This method
    modifies the effective :math:`B:=\sigma_{eff}` term in the Poisson distribution of the
    simplified likelihood framework. Note that this approach does not modify the Gaussian
    of the likelihood, the naming of the approach is purely because it is originated from
    :xref:`physics/0406120` sec. 3.6. The effective sigma term of the Poissonian can be
    modified using upper, :math:`\sigma^+` and lower :math:`\sigma^-` envelops of the
    absolute background uncertainties (see eqs 18-19 in :xref:`physics/0406120`).

    .. math::

        \sigma_{eff}(\theta) = \sqrt{\sigma^+\sigma^-  + (\sigma^+ - \sigma^-)(\theta - n_{bkg})}

    where the simplified likelihoo is modified as

    .. math::

        \mathcal{L}(\mu,\theta) = \left[\prod_i^N{\rm Poiss}(n^i_{obs}|\mu n^i_s + n^i_{bkg} +
        \theta^i\sigma_{eff}^i(\theta)) \right]\cdot \mathcal{N}(\theta| 0, \rho)

    .. note::

        This likelihood is constrained by

        .. math::

            n^i_{bkg} + \theta^i\sigma_{eff}^i(\theta) \geq 0

    Args:
        signal_yields (``np.ndarray``): signal yields
        background_yields (``np.ndarray``): background yields
        data (``np.ndarray``): observations
        correlation_matrix (``np.ndarray``): correlations between regions
        absolute_uncertainty_envelops (``List[Tuple[float, float]]``): upper and lower uncertainty
          envelops for each background yield.
        signal_uncertainty_configuration (``Dict[Text, Any]]``, default ``None``): Configuration
          input for signal uncertainties

          * absolute_uncertainties (``List[float]``): Absolute uncertainties for the signal
          * absolute_uncertainty_envelops (``List[Tuple[float, float]]``): upper and lower
              uncertainty envelops
          * correlation_matrix (``List[List[float]]``): Correlation matrix
          * third_moments (``List[float]``): diagonal elemetns of the third moment
    """

    name: Text = "default_pdf.effective_sigma"
    """Name of the backend"""
    version: Text = DefaultPDFBase.version
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = DefaultPDFBase.spey_requires
    """Spey version required for the backend"""
    doi: List[Text] = ["10.1142/9781860948985_0013"]
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
        signal_uncertainty_configuration: Optional[Dict[Text, Any]] = None,
    ):
        assert len(absolute_uncertainty_envelops) == len(
            background_yields
        ), "Dimensionality of the uncertainty envelops does not match to the number of regions."
        assert len(correlation_matrix) == len(
            background_yields
        ), "Dimensionality of the correlation matrix does not match to the number of regions."

        sigma_plus, sigma_minus = [], []
        for upper, lower in absolute_uncertainty_envelops:
            sigma_plus.append(abs(upper))
            sigma_minus.append(abs(lower))
        sigma_plus = np.array(sigma_plus)
        sigma_minus = np.array(sigma_minus)
        correlation_matrix = np.array(correlation_matrix)

        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            signal_uncertainty_configuration=signal_uncertainty_configuration,
        )

        self._constraint_model: ConstraintModel = ConstraintModel(
            [
                {
                    "distribution_type": "multivariatenormal",
                    "args": [np.zeros(len(self.data)), correlation_matrix],
                    "kwargs": {"domain": slice(1, None)},
                }
            ]
            + self.signal_uncertainty_configuration.get("constraint", [])
        )

        A = self.background_yields

        # arXiv:pyhsics/0406120 eq. 18-19
        def effective_sigma(pars: np.ndarray) -> np.ndarray:
            """Compute effective sigma"""
            # clip from 1e-10 to avoid negative or zero values
            # this allows more numeric stability
            return np.sqrt(
                np.clip(
                    sigma_plus * sigma_minus
                    + (sigma_plus - sigma_minus) * (pars[slice(1, len(A) + 1)] - A),
                    1e-10,
                    None,
                )
            )

        def lam(pars: np.ndarray) -> np.ndarray:
            """Compute lambda for Main model"""
            return (
                A
                + effective_sigma(pars) * pars[slice(1, len(A) + 1)]
                + pars[0] * self.signal_yields
            )

        def constraint(pars: np.ndarray) -> np.ndarray:
            """Compute the constraint term"""
            return A + effective_sigma(pars) * pars[slice(1, len(A) + 1)]

        jac_constr = jacobian(constraint)
        self.constraints.append(
            NonlinearConstraint(constraint, 0.0, np.inf, jac=jac_constr)
        )
        if self.signal_uncertainty_configuration.get("lambda", None) is not None:
            signal_lambda = self.signal_uncertainty_configuration["lambda"]

            def poiss_lamb(pars: np.ndarray) -> np.ndarray:
                """combined lambda expression"""
                return lam(pars) + signal_lambda(pars)

        else:
            poiss_lamb = lam

        self._main_model = MainModel(poiss_lamb)
