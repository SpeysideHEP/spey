"""This file contains basic likelihood implementations"""

from typing import Callable, List, Optional, Tuple, Union

from autograd import hessian, jacobian
from autograd import numpy as np
from autograd import value_and_grad
from scipy.optimize import NonlinearConstraint

from spey._version import __version__
from spey.backends.distributions import MainModel
from spey.base import BackendBase, ModelConfig
from spey.system.exceptions import InvalidInput
from spey.utils import ExpectationType

# pylint: disable=E1101,E1120,W0613


class SimplePDFBase(BackendBase):
    """Base structure for simple PDFs"""

    name: str = "__simplepdf_base__"
    """Name of the backend"""
    version: str = __version__
    """Version of the backend"""
    author: str = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: str = __version__
    """Spey version required for the backend"""

    __slots__ = [
        "data",
        "signal_yields",
        "background_yields",
        "_main_model",
        "_main_kwargs",
        "_config",
    ]

    def __init__(
        self,
        signal_yields: Union[List[float], Callable[[np.ndarray], np.ndarray]],
        background_yields: List[float],
        data: List[int],
    ):
        self.data = np.array(data, dtype=np.float64)
        if callable(signal_yields):
            self.signal_yields = signal_yields
        else:
            self.signal_yields = np.array(signal_yields, dtype=np.float64)
        self.background_yields = np.array(background_yields, dtype=np.float64)
        self._main_model = None
        """main model"""
        self._main_kwargs = {}
        """Keyword arguments for main model"""
        self.constraints = []
        """Constraints to be used during optimisation process"""

        minimum_poi = -np.inf
        if self.is_alive and not callable(self.signal_yields):
            sig = self.signal_yields
            minimum_poi = -np.min(self.background_yields[sig > 0.0] / sig[sig > 0.0])

        self._config = ModelConfig(
            poi_index=0,
            minimum_poi=minimum_poi,
            suggested_init=[1.0],
            suggested_bounds=[(minimum_poi, 10)],
        )

    @property
    def is_alive(self) -> bool:
        """
        Returns True if at least one bin has non-zero signal yield.

        .. versionchanged:: 0.2.7
            When ``signal_yields`` is a callable, always returns ``True`` since the
            actual yields are not known until evaluation time.
        """
        if callable(self.signal_yields):
            return True
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

        .. versionchanged:: 0.2.7
            When the model has extra signal or nuisance parameters (e.g. added by
            subclasses via :attr:`n_signal_parameters`), those additional bounds and
            :attr:`~spey.base.ModelConfig.parameter_names` are now correctly preserved
            when ``allow_negative_signal`` is ``False`` or ``poi_upper_bound`` differs
            from the default.
        """
        if allow_negative_signal and poi_upper_bound == 10.0:
            return self._config

        extra_bounds = self._config.suggested_bounds[1:]
        return ModelConfig(
            self._config.poi_index,
            self._config.minimum_poi,
            self._config.suggested_init,
            [(0, poi_upper_bound)] + extra_bounds,
            parameter_names=self._config.parameter_names,
        )

    @property
    def main_model(self) -> MainModel:
        r"""
        Retrieve the main model distribution.

        .. versionchanged:: 0.2.7
            The internal ``lam`` function now supports a callable ``signal_yields``.
            When ``signal_yields`` is callable, it is evaluated as
            ``signal_yields(pars[1:])`` so that the extra signal parameters (beyond
            :math:`\mu` at index 0) are forwarded to the function at each evaluation.
        """
        if self._main_model is None:

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
                sig = (
                    self.signal_yields(pars[1:])
                    if callable(self.signal_yields)
                    else self.signal_yields
                )
                return pars[0] * sig + self.background_yields

            self._main_model = MainModel(lam, **self._main_kwargs)

        return self._main_model

    def get_objective_function(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.ndarray] = None,
        do_grad: bool = True,
    ) -> Callable[[np.ndarray], Union[Tuple[float, np.ndarray], float]]:
        r"""
        Objective function i.e. negative log-likelihood, :math:`-\log\mathcal{L}(\mu, \theta)`

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

        def negative_loglikelihood(pars: np.ndarray) -> np.ndarray:
            """Compute twice negative log-likelihood"""
            return -self.main_model.log_prob(pars, data)

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

        return lambda pars: self.main_model.log_prob(pars, data)

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

        def log_prob(pars: np.ndarray) -> np.ndarray:
            """Compute log-probability"""
            return self.main_model.log_prob(pars, data)

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

        def sampler(sample_size: int, *args, **kwargs) -> np.ndarray:
            """
            Fucntion to generate samples.

            Args:
                sample_size (``int``): number of samples to be generated.

            Returns:
                ``np.ndarray``:
                generated samples
            """
            return self.main_model.sample(pars, sample_size)

        return sampler

    def expected_data(self, pars: List[float], **kwargs) -> List[float]:
        r"""
        Compute the expected value of the statistical model

        Args:
            pars (``List[float]``): nuisance, :math:`\theta` and parameter of interest,

        Returns:
            ``List[float]``:
            Expected data of the statistical model
        """
        return self.main_model.expected_data(pars)


class Poisson(SimplePDFBase):
    r"""
    Poisson distribution without uncertainty implementation.

    .. math::

        \mathcal{L}(\mu) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + n_b^i)

    where :math:`n_{s,b}` are signal and background yields and :math:`n` are the observations.

    Args:
        signal_yields (``List[float]``): signal yields
        background_yields (``List[float]``): background yields
        data (``List[int]``): data
        absolute_uncertainties (``List[float]``, optional): background uncertainties to be added.
          :math:`\mu n_s^i + n_b^i + \theta^i\sigma^i`.
    """

    name: str = "default.poisson"
    """Name of the backend"""
    version: str = SimplePDFBase.version
    """Version of the backend"""
    author: str = SimplePDFBase.author
    """Author of the backend"""
    spey_requires: str = SimplePDFBase.spey_requires
    """Spey version required for the backend"""

    def __init__(
        self,
        signal_yields: List[float],
        background_yields: List[float],
        data: List[int],
        absolute_uncertainties: Optional[List[float]] = None,
    ):
        super().__init__(
            signal_yields=signal_yields, background_yields=background_yields, data=data
        )

        if absolute_uncertainties is not None:
            self.absolute_uncertainties = np.array(absolute_uncertainties)
            assert len(self.absolute_uncertainties) == len(
                self.data
            ), "Invalid input dimension."

            def lam(pars: np.ndarray) -> np.ndarray:
                return (
                    pars[0] * self.signal_yields
                    + self.background_yields
                    + pars[slice(1, len(self.data) + 1)] * self.absolute_uncertainties
                )

            self._main_model = MainModel(lam)

            self._config = ModelConfig(
                poi_index=0,
                minimum_poi=self._config.minimum_poi,
                suggested_init=[1.0] * (len(self.data) + 1),
                suggested_bounds=[(self._config.minimum_poi, 10)]
                + [(None, None)] * len(self.data),
            )

            def constraint(pars: np.ndarray) -> np.ndarray:
                """Compute the constraint term"""
                return (
                    self.background_yields
                    + pars[slice(1, len(self.data) + 1)] * self.background_yields
                )

            jac_constr = jacobian(constraint)

            self.constraints.append(
                NonlinearConstraint(constraint, 0.0, np.inf, jac=jac_constr)
            )


class Gaussian(SimplePDFBase):
    r"""
    Gaussian distribution for uncorrelated likelihoods.

    .. math::

        \mathcal{L}(\mu) = \prod_{i\in{\rm bins}} \frac{1}{\sigma^i \sqrt{2\pi}}
        \exp\left[-\frac{1}{2} \left(\frac{\mu n_s^i + n_b^i - n^i}{\sigma^i} \right)^2 \right]

    where :math:`n_{s,b}` are signal and background yields and :math:`n` are the observations.

    .. versionadded:: 0.1.9

    Args:
        signal_yields (``List[float]``): signal yields
        background_yields (``List[float]``): background yields
        data (``List[int]``): data
        absolute_uncertainties (``List[float]``): absolute uncertainties on the background
    """

    name: str = "default.normal"
    """Name of the backend"""
    version: str = SimplePDFBase.version
    """Version of the backend"""
    author: str = SimplePDFBase.author
    """Author of the backend"""
    spey_requires: str = SimplePDFBase.spey_requires
    """Spey version required for the backend"""

    __slots__ = ["absolute_uncertainties"]

    def __init__(
        self,
        signal_yields: List[float],
        background_yields: List[float],
        data: List[int],
        absolute_uncertainties: List[float],
    ):
        super().__init__(
            signal_yields=signal_yields, background_yields=background_yields, data=data
        )
        self.absolute_uncertainties = np.array(absolute_uncertainties, dtype=np.float64)
        """absolute uncertainties on the background"""
        self._main_kwargs = {"cov": self.absolute_uncertainties, "pdf_type": "gauss"}


class MultivariateNormal(SimplePDFBase):
    r"""
    Multivariate Gaussian distribution.

    .. math::

        \mathcal{L}(\mu) = \frac{1}{\sqrt{(2\pi)^k {\rm det}[\Sigma] }}
        \exp\left[-\frac{1}{2} (\mu n_s + n_b - n)\Sigma^{-1} (\mu n_s + n_b - n)^T \right]

    where :math:`n_{s,b}` are signal and background yields and :math:`n` are the observations.

    .. versionadded:: 0.1.9

    ``covariance_matrix`` can also take callable function as an input where
    function takes nuisance parameters as inputs and return a new covariance matrix as output.

    **Example — callable covariance matrix:**

    .. code:: python3

        >>> import spey
        >>> import numpy as np

        >>> signal_yields = np.array([12.0, 15.0])
        >>> background_yields = np.array([50.0, 48.0])
        >>> data = np.array([36., 33.])
        >>> covariance_matrix = np.array([[144.0, 13.0], [25.0, 256.0]])
        >>> covariance_signal = np.array([[5.0, 1.0], [2.0, 3.0]])

        >>> def cov_matrix(pars: np.ndarray) -> np.ndarray:
        >>>     return covariance_matrix + covariance_signal * pars[0]**2

        >>> pdf_wrapper = spey.get_backend('default.multivariate_normal')
        >>> model = pdf_wrapper(
        ...     signal_yields=signal_yields,
        ...     background_yields=background_yields,
        ...     data=data,
        ...     covariance_matrix=cov_matrix,
        ... )

    **Example — callable signal yields with additional parameters:**

    ``signal_yields`` can be a callable that accepts extra parameters beyond :math:`\mu`
    and returns the per-bin signal yields as a ``np.ndarray``. The number of extra
    parameters must be declared via ``n_signal_parameters`` so that the optimiser can
    construct the correct parameter vector. The first element of the parameter vector is
    always :math:`\mu`; the remaining elements (``pars[1:]``) are forwarded to the
    callable at each function evaluation.

    .. code:: python3

        >>> import spey
        >>> import numpy as np
        >>> from spey.backends.default_pdf.simple_pdf import MultivariateNormal

        >>> background_yields = np.array([50.0, 48.0])
        >>> data = np.array([36., 33.])
        >>> covariance_matrix = np.array([[144.0, 13.0], [25.0, 256.0]])
        >>> base_signal = np.array([12.0, 15.0])

        >>> # signal_yields is a function of one extra parameter (signal_par_0)
        >>> def signal_yields(extra_pars: np.ndarray) -> np.ndarray:
        ...     return base_signal * (1.0 + extra_pars[0])

        >>> pdf_wrapper = spey.get_backend('default.multivariate_normal')
        >>> model = pdf_wrapper(
        ...     signal_yields=signal_yields,
        ...     background_yields=background_yields,
        ...     data=data,
        ...     covariance_matrix=covariance_matrix,
        ...     n_signal_parameters=1,
        ... )

        >>> muhat, nll = model.maximize_likelihood()

        The resulting :class:`~spey.base.ModelConfig` will contain two parameters:
        ``["mu", "signal_par_0"]``.

    .. versionchanged:: 0.2.6
        The ability to input a callable covariance matrix has been added.

    .. versionchanged:: 0.2.7
        ``signal_yields`` now accepts a callable with signature
        ``(extra_pars: np.ndarray) -> np.ndarray`` in addition to a plain array.
        The companion argument ``n_signal_parameters`` (default ``0``) declares how many
        extra parameters the callable expects; they are appended to the parameter vector
        after :math:`\mu` and their names are automatically set to
        ``signal_par_0``, ``signal_par_1``, …

    Args:
        signal_yields (``List[float] | Callable[[np.ndarray], np.ndarray]``): signal yields
          per bin, or a callable that takes the extra signal parameters (``pars[1:]``) and
          returns the signal yields as a ``np.ndarray``.
        background_yields (``List[float]``): background yields per bin.
        data (``List[int]``): observed data per bin.
        covariance_matrix (``List[List[float]] | callable``): covariance matrix (square matrix),
          or a callable that takes the full parameter vector and returns a covariance matrix.

          * If you have a correlation matrix and absolute uncertainties please use
            :func:`~spey.helper_functions.correlation_to_covariance`

        n_signal_parameters (``int``, default ``0``): number of additional free parameters
          to pass to a callable ``signal_yields``.  Has no effect when ``signal_yields``
          is a plain array.  When greater than zero the optimiser parameter vector is
          extended to ``[mu, signal_par_0, ..., signal_par_{n-1}]``.

    """

    name: str = "default.multivariate_normal"
    """Name of the backend"""
    version: str = SimplePDFBase.version
    """Version of the backend"""
    author: str = SimplePDFBase.author
    """Author of the backend"""
    spey_requires: str = SimplePDFBase.spey_requires
    """Spey version required for the backend"""

    __slots__ = ["covariance_matrix", "n_signal_parameters"]

    def __init__(
        self,
        signal_yields: Union[List[float], Callable[[np.ndarray], np.ndarray]],
        background_yields: List[float],
        data: List[int],
        covariance_matrix: Union[List[List[float]], callable],
        n_signal_parameters: int = 0,
    ):
        super().__init__(
            signal_yields=signal_yields, background_yields=background_yields, data=data
        )
        self.n_signal_parameters = n_signal_parameters

        if not callable(covariance_matrix):
            self.covariance_matrix = np.array(covariance_matrix, dtype=np.float64)
            if (
                self.covariance_matrix.shape[0] != len(self.background_yields)
                and len(self.covariance_matrix.shape) == 2
            ):
                raise InvalidInput(
                    "Dimensionality of the covariance matrix should match to the background"
                )
        else:
            self.covariance_matrix = covariance_matrix

        self._main_kwargs = {
            "cov": self.covariance_matrix,
            "pdf_type": "multivariategauss",
        }

        if n_signal_parameters > 0:
            self._config = ModelConfig(
                poi_index=0,
                minimum_poi=self._config.minimum_poi,
                suggested_init=[1.0] + [1.0] * n_signal_parameters,
                suggested_bounds=[(self._config.minimum_poi, 10)]
                + [(None, None)] * n_signal_parameters,
                parameter_names=["mu"]
                + [f"signal_par_{i}" for i in range(n_signal_parameters)],
            )
