"""Simplified Likelihood Interface"""

from typing import Optional, Text, Callable, List, Union, Tuple, Any
from autograd import numpy as np

from spey.optimizer import fit
from spey.base import BackendBase
from spey.base import ModelConfig
from spey.utils import ExpectationType
from spey._version import __version__
from .sldata import SLData, expansion_output
from .operators import logpdf, hessian_logpdf_func, objective_wrapper
from .sampler import sample_generator
from .distributions import MultivariateNormal, Poisson, Normal


def __dir__():
    return []


# pylint: disable=E1101


class SimplifiedLikelihoodBase(BackendBase):
    """
    Simplified Likelihood Interface.

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

        delta_sys (``float``, default ``0.0``): systematic uncertainty on signal.
        third_moment (``np.ndarray``, default ``None``): third moment for skewed gaussian.
          See eqs. 3.10, 3.11, 3.12, 3.13 in :xref:`1809.05548` for details.

    .. note::

        To enable a differentiable statistical model, all inputs are wrapped with
        :func:`autograd.numpy.array` function.
    """

    name: Text = "simplified_likelihoods.base"
    """Name of the backend"""
    version: Text = __version__
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = __version__
    """Spey version required for the backend"""
    doi: List[Text] = ["10.1007/JHEP04(2019)064"]
    """Citable DOI for the backend"""
    arXiv: List[Text] = ["1809.05548"]
    """arXiv reference for the backend"""

    __slots__ = ["_model", "_third_moment_expansion"]

    def __init__(
        self,
        signal_yields: np.ndarray,
        background_yields: np.ndarray,
        data: np.ndarray,
        covariance_matrix: np.ndarray,
        delta_sys: float = 0.0,
        third_moment: Optional[np.ndarray] = None,
    ):
        self._model = SLData(
            observed=np.array(data, dtype=np.float64),
            signal=np.array(signal_yields, dtype=np.float64),
            background=np.array(background_yields, dtype=np.float64),
            covariance=np.array(covariance_matrix, dtype=np.float64)
            if not callable(covariance_matrix) and covariance_matrix is not None
            else covariance_matrix,
            delta_sys=delta_sys,
            third_moment=None
            if third_moment is None
            else np.array(third_moment, dtype=np.float64),
            name="sl_model",
        )
        self._third_moment_expansion: Optional[expansion_output] = None
        self._gaussian = None

    @property
    def gaussian(self):
        """Gaussian distribution"""
        if self._gaussian is None:
            self._gaussian = MultivariateNormal(
                np.zeros(len(self.model.observed)), self.third_moment_expansion.V
            )
        return self._gaussian

    @property
    def model(self) -> SLData:
        """
        Accessor to the model container.

        Returns:
            ~spey.backends.simplifiedlikelihood_backend.sldata.SLData:
            Data container object that inherits :obj:`~spey.DataBase`.
        """
        return self._model

    @property
    def is_alive(self) -> bool:
        """Returns True if at least one bin has non-zero signal yield."""
        return self.model.isAlive

    @property
    def third_moment_expansion(self) -> expansion_output:
        """Get third moment expansion"""
        if self._third_moment_expansion is None:
            self._third_moment_expansion = self.model.compute_expansion()
        return self._third_moment_expansion

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
        return self.model.config(
            allow_negative_signal=allow_negative_signal, poi_upper_bound=poi_upper_bound
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
        current_model: SLData = (
            self.model
            if expected != ExpectationType.apriori
            else self.model.expected_dataset
        )

        return objective_wrapper(
            signal=current_model.signal,
            background=current_model.background,
            data=data if data is not None else current_model.observed,
            third_moment_expansion=self.third_moment_expansion,
            gaussian=self.gaussian,
            do_grad=do_grad,
        )

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
        current_model: SLData = (
            self.model
            if expected != ExpectationType.apriori
            else self.model.expected_dataset
        )
        return lambda pars: logpdf(
            pars=pars,
            signal=current_model.signal,
            background=current_model.background,
            observed=current_model.observed if data is None else data,
            third_moment_expansion=self.third_moment_expansion,
            gaussian=self.gaussian,
        )

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
        current_model: SLData = (
            self.model
            if expected != ExpectationType.apriori
            else self.model.expected_dataset
        )

        hess = hessian_logpdf_func(
            current_model.signal,
            current_model.background,
            self.third_moment_expansion,
            gaussian=self.gaussian,
        )

        return lambda pars: hess(pars, data or current_model.observed)

    def get_sampler(self, pars: np.ndarray) -> Callable[[int], np.ndarray]:
        r"""
        Retreives the function to sample from.

        Args:
            pars (``np.ndarray``): fit parameters (:math:`\mu` and :math:`\theta`)

        Returns:
            ``Callable[[int], np.ndarray]``:
            Function that takes ``number_of_samples`` as input and draws as many samples
            from the statistical model.
        """
        return sample_generator(
            pars=pars,
            signal=self.model.signal,
            background=self.model.background,
            third_moment_expansion=self.third_moment_expansion,
        )

    def expected_data(self, pars: List[float]) -> List[float]:
        r"""
        Compute the expected value of the statistical model

        Args:
            pars (``List[float]``): nuisance, :math:`\theta` and parameter of interest,
              :math:`\mu`.

        Returns:
            ``List[float]``:
            Expected data of the statistical model
        """
        return self.model.background + pars[1:]

    def generate_asimov_data(
        self,
        poi_asimov: float = 0.0,
        expected: ExpectationType = ExpectationType.observed,
        init_pars: Optional[List[float]] = None,
        par_bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Backend specific method to generate Asimov data.

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

            init_pars (``List[float]``, default ``None``): initial parameters for the optimiser
            par_bounds (``List[Tuple[float, float]]``, default ``None``): parameter bounds for
              the optimiser.
            kwargs: keyword arguments for the optimiser.

        Returns:
            ``np.ndarray``:
            Asimov data.
        """
        model: SLData = (
            self.model
            if expected != ExpectationType.apriori
            else self.model.expected_dataset
        )

        # Do not allow asimov data to be negative!
        par_bounds = kwargs.get("par_bounds", None)
        if par_bounds is None:
            par_bounds = [(0.0, 1.0)] + [
                (-1 * (bkg + sig * poi_asimov), 100.0)
                for sig, bkg in zip(model.signal, model.background)
            ]

        func = objective_wrapper(
            signal=model.signal,
            background=model.background,
            data=model.observed,
            third_moment_expansion=self.third_moment_expansion,
            gaussian=self.gaussian,
            do_grad=True,
        )

        _, fit_pars = fit(
            func=func,
            model_configuration=model.config(),
            do_grad=True,
            fixed_poi_value=poi_asimov,
            initial_parameters=init_pars,
            bounds=par_bounds,
            **kwargs,
        )

        return model.background + fit_pars[1:]


class UncorrelatedBackground(SimplifiedLikelihoodBase):
    r"""
    Simplified likelihood interface with uncorrelated regions.
    This simple backend is designed to handle single region statistical models or
    multi-region statistical models with uncorrelated regions. Inputs has to be given
    as list of ``NumPy`` array where each input should include same number of regions.
    It assumes absolute uncertainties on the background sample e.g. for a background
    sample yield reported as :math:`3.1\pm0.5` the background yield is ``3.1`` and the
    absolute uncertainty is ``0.5``. It forms a combination of normal and poisson
    distributions from the input data where the log-probability is computed as sum of
    all normal and poisson distributions.

    Args:
        signal_yields (``np.ndarray``): signal yields
        background_yields (``np.ndarray``): background yields
        data (``np.ndarray``): observations
        absolute_uncertainties (``np.ndarray``): absolute uncertainties on the background

    .. note::

        Each input should have the same dimensionality, i.e. if ``data`` has three regions,
        ``signal_yields``, ``background_yields`` and ``absolute_uncertainties`` inputs should
        have three regions as well.
    """

    name: Text = "simplified_likelihoods.uncorrelated_background"
    """Name of the backend"""
    version: Text = __version__
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = __version__
    """Spey version required for the backend"""
    doi: List[Text] = SimplifiedLikelihoodBase.doi
    """Citable DOI for the backend"""
    arXiv: List[Text] = SimplifiedLikelihoodBase.arXiv
    """arXiv reference for the backend"""

    def __init__(
        self,
        signal_yields: np.ndarray,
        background_yields: np.ndarray,
        data: np.ndarray,
        absolute_uncertainties: np.ndarray,
    ):
        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            covariance_matrix=None,
        )

        self._gaussian = Normal(
            np.zeros(len(self.model.observed)), absolute_uncertainties
        )


class SimplifiedLikelihoods(SimplifiedLikelihoodBase):
    """
    Simplified likelihoods for correlated multi-region statistical models.

    Args:
        signal_yields (``np.ndarray``): signal yields
        background_yields (``np.ndarray``): background yields
        data (``np.ndarray``): observations
        covariance_matrix (``np.ndarray``): covariance matrix (square matrix)

    .. note::

        Each input should have the same dimensionality, i.e. if ``data`` has three regions,
        ``signal_yields`` and ``background_yields`` inputs should have three regions as well.
        Additionally ``covariance_matrix`` is expected to be square matrix, thus for a three
        region statistical model it is expected to be 3x3 matrix.
    """

    name: Text = "simplified_likelihoods"
    """Name of the backend"""
    version: Text = __version__
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = __version__
    """Spey version required for the backend"""
    doi: List[Text] = SimplifiedLikelihoodBase.doi
    """Citable DOI for the backend"""
    arXiv: List[Text] = SimplifiedLikelihoodBase.arXiv
    """arXiv reference for the backend"""

    def __init__(
        self,
        signal_yields: np.ndarray,
        background_yields: np.ndarray,
        data: np.ndarray,
        covariance_matrix: np.ndarray,
    ):
        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            covariance_matrix=covariance_matrix,
        )


class ThirdMomentExpansion(SimplifiedLikelihoodBase):
    """
    Simplified likelihood interface with third moment expansion.

    Args:
        signal_yields (``np.ndarray``): signal yields
        background_yields (``np.ndarray``): background yields
        data (``np.ndarray``): observations
        covariance_matrix (``np.ndarray``): covariance matrix (square matrix)
        third_moment (``np.ndarray``): third moment for each region.

    .. note::

        Each input should have the same dimensionality, i.e. if ``data`` has three regions,
        ``signal_yields`` and ``background_yields`` inputs should have three regions as well.
        Additionally ``covariance_matrix`` is expected to be square matrix, thus for a three
        region statistical model it is expected to be 3x3 matrix. Following these,
        ``third_moment`` should also have three inputs.
    """

    name: Text = "simplified_likelihoods.third_moment_expansion"
    """Name of the backend"""
    version: Text = __version__
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = __version__
    """Spey version required for the backend"""
    doi: List[Text] = SimplifiedLikelihoodBase.doi
    """Citable DOI for the backend"""
    arXiv: List[Text] = SimplifiedLikelihoodBase.arXiv
    """arXiv reference for the backend"""

    def __init__(
        self,
        signal_yields: np.ndarray,
        background_yields: np.ndarray,
        data: np.ndarray,
        covariance_matrix: np.ndarray,
        third_moment: np.ndarray,
    ):

        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            covariance_matrix=covariance_matrix,
            third_moment=third_moment,
        )


class VariableGaussian(SimplifiedLikelihoodBase):
    r"""
    Simplified likelihood interface with variable Gaussian. Simplified likelihood approach
    relies on combination of Poissonian and Multivariate Normal distributions where correlations between
    regions determines the behaviour of the Gaussian portion of the likelihood.
    The variable Gaussian approach captures asymmetric uncertainties on the background yields by
    modifying the covariance matrix, :math:`\sigma`, as follows

    .. math::

        \Sigma(\mu) = diag\left(\sqrt{\sigma^+_i\sigma^-_i + (\sigma^+_i - \sigma^-_i)(\theta_i - \hat{\theta}_i)}\right)

        \sigma = \Sigma(\mu)\rho\Sigma(\mu)

    where :math:`\rho` is the correlation matrix, :math:`\theta` are the nuisance parameters and
    :math:`\hat\theta` are the best fit parameters. :math:`\sigma^\pm` are the upper and lower absolute
    uncertainty envelopes per bin.

    Args:
        signal_yields (``np.ndarray``): signal yields
        background_yields (``np.ndarray``): background yields
        data (``np.ndarray``): observations
        correlation_matrix (``np.ndarray``): correlations between regions
        absolute_uncertainty_envelops (``List[Tuple[float, float]]``): upper and lower uncertainty
          envelops for each background yield.
        best_fit_values (``List[float]``): bestfit values for the covariance matrix computation given
          as :math:`\hat\theta`.

    .. note::

        Each input should have the same dimensionality, i.e. if ``data`` has three regions,
        ``signal_yields`` and ``background_yields`` inputs should have three regions as well.
        Additionally ``covariance_matrix`` is expected to be square matrix, thus for a three
        region statistical model it is expected to be 3x3 matrix. Following these,
        ``third_moment`` should also have three inputs.
    """

    name: Text = "simplified_likelihoods.variable_gaussian"
    """Name of the backend"""
    version: Text = __version__
    """Version of the backend"""
    author: Text = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: Text = __version__
    """Spey version required for the backend"""
    doi: List[Text] = SimplifiedLikelihoodBase.doi + ["10.1142/9781860948985_0013"]
    """Citable DOI for the backend"""
    arXiv: List[Text] = SimplifiedLikelihoodBase.arXiv + ["physics/0406120"]
    """arXiv reference for the backend"""

    def __init__(
        self,
        signal_yields: np.ndarray,
        background_yields: np.ndarray,
        data: np.ndarray,
        correlation_matrix: np.ndarray,
        absolute_uncertainty_envelops: List[Tuple[float, float]],
        best_fit_values: List[float],
    ):
        assert len(absolute_uncertainty_envelops) == len(
            background_yields
        ), "Dimensionality of the uncertainty envelops does not match to the number of regions."
        assert correlation_matrix.shape[0] == len(
            background_yields
        ), "Dimensionality of the correlation matrix does not match to the number of regions."
        assert len(best_fit_values) == len(
            background_yields
        ), "Dimensionality of best fit values does not match with the number of regions."

        sigma_plus, sigma_minus = [], []
        for upper, lower in absolute_uncertainty_envelops:
            sigma_plus.append(upper)
            sigma_minus.append(lower)
        sigma_plus = np.array(sigma_plus)
        sigma_minus = np.array(sigma_minus)
        best_fit_values = np.array(best_fit_values)
        correlation_matrix = np.array(correlation_matrix)

        def covariance_matrix(nuisance_parameters: np.ndarray) -> np.ndarray:
            """Compute covariance matrix using variable gaussian formulation"""
            sigma = np.diag(
                np.sqrt(
                    sigma_plus * sigma_minus
                    + (sigma_plus - sigma_minus) * (nuisance_parameters - best_fit_values)
                )
            )
            return sigma @ correlation_matrix @ sigma

        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            covariance_matrix=covariance_matrix,
        )

        self._gaussian = MultivariateNormal(
            np.zeros(len(self.model.observed)), covariance_matrix
        )
