"""Simplified Likelihood Interface"""

from typing import Optional, Text, Callable, List, Union, Tuple, Any
from autograd import numpy as np
from autograd import grad, hessian

from spey.optimizer import fit
from spey.base import BackendBase
from spey.base import ModelConfig
from spey.utils import ExpectationType
from spey._version import __version__
from .sldata import SLData
from .distributions import MainModel, ConstraintModel
from .third_moment import third_moment_expansion


def __dir__():
    return []


# pylint: disable=E1101,E1120


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

    __slots__ = ["_model", "_main_model", "_constraint_model"]

    def __init__(
        self,
        signal_yields: np.ndarray,
        background_yields: np.ndarray,
        data: np.ndarray,
        covariance_matrix: Optional[
            Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]
        ] = None,
        third_moment: Optional[np.ndarray] = None,
    ):
        self._model = SLData(
            observed=np.array(data, dtype=np.float64),
            signal=np.array(signal_yields, dtype=np.float64),
            background=np.array(background_yields, dtype=np.float64),
            covariance_matrix=np.array(covariance_matrix, dtype=np.float64)
            if not callable(covariance_matrix) and covariance_matrix is not None
            else covariance_matrix,
            third_moment=None
            if third_moment is None
            else np.array(third_moment, dtype=np.float64),
            name="sl_model",
        )
        self._main_model = None
        self._constraint_model = None

    @property
    def constraint_model(self) -> ConstraintModel:
        """retreive constraint model distribution"""
        if self._constraint_model is None:
            self._constraint_model = ConstraintModel(
                "multivariatenormal",
                np.zeros(len(self.model.observed)),
                self.model.covariance_matrix,
            )
        return self._constraint_model

    @property
    def main_model(self) -> MainModel:
        """retreive the main model distribution"""
        if self._main_model is None:

            def lam(pars: np.ndarray) -> np.ndarray:
                """Compute lambda for Main model"""
                return np.clip(
                    self.model.background + pars[1:] + pars[0] * self.model.signal,
                    1e-5,
                    None,
                )

            self._main_model = MainModel(lam)

        return self._main_model

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

        data = data if data is not None else current_model.observed

        def twice_nll(pars: np.ndarray) -> np.ndarray:
            """Compute twice negative log-likelihood"""
            return -2.0 * (
                self.main_model.log_prob(pars, data[: len(current_model)])
                + self.constraint_model.log_prob(pars[1:])
            )

        if do_grad:
            grad_twice_nll = grad(twice_nll, argnum=0)
            return lambda pars: (twice_nll(pars), grad_twice_nll(pars))

        return twice_nll

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

        data = data if data is not None else current_model.observed

        return lambda pars: self.main_model.log_prob(
            pars, data[: len(current_model)]
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
        current_model: SLData = (
            self.model
            if expected != ExpectationType.apriori
            else self.model.expected_dataset
        )

        data = data if data is not None else current_model.observed

        def log_prob(pars: np.ndarray) -> np.ndarray:
            """Compute log-probability"""
            return self.main_model.log_prob(
                pars, data[: len(current_model)]
            ) + self.constraint_model.log_prob(pars[1:])

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

        _, fit_pars = fit(
            func=self.get_objective_function(expected=expected, do_grad=True),
            model_configuration=model.config(),
            do_grad=True,
            fixed_poi_value=poi_asimov,
            initial_parameters=init_pars,
            bounds=par_bounds,
            **kwargs,
        )

        return self.expected_data(fit_pars)


class UncorrelatedBackground(SimplifiedLikelihoodBase):
    r"""
    Simplified likelihood interface with uncorrelated regions.
    This simple backend is designed to handle single or multi-bin statistical models
    with uncorrelated regions. Inputs has to be given as list of ``NumPy`` array where
    each input should include same number of regions. It assumes absolute uncertainties
    on the background sample e.g. for a background sample yield reported as
    :math:`3.1\pm0.5` the background yield is ``3.1`` and the absolute uncertainty is
    ``0.5``. It forms a combination of normal and poisson distributions from the input
    data where the log-probability is computed as sum of all normal and poisson distributions.

    Args:
        signal_yields (``List[float]``): signal yields
        background_yields (``List[float]``): background yields
        data (``List[int]``): observations
        absolute_uncertainties (``List[float]``): absolute uncertainties on the background

    .. note::

        Each input should have the same dimensionality, i.e. if ``data`` has three regions,
        ``signal_yields``, ``background_yields`` and ``absolute_uncertainties`` inputs should
        have three regions as well.

    Example:

    .. code:: python3

        >>> import spey
        >>> stat_wrapper = spey.get_backend('simplified_likelihoods.uncorrelated_background')

        >>> data = [1, 3]
        >>> signal = [0.5, 2.0]
        >>> background = [2.0, 2.8]
        >>> background_unc = [1.1, 0.8]

        >>> stat_model = stat_wrapper(
        ...     signal, background, data, background_unc, analysis="multi-bin", xsection=0.123
        ... )
        >>> print("1-CLs : %.3f" % tuple(stat_model.exclusion_confidence_level()))
        >>> # 1-CLs : 0.702
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
        signal_yields: List[float],
        background_yields: List[float],
        data: List[int],
        absolute_uncertainties: List[float],
    ):
        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            covariance_matrix=None,
        )

        self._constraint_model: ConstraintModel = ConstraintModel(
            "normal", np.zeros(len(self.model.observed)), absolute_uncertainties
        )


class SimplifiedLikelihoods(SimplifiedLikelihoodBase):
    """
    Simplified likelihoods for correlated multi-region statistical models.
    Main simplified likelihood backend which uses a Multivariate Normal and
    a Poisson distributions to construct log-probability of the statistical
    model. The Multivariate Normal distribution is constructed by the help
    of a covariance matrix provided by the user which captures the
    uncertainties and background correlations between each histogram bin.
    This statistical model has been first proposed in :xref:`1809.05548`.

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

    Example:

    .. code:: python3

        >>> import spey

        >>> stat_wrapper = spey.get_backend('simplified_likelihoods')

        >>> signal_yields = [12.0, 11.0]
        >>> background_yields = [50.0, 52.0]
        >>> data = [51, 48]
        >>> covariance_matrix = [[3.,0.5], [0.6,7.]]

        >>> statistical_model = stat_wrapper(signal_yields,background_yields,data,covariance_matrix)
        >>> print(statistical_model.exclusion_confidence_level())
        >>> # [0.9734448420632104]
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

        background = np.array(background_yields)
        covariance = np.array(covariance_matrix)
        third_moments = np.array(third_moment)

        A, B, C, corr = third_moment_expansion(
            background, covariance, third_moments, True
        )

        super().__init__(
            signal_yields=signal_yields,
            background_yields=background,
            data=data,
            covariance_matrix=covariance,
            third_moment=third_moments,
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
            nI = A + B * pars[1:] + C * np.square(pars[1:])
            return np.clip(pars[0] * self.model.signal + nI, 1e-5, None)

        self._main_model = MainModel(lam)
        self._constraint_model = ConstraintModel(
            "multivariatenormal", np.zeros(corr.shape[0]), np.linalg.inv(corr)
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

        self._constraint_model: ConstraintModel = ConstraintModel(
            "multivariatenormal", np.zeros(len(self.model.observed)), covariance_matrix
        )
