r"""
Nuisance-Free (Simple) Likelihood Backends
===========================================

This module provides three backend classes that implement **nuisance-free** (or
trivially-marginalised) likelihoods in which the expected counts depend only on
the signal strength :math:`\mu` — no per-bin systematic nuisance parameters appear
in the optimisation.

All three share :class:`SimplePDFBase`, which stores signal and background yields,
constructs the minimum-POI bound, and provides the standard
:meth:`~spey.base.BackendBase.get_objective_function`,
:meth:`~spey.base.BackendBase.get_logpdf_func`, and
:meth:`~spey.base.BackendBase.get_hessian_logpdf_func` interfaces required by the
spey optimisation engine.

Available backends
------------------

``default.poisson`` — :class:`Poisson`
    Pure Poisson likelihood.  Each bin contributes independently:

    .. math::

        \mathcal{L}(\mu) = \prod_{i=1}^{N}
        \mathrm{Poiss}\!\left(n^{\rm obs}_i \,\big|\, \mu n^s_i + n^b_i\right).

    An optional ``absolute_uncertainties`` argument extends the model with
    unconstrained per-bin nuisance parameters:

    .. math::

        \mathcal{L}(\mu, \boldsymbol{\theta}) = \prod_{i=1}^{N}
        \mathrm{Poiss}\!\left(n^{\rm obs}_i \,\big|\,
        \mu n^s_i + n^b_i + \theta_i \sigma_i\right).

    .. autosummary::
        :toctree: ../_generated/

        Poisson

``default.normal`` — :class:`Gaussian`
    Product of independent Gaussians (uncorrelated):

    .. math::

        \mathcal{L}(\mu) = \prod_{i=1}^{N}
        \frac{1}{\sigma_i \sqrt{2\pi}}
        \exp\!\left[-\frac{(\mu n^s_i + n^b_i - n^{\rm obs}_i)^2}
                        {2\sigma_i^2}\right].

    This is appropriate when the expected counts are large enough that Poisson
    statistics can be well approximated by a Gaussian.

    .. autosummary::
        :toctree: ../_generated/

        Gaussian

``default.multivariate_normal`` — :class:`MultivariateNormal`
    Multivariate Gaussian with an arbitrary (possibly :math:`\mu`-dependent)
    covariance matrix:

    .. math::

        \mathcal{L}(\mu) = \frac{1}{\sqrt{(2\pi)^N \det\Sigma}}
        \exp\!\left[-\tfrac{1}{2}
        (\boldsymbol{\lambda}(\mu) - \mathbf{n}^{\rm obs})^\top
        \Sigma^{-1}
        (\boldsymbol{\lambda}(\mu) - \mathbf{n}^{\rm obs})\right],

    where :math:`\lambda_i(\mu) = \mu n^s_i + n^b_i`.  The covariance matrix
    :math:`\Sigma` may be a constant array or a callable of the full parameter
    vector, enabling :math:`\mu`-dependent uncertainties.

    .. autosummary::
        :toctree: ../_generated/

        MultivariateNormal

Parameter layout
----------------

All simple backends use a minimal parameter vector::

    pars = [μ, (signal_par_0, signal_par_1, …)]

Index 0 is always :math:`\mu`.  Additional signal parameters are only present when
:class:`MultivariateNormal` is initialised with a callable ``signal_yields`` and
``n_signal_parameters > 0``.

Minimum POI bound
-----------------

The lower bound on :math:`\mu` is set to the most negative signal strength for which
no bin has negative expected yield,

.. math::

    \mu_{\min} = -\min_{i:\,n^s_i > 0} \frac{n^b_i}{n^s_i},

matching the convention used in the other default PDF backends.  When
``signal_yields`` is a callable the bound is set to :math:`-\infty`.
"""

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
    r"""
    Abstract base class for nuisance-free (simple) PDF backends.

    Subclasses implement statistical models in which the expected bin counts
    depend only on the signal strength :math:`\mu`,

    .. math::

        \lambda_i(\mu) = \mu\, n^s_i + n^b_i,

    with no systematic nuisance parameters entering the fit (or, in the case of
    :class:`Poisson` with ``absolute_uncertainties``, unconstrained nuisances that
    are marginalised by the optimiser without an explicit constraint model).

    This class provides:

    * Common storage and :mod:`autograd`-compatible array initialisation.
    * Minimum-POI computation: :math:`\mu_{\min} = -\min_{i} n^b_i / n^s_i` over
      bins with :math:`n^s_i > 0`.
    * A lazily-initialised :attr:`main_model` property based on the
      :math:`\lambda(\mu)` function above.
    * Default implementations of :meth:`get_objective_function`,
      :meth:`get_logpdf_func`, :meth:`get_hessian_logpdf_func`,
      :meth:`get_sampler`, and :meth:`expected_data` that all delegate to
      :attr:`main_model`.

    Subclasses customise the model by overriding :attr:`main_model` (by setting
    ``self._main_model`` directly in ``__init__``) and by passing ``self._main_kwargs``
    to change the distribution type used by :class:`~spey.backends.distributions.MainModel`.
    """

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
        n_signal_parameters: int = 0,
        signal_parameter_bounds: Optional[
            List[Tuple[Optional[float], Optional[float]]]
        ] = None,
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
        self.n_signal_parameters = n_signal_parameters
        """Number of signal parameters"""

        minimum_poi = -np.inf
        if self.is_alive and not callable(self.signal_yields):
            sig = self.signal_yields
            minimum_poi = -np.min(self.background_yields[sig > 0.0] / sig[sig > 0.0])

        if signal_parameter_bounds is None:
            _sig_par_bounds = [(None, None)] * n_signal_parameters
        else:
            if len(signal_parameter_bounds) != n_signal_parameters:
                raise ValueError(
                    f"signal_parameter_bounds has {len(signal_parameter_bounds)} entries "
                    f"but n_signal_parameters is {n_signal_parameters}."
                )
            _sig_par_bounds = [
                b if b is not None else (None, None) for b in signal_parameter_bounds
            ]

        self._config = ModelConfig(
            poi_index=0,
            minimum_poi=minimum_poi,
            suggested_init=[1.0] + [1.0] * n_signal_parameters,
            suggested_bounds=[(minimum_poi, 10)] + _sig_par_bounds,
            parameter_names=["mu"]
            + [f"signal_par_{i}" for i in range(n_signal_parameters)],
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
        Main model distribution — Poisson (or Gaussian) term of the likelihood.

        For :class:`SimplePDFBase` the expected count in bin :math:`i` is simply

        .. math::

            \lambda_i(\mu) = \mu\, n^s_i + n^b_i,

        with no nuisance parameters.  When ``signal_yields`` is a callable, the
        extra signal parameters ``pars[1:]`` are forwarded to it at each evaluation:

        .. math::

            \lambda_i(\mu, \boldsymbol{\phi}) =
            \mu\, n^s_i(\boldsymbol{\phi}) + n^b_i,

        where :math:`\boldsymbol{\phi}` denotes the additional signal parameters.

        Subclasses may replace this model entirely by setting ``self._main_model``
        before the first access (e.g. :class:`Poisson` with uncertainty extension).

        .. versionchanged:: 0.2.7
            Callable ``signal_yields`` support added.

        Returns:
            :class:`~spey.backends.distributions.MainModel`:
            Lazily-initialised main model.
        """
        if self._main_model is None:

            def lam(pars: np.ndarray) -> np.ndarray:
                """
                Compute the per-bin expected count :math:`\\lambda_i(\\mu)`.

                Args:
                    pars (``np.ndarray``): Parameter vector; ``pars[0]`` is :math:`\\mu`
                      and ``pars[1:]`` are forwarded to a callable ``signal_yields``.

                Returns:
                    ``np.ndarray``:
                    Per-bin expected counts :math:`\\{\\lambda_i\\}`.
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
        Return the objective function :math:`-\ln\mathcal{L}(\mu)` used by the optimiser.

        Because there is no constraint model, the objective is simply the negative
        log-likelihood of the main model:

        .. math::

            -\ln\mathcal{L}(\mu) = -\ln\mathcal{L}_{\rm main}(\mu).

        When ``do_grad=True`` the returned callable is wrapped with
        :func:`autograd.value_and_grad` to supply exact gradients via automatic
        differentiation.

        Args:
            expected (:class:`~spey.ExpectationType`): Controls which dataset is used
              during fitting.

              * :obj:`~spey.ExpectationType.observed`: Real observed data (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Post-fit expected data.
              * :obj:`~spey.ExpectationType.apriori`: Pre-fit SM expectation
                (background yields used as pseudo-data).

            data (``np.ndarray``, default ``None``): Override the observed data.
              When ``None``, the array selected by ``expected`` is used.
            do_grad (``bool``, default ``True``): If ``True``, the returned callable
              yields ``(nll, grad_nll)``; if ``False`` it yields only ``nll``.

        Returns:
            ``Callable[[np.ndarray], float | Tuple[float, np.ndarray]]``:
            A function of the parameter vector :math:`(\mu, \ldots)` that returns
            the NLL and optionally its gradient.
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
        Return a callable that evaluates :math:`\ln\mathcal{L}(\mu)`.

        The log-likelihood is computed directly from :attr:`main_model` with no
        constraint term:

        .. math::

            \ln\mathcal{L}(\mu) = \ln\mathcal{L}_{\rm main}(\mu).

        This is used internally for the profile likelihood ratio and Hessian-based
        variance estimation.

        Args:
            expected (:class:`~spey.ExpectationType`): Dataset prescription.

              * :obj:`~spey.ExpectationType.observed`: Real observed data (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Post-fit expected data.
              * :obj:`~spey.ExpectationType.apriori`: Pre-fit SM expectation
                (background yields used as pseudo-data).

            data (``np.ndarray``, default ``None``): Override data.

        Returns:
            ``Callable[[np.ndarray], float]``:
            Function of the parameter vector :math:`(\mu, \ldots)` returning the
            scalar log-likelihood.
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
        Return a callable that evaluates the Hessian of :math:`\ln\mathcal{L}(\mu)`.

        The Hessian :math:`H_{ab} = \partial^2 \ln\mathcal{L} / \partial p_a \partial p_b`
        is computed via :func:`autograd.hessian`.  Its primary use is to estimate the
        variance on :math:`\hat\mu` through the Fisher information:

        .. math::

            \sigma_{\hat\mu}^2 \approx \left[H^{-1}\right]_{00},

        where index 0 corresponds to :math:`\mu`.  For purely Poissonian or Gaussian
        likelihoods with no nuisance parameters the Hessian is a :math:`1 \times 1`
        matrix.

        Args:
            expected (:class:`~spey.ExpectationType`): Dataset prescription.

              * :obj:`~spey.ExpectationType.observed`: Real observed data (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Post-fit expected data.
              * :obj:`~spey.ExpectationType.apriori`: Pre-fit SM expectation.

            data (``np.ndarray``, default ``None``): Override data.

        Returns:
            ``Callable[[np.ndarray], np.ndarray]``:
            Function of the parameter vector returning the
            :math:`(N_{\rm par} \times N_{\rm par})` Hessian matrix of
            :math:`\ln\mathcal{L}`.
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
        Return a callable that draws pseudo-data from the main model.

        Because there is no constraint model, all samples come from the Poisson (or
        Gaussian) main model evaluated at the expected yields
        :math:`\lambda(\mu) = \mu\, n^s + n^b`.

        Args:
            pars (``np.ndarray``): Fixed parameter vector :math:`(\mu, \ldots)` at
              which expected yields are evaluated before sampling.

        Returns:
            ``Callable[[int], np.ndarray]``:
            A function ``sampler(sample_size)`` that returns an array of shape
            ``(sample_size, N_bins)`` with generated pseudo-counts.
        """

        def sampler(sample_size: int, *args, **kwargs) -> np.ndarray:
            """
            Draw ``sample_size`` pseudo-experiments from the main model.

            Args:
                sample_size (``int``): Number of pseudo-experiments to generate.

            Returns:
                ``np.ndarray``:
                Array of shape ``(sample_size, N_bins)`` with the generated counts.
            """
            return self.main_model.sample(pars, sample_size)

        return sampler

    def expected_data(self, pars: List[float], **kwargs) -> List[float]:
        r"""
        Compute the expected data vector :math:`\{\lambda_i(\mu)\}` at the given
        parameter point.

        Args:
            pars (``List[float]``): Parameter vector :math:`(\mu, \ldots)`.

        Returns:
            ``List[float]``:
            Per-bin expected counts :math:`\lambda_i(\mu) = \mu n^s_i + n^b_i`.
        """
        return self.main_model.expected_data(pars)


class Poisson(SimplePDFBase):
    r"""
    Pure Poisson likelihood (``default.poisson``).

    The simplest possible statistical model: no systematic uncertainties, no
    constraint model.  The likelihood is a product of independent Poisson terms,

    .. math::

        \mathcal{L}(\mu) = \prod_{i=1}^{N}
        \mathrm{Poiss}\!\left(n^{\rm obs}_i \,\big|\, \mu n^s_i + n^b_i\right),

    where :math:`n^s_i`, :math:`n^b_i`, and :math:`n^{\rm obs}_i` are the signal
    yield, background yield, and observed count in bin :math:`i` respectively.

    **Optional background uncertainties**

    When ``absolute_uncertainties`` is provided, unconstrained per-bin nuisance
    parameters :math:`\theta_i` are added to the expected count:

    .. math::

        \mathcal{L}(\mu, \boldsymbol{\theta}) = \prod_{i=1}^{N}
        \mathrm{Poiss}\!\left(n^{\rm obs}_i \,\big|\,
        \mu n^s_i + n^b_i + \theta_i \sigma_i\right).

    .. note::

        These nuisance parameters are **unconstrained** (no Gaussian penalty is
        applied).  For constrained nuisances see
        :class:`~spey.backends.default_pdf.UncorrelatedBackground`.

    The positivity constraint :math:`n^b_i + \theta_i \sigma_i \geq 0` is
    enforced via :class:`~scipy.optimize.NonlinearConstraint`.

    Args:
        signal_yields (``List[float]``): Per-bin signal yields :math:`\{n^s_i\}`.
        background_yields (``List[float]``): Per-bin expected background yields
          :math:`\{n^b_i\}`.
        data (``List[int]``): Per-bin observed counts :math:`\{n^{\rm obs}_i\}`.
        absolute_uncertainties (``List[float]``, default ``None``): Per-bin absolute
          background uncertainties :math:`\{\sigma_i\}`.  When provided, the model
          gains :math:`N` additional unconstrained nuisance parameters.
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
    Product of independent Gaussians — uncorrelated normal likelihood
    (``default.normal``).

    This backend replaces the Poisson term with a Gaussian approximation, which is
    accurate when the expected counts are large.  The likelihood is

    .. math::

        \mathcal{L}(\mu) = \prod_{i=1}^{N}
        \frac{1}{\sigma_i \sqrt{2\pi}}
        \exp\!\left[-\frac{(\mu n^s_i + n^b_i - n^{\rm obs}_i)^2}
                        {2\sigma_i^2}\right],

    where :math:`\sigma_i` is the absolute uncertainty in bin :math:`i`.

    Because the bins are independent this is equivalent to a :math:`\chi^2`
    goodness-of-fit test with the total prediction :math:`\mu n^s_i + n^b_i`
    as the hypothesis.

    .. versionadded:: 0.1.9

    Args:
        signal_yields (``List[float]``): Per-bin signal yields :math:`\{n^s_i\}`.
        background_yields (``List[float]``): Per-bin expected background yields
          :math:`\{n^b_i\}`.
        data (``List[int]``): Per-bin observed counts :math:`\{n^{\rm obs}_i\}`.
        absolute_uncertainties (``List[float]``): Per-bin absolute uncertainties
          :math:`\{\sigma_i\}` that enter the Gaussian widths.
        n_signal_parameters (``int``, default ``0``): number of additional free parameters
          to pass to a callable ``signal_yields``.  Has no effect when ``signal_yields``
          is a plain array.  When greater than zero the optimiser parameter vector is
          extended to ``[mu, signal_par_0, ..., signal_par_{n-1}]``.
        signal_parameter_bounds (:code:`List[Tuple[Optional[float], Optional[float]]] | None`):
          Optimiser bounds for each extra signal parameter.  Each entry
          is a ``(lower, upper)`` pair; use ``None`` for an unbounded side.  When
          ``None``, every extra signal parameter receives ``(None, None)``.  Must have
          exactly ``n_signal_parameters`` entries when provided.
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
        signal_yields: Union[List[float], Callable[[np.ndarray], np.ndarray]],
        background_yields: List[float],
        data: List[int],
        absolute_uncertainties: List[float],
        n_signal_parameters: int = 0,
        signal_parameter_bounds: Optional[
            List[Tuple[Optional[float], Optional[float]]]
        ] = None,
    ):
        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            n_signal_parameters=n_signal_parameters,
            signal_parameter_bounds=signal_parameter_bounds,
        )
        self.absolute_uncertainties = np.array(absolute_uncertainties, dtype=np.float64)
        """absolute uncertainties on the background"""
        self._main_kwargs = {"cov": self.absolute_uncertainties, "pdf_type": "normal"}


class MultivariateNormal(SimplePDFBase):
    r"""
    Multivariate Normal likelihood with optional parameter-dependent covariance
    (``default.multivariate_normal``).

    This backend models the likelihood as a single multivariate normal distribution
    over all bins simultaneously, capturing inter-bin correlations through an
    :math:`N \times N` covariance matrix :math:`\Sigma`:

    .. math::

        \mathcal{L}(\mu) =
        \frac{1}{\sqrt{(2\pi)^N \det\Sigma(\mu)}}
        \exp\!\left[-\frac{1}{2}
        \bigl(\boldsymbol{\lambda}(\mu) - \mathbf{n}^{\rm obs}\bigr)^\top
        \Sigma(\mu)^{-1}
        \bigl(\boldsymbol{\lambda}(\mu) - \mathbf{n}^{\rm obs}\bigr)
        \right],

    where :math:`\lambda_i(\mu) = \mu n^s_i + n^b_i` and :math:`\Sigma(\mu)` may
    be constant or :math:`\mu`-dependent (see below).

    **Callable covariance matrix**

    ``covariance_matrix`` can be a callable that takes the full parameter vector
    ``pars`` and returns an :math:`N \times N` covariance matrix.  This allows, for
    example, signal-strength-dependent uncertainties:

    .. math::

        \Sigma(\mu) = \Sigma_{\rm bkg} + \mu^2\, \Sigma_{\rm sig}.

    .. versionadded:: 0.1.9

    .. versionchanged:: 0.2.6
        Callable ``covariance_matrix`` support added.

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

    .. attention::
        The functional ``signal_yields`` is used to compute the expected signal yields in each bin.
        The optimiser only passes relevant parameter set to ``signal_yields`` function, excluding
        signal strength (i.e. the first parameter :math:`\mu`). Callable ``covariance_matrix``
        receives the full parameter vector, including the POI and any extra signal parameters.

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
        signal_parameter_bounds (:code:`List[Tuple[Optional[float], Optional[float]]] | None`):
          Optimiser bounds for each extra signal parameter.  Each entry
          is a ``(lower, upper)`` pair; use ``None`` for an unbounded side.  When
          ``None``, every extra signal parameter receives ``(None, None)``.  Must have
          exactly ``n_signal_parameters`` entries when provided.

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
        signal_parameter_bounds: Optional[
            List[Tuple[Optional[float], Optional[float]]]
        ] = None,
    ):
        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            n_signal_parameters=n_signal_parameters,
            signal_parameter_bounds=signal_parameter_bounds,
        )

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
            "pdf_type": "multivariate_normal",
        }
