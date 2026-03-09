r"""
Default PDF Backends
====================

This module implements four built-in statistical model backends that cover the most
common simplified-likelihood constructions used in particle-physics analyses.  All
four share a common base class (:class:`DefaultPDFBase`) and therefore the same
gradient-enabled evaluation infrastructure (via :mod:`autograd`).

Overview of the likelihood structure
--------------------------------------

Every backend decomposes the likelihood into a **main model** and a **constraint model**:

.. math::

    \mathcal{L}(\mu, \boldsymbol{\theta})
    = \underbrace{\prod_{i=1}^{N} \mathrm{Poiss}\!\left(n_i^{\rm obs}
      \,\big|\, \lambda_i(\mu, \boldsymbol{\theta})\right)}_{\text{main model}}
    \cdot
    \underbrace{\mathcal{C}(\boldsymbol{\theta})}_{\text{constraint model}}

where :math:`\mu` is the signal-strength parameter of interest, :math:`\boldsymbol{\theta}`
is the vector of nuisance parameters (one per bin), :math:`n_i^{\rm obs}` are the
observed counts, and :math:`\lambda_i` is the expected count in bin :math:`i`.

The four backends differ only in how :math:`\lambda_i` and :math:`\mathcal{C}` are
parametrised.

Available Backends
------------------

``default.uncorrelated_background`` — :class:`UncorrelatedBackground`
    Independent per-bin Gaussian constraints; bins are treated as uncorrelated.

    .. math::

        \lambda_i(\mu, \theta_i) &= \mu\, n^s_i + n^b_i + \theta_i \sigma_i \\[4pt]
        \mathcal{C}(\boldsymbol{\theta}) &= \prod_{i=1}^{N} \mathcal{N}(\theta_i \mid 0, 1)

    where :math:`\sigma_i` is the absolute background uncertainty in bin :math:`i`.
    The constraint :math:`n^b_i + \theta_i \sigma_i \geq 0` is enforced during
    optimisation.

    .. autosummary::
        :toctree: ../_generated/

        UncorrelatedBackground

``default.correlated_background`` — :class:`CorrelatedBackground`
    Simplified likelihood :xref:`1809.05548` with a multivariate normal constraint that
    captures inter-bin background correlations.

    .. math::

        \lambda_i(\mu, \theta_i) &= \mu\, n^s_i + n^b_i + \sigma_i \theta_i \\[4pt]
        \mathcal{C}(\boldsymbol{\theta}) &= \mathcal{N}(\boldsymbol{\theta} \mid \mathbf{0}, \rho)

    where :math:`\sigma_i = \sqrt{\Sigma_{ii}}` are the per-bin background
    uncertainties and :math:`\rho` is the correlation matrix derived from the
    user-supplied covariance matrix :math:`\Sigma`.

    .. autosummary::
        :toctree: ../_generated/

        CorrelatedBackground

``default.third_moment_expansion`` — :class:`ThirdMomentExpansion`
    Extends the simplified likelihood by including third-moment (skewness) information
    to better describe asymmetric uncertainties :xref:`1809.05548`.  The
    :math:`\lambda` function receives a quadratic correction:

    .. math::

        \lambda_i(\mu, \theta_i) = \mu\, n^s_i + A_i + B_i \theta_i + C_i \theta_i^2

    with :math:`A_i`, :math:`B_i`, :math:`C_i` derived from the first three moments
    of the background distribution, and :math:`\rho` is modified accordingly.  See
    :class:`ThirdMomentExpansion` for the full expressions.

    .. autosummary::
        :toctree: ../_generated/

        ThirdMomentExpansion

``default.effective_sigma`` — :class:`EffectiveSigma`
    Variable-Gaussian (effective-:math:`\sigma`) approach inspired by
    :xref:`physics/0406120` Sec. 3.6.  Asymmetric uncertainties
    :math:`(\sigma^+_i, \sigma^-_i)` are absorbed into a :math:`\theta`-dependent
    effective width:

    .. math::

        \sigma^{\rm eff}_i(\theta_i)
        &= \sqrt{\sigma^+_i \sigma^-_i
                 + (\sigma^+_i - \sigma^-_i)(\theta_i - n^b_i)} \\[4pt]
        \lambda_i(\mu, \theta_i)
        &= \mu\, n^s_i + n^b_i + \theta_i\, \sigma^{\rm eff}_i(\theta_i)

    The constraint model is a multivariate normal with a user-supplied correlation
    matrix.

    .. autosummary::
        :toctree: ../_generated/

        EffectiveSigma

Parameter layout
----------------

All backends use the same parameter-vector convention::

    pars = [μ, θ₁, θ₂, …, θ_N, (signal-uncertainty pars)]

* Index 0: signal strength :math:`\mu` (the parameter of interest, POI).
* Indices 1…N: per-bin nuisance parameters :math:`\theta_i`.
* Remaining indices: optional parameters introduced by signal-uncertainty *modifiers*.

Signal uncertainty modifiers
-----------------------------

All backends accept an optional ``modifiers`` argument that adds multiplicative
signal-yield corrections, allowing the user to propagate signal systematic
uncertainties.  Supported modifier types are:

* ``absolute_uncertainties``: symmetric absolute uncertainties on the signal.
* ``absolute_uncertainty_envelops``: asymmetric upper/lower envelopes.

The effective signal yield in bin :math:`i` becomes
:math:`\mu\, n^s_i \cdot f(\boldsymbol{\theta}_{\rm sig})`, where
:math:`f` is constructed by :func:`~.signal_uncertainty_synthesizer`.

References
----------
* Collaboration, CMS, *Simplified likelihood for the re-interpretation of public CMS
  results*, CMS-NOTE-2017-001, :xref:`1809.05548`.
* Barlow, R., *Asymmetric Errors*, :xref:`physics/0406120`, PHYSTAT 2003.
"""

import logging
from typing import Callable, List, Optional, Tuple, Union

from autograd import hessian, jacobian
from autograd import numpy as np
from autograd import value_and_grad
from scipy.optimize import NonlinearConstraint

from spey._version import __version__
from spey.backends.distributions import ConstraintModel, MainModel
from spey.base import BackendBase, ModelConfig
from spey.helper_functions import covariance_to_correlation
from spey.utils import ExpectationType

from .third_moment import third_moment_expansion
from .uncertainty_synthesizer import signal_uncertainty_synthesizer

# pylint: disable=E1101,E1120
log = logging.getLogger("Spey")

# pylint: disable=W1203


class DefaultPDFBase(BackendBase):
    r"""
    Abstract base class shared by all default PDF backends.

    This class handles the common infrastructure:

    * storing signal, background, and observed yields as :mod:`autograd`-compatible
      arrays to enable automatic differentiation of the likelihood;
    * constructing the :class:`~spey.backends.distributions.ModelConfig` with the
      minimum POI, suggested initialisations, and parameter bounds;
    * computing the lower bound on the POI as

      .. math::

          \mu_{\min} = -\min_{i:\,n^s_i>0}
          \frac{n^b_i}{n^s_i},

      which is the most negative signal strength for which no bin has negative
      expected yield;
    * providing default implementations of :meth:`get_objective_function`,
      :meth:`get_logpdf_func`, :meth:`get_hessian_logpdf_func`,
      :meth:`get_sampler`, and :meth:`expected_data` that delegate to the
      per-subclass :attr:`main_model` and :attr:`constraint_model` properties.

    Subclasses must override :attr:`main_model` and :attr:`constraint_model` to
    supply the appropriate :math:`\lambda(\mu,\boldsymbol{\theta})` function and
    constraint distribution.

    Args:
        signal_yields (:code:`np.ndarray | Callable[[np.ndarray], np.ndarray]`): Per-bin
          signal yields :math:`\{n^s_i\}`, or a callable that accepts the extra signal
          parameters ``pars[1 : 1 + n_signal_parameters]`` and returns the per-bin
          yields as a ``np.ndarray``.  When a callable is supplied ``n_signal_parameters``
          must be set to the number of extra parameters the function expects.

          .. note::

              When ``signal_yields`` is callable the minimum POI is set to
              :math:`-\infty` and ``modifiers`` cannot be used simultaneously.

        background_yields (``np.ndarray``): Per-bin expected background yields
          :math:`\{n^b_i\}`.
        data (``np.ndarray``): Per-bin observed counts :math:`\{n^{\rm obs}_i\}`.
        covariance_matrix (``np.ndarray | Callable | None``): Background covariance
          matrix :math:`\Sigma`.  Diagonal elements are squared absolute uncertainties;
          off-diagonal elements encode inter-bin correlations.  May be ``None`` for
          backends that do not use a covariance matrix (e.g.
          :class:`UncorrelatedBackground`), or a callable that returns the matrix
          given an array of parameters.

          .. warning::

              The diagonal entries must be **squared** absolute uncertainties, not
              standard deviations.  For uncorrelated bins pass a diagonal matrix with
              :math:`\Sigma_{ii} = \sigma_i^2`.

        modifiers (``list``, default ``None``): Optional signal-uncertainty
          modifiers.  Each entry is a dict with one of the following keys:

          - ``absolute_uncertainties`` (``List[float]``): symmetric absolute
            uncertainties on the signal.
          - ``absolute_uncertainty_envelops`` (``List[Tuple[float, float]]``):
            asymmetric upper/lower envelopes on the signal.

          Not supported when ``signal_yields`` is a callable.

        n_signal_parameters (``int``, default ``0``): Number of additional free
          parameters that a callable ``signal_yields`` function accepts.  These
          parameters are inserted into the parameter vector *after* :math:`\mu` and
          *before* the background nuisance parameters::

              pars = [μ, sig_par_0, …, sig_par_{n-1}, θ_bkg_1, …, θ_bkg_N, θ_sig_1, …]

          Has no effect when ``signal_yields`` is a plain array.
        signal_parameter_bounds (:code:`List[Tuple[Optional[float], Optional[float]]] | None`):
          Optimiser bounds for each of the ``n_signal_parameters``
          extra signal parameters.  Each entry is a ``(lower, upper)`` pair; use
          ``None`` for either element to leave that side unbounded.  When the argument
          itself is ``None`` every extra signal parameter receives ``(None, None)``.
          Must have exactly ``n_signal_parameters`` entries when provided.

    .. note::

        All array inputs are immediately cast to :func:`autograd.numpy.array` with
        ``dtype=float64`` so that the full parameter-space gradient and Hessian of
        the likelihood can be computed via automatic differentiation.
    """

    name: str = "default.base"
    """Name of the backend"""
    version: str = __version__
    """Version of the backend"""
    author: str = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: str = __version__
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
        signal_yields: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
        background_yields: np.ndarray,
        data: np.ndarray,
        covariance_matrix: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]] = None,
        modifiers: list[Union[List[float], List[Tuple[float, float]]]] = None,
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
        self.covariance_matrix = (
            np.array(covariance_matrix, dtype=np.float64)
            if not callable(covariance_matrix) and covariance_matrix is not None
            else covariance_matrix
        )
        self.n_signal_parameters = n_signal_parameters

        if modifiers is None:
            modifiers = []
            self.signal_uncertainty_configuration = {}
        else:
            if callable(self.signal_yields):
                raise ValueError(
                    "modifiers cannot be combined with a callable signal_yields. "
                    "Provide signal_yields as a plain array when using modifiers."
                )
            self.signal_uncertainty_configuration = signal_uncertainty_synthesizer(
                signal_yields=self.signal_yields,
                modifiers=modifiers,
                n_signal_parameters=n_signal_parameters,
            )

        minimum_poi = -np.inf
        if not callable(self.signal_yields) and self.is_alive:
            minimum_poi = -np.min(
                self.background_yields[self.signal_yields > 0.0]
                / self.signal_yields[self.signal_yields > 0.0]
            )
        log.debug(f"Min POI set to : {minimum_poi}")

        self._main_model = None
        self._constraint_model = None
        self.constraints = []
        """Constraints to be used during optimisation process"""

        n_bkg_pars = len(data)
        n_sig_unc_pars = len(modifiers)
        parameter_names = None
        if n_signal_parameters > 0:
            parameter_names = (
                ["mu"]
                + [f"signal_par_{i}" for i in range(n_signal_parameters)]
                + [f"theta_bkg_{i}" for i in range(n_bkg_pars)]
                + [f"theta_sig_{i}" for i in range(n_sig_unc_pars)]
            )

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
            suggested_init=(
                [1.0]
                + [1.0] * n_signal_parameters
                + [1.0] * n_bkg_pars
                + [1.0] * n_sig_unc_pars
            ),
            suggested_bounds=(
                [(minimum_poi, 10.0)]
                + _sig_par_bounds
                + [(None, None)] * n_bkg_pars
                + [(None, None)] * n_sig_unc_pars
            ),
            parameter_names=parameter_names,
        )

    @property
    def is_alive(self) -> bool:
        """
        Returns True if at least one bin has non-zero signal yield.

        .. versionchanged:: 0.2.7
            When ``signal_yields`` is callable, always returns ``True`` since the
            actual yields are not known at construction time.
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
            poi_upper_bound (``float``, default ``10.0``): upper bound for parameter
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
            parameter_names=self._config.parameter_names,
        )

    @property
    def constraint_model(self) -> ConstraintModel:
        r"""
        Constraint model distribution :math:`\mathcal{C}(\boldsymbol{\theta})`.

        For :class:`DefaultPDFBase` this is a multivariate normal centred at zero
        with the correlation matrix :math:`\rho` derived from
        :attr:`covariance_matrix`,

        .. math::

            \mathcal{C}(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta} \mid \mathbf{0}, \rho).

        Subclasses may override this by assigning ``self._constraint_model`` directly
        in their ``__init__`` (e.g. :class:`UncorrelatedBackground` uses independent
        univariate normals instead).

        Returns:
            :class:`~spey.backends.distributions.ConstraintModel`:
            Lazily-initialised constraint model.
        """
        if self._constraint_model is None:
            corr = covariance_to_correlation(self.covariance_matrix)
            nsp = self.n_signal_parameters
            self._constraint_model = ConstraintModel(
                [
                    {
                        "distribution_type": "multivariatenormal",
                        "args": [np.zeros(len(self.data)), corr],
                        "kwargs": {"domain": slice(1 + nsp, 1 + nsp + corr.shape[0])},
                    }
                ]
                + self.signal_uncertainty_configuration.get("constraint", [])
            )
        return self._constraint_model

    @property
    def main_model(self) -> MainModel:
        r"""
        Main model distribution — the Poisson term of the likelihood.

        For :class:`DefaultPDFBase`, the expected count in bin :math:`i` is

        .. math::

            \lambda_i(\mu, \boldsymbol{\theta})
            = \mu\, n^s_i \cdot f(\boldsymbol{\theta}_{\rm sig})
              + n^b_i + \sigma_i \theta_i,

        where :math:`\sigma_i = \sqrt{\Sigma_{ii}}` and
        :math:`f(\boldsymbol{\theta}_{\rm sig})` is the optional signal-uncertainty
        modifier (unity when no modifiers are provided).

        The positivity constraint :math:`\lambda_i \geq 0` is registered as a
        :class:`~scipy.optimize.NonlinearConstraint` and enforced by the optimiser.

        Returns:
            :class:`~spey.backends.distributions.MainModel`:
            Lazily-initialised main model.
        """
        if self._main_model is None:
            A = self.background_yields
            B = np.sqrt(np.diag(self.covariance_matrix))
            nsp = self.n_signal_parameters

            signal_unc = self.signal_uncertainty_configuration.get(
                "lambda", lambda pars: 1.0
            )

            if callable(self.signal_yields):
                _sig = lambda pars: self.signal_yields(pars[1 : 1 + nsp])  # noqa: E731
            else:
                _yields = self.signal_yields

                def _sig(_, _y=_yields):
                    return _y

            def poiss_lamb(pars: np.ndarray) -> np.ndarray:
                """
                Compute lambda for Main model.
                For details see above eq 2.6 in :xref:`1809.05548`

                Args:
                    pars (``np.ndarray``): nuisance parameters

                Returns:
                    ``np.ndarray``:
                    expectation value of the poisson distribution with respect to
                    nuisance parameters.
                """
                return (
                    pars[0] * _sig(pars) * signal_unc(pars)
                    + A
                    + B * pars[slice(1 + nsp, 1 + nsp + len(B))]
                )

            def constraint(pars: np.ndarray) -> np.ndarray:
                """Compute constraint term"""
                return A + B * pars[slice(1 + nsp, 1 + nsp + len(B))]

            jac_constr = jacobian(constraint)

            self.constraints.append(
                NonlinearConstraint(constraint, 0.0, np.inf, jac=jac_constr)
            )

            self._main_model = MainModel(poiss_lamb)

        return self._main_model

    def get_objective_function(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.ndarray] = None,
        do_grad: bool = True,
    ) -> Callable[[np.ndarray], Union[Tuple[float, np.ndarray], float]]:
        r"""
        Return the objective function :math:`-\ln\mathcal{L}(\mu, \boldsymbol{\theta})` used by
        the optimiser.

        The objective is the negative log-likelihood (NLL) summed over the main and
        constraint models:

        .. math::

            -\ln\mathcal{L}(\mu, \boldsymbol{\theta})
            = -\ln\mathcal{L}_{\rm main}(\mu, \boldsymbol{\theta})
              - \ln\mathcal{C}(\boldsymbol{\theta}).

        When ``do_grad=True`` the function is wrapped with
        :func:`autograd.value_and_grad` so that it simultaneously returns the value
        and the exact gradient with respect to all parameters, which is required by
        gradient-based optimisers.

        Args:
            expected (:class:`~spey.ExpectationType`): Controls which dataset is used
              when the nuisance parameters are profiled.

              * :obj:`~spey.ExpectationType.observed`: Use the real observed data
                (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Use the post-fit expected
                dataset.
              * :obj:`~spey.ExpectationType.apriori`: Use the pre-fit SM expectation
                (background yields) as data.

            data (``np.ndarray``, default ``None``): Override the observed data used
              during fitting.  When ``None`` the data array selected by ``expected``
              is used.
            do_grad (``bool``, default ``True``): If ``True``, the returned callable
              yields ``(nll, grad_nll)``; if ``False`` it yields only ``nll``.

        Returns:
            ``Callable[[np.ndarray], float | Tuple[float, np.ndarray]]``:
            A function of the full parameter vector
            :math:`(\mu, \theta_1, \ldots, \theta_N)` that returns the NLL (and
            optionally its gradient).
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
        Return a callable that evaluates :math:`\ln\mathcal{L}(\mu, \boldsymbol{\theta})`.

        The log-likelihood is

        .. math::

            \ln\mathcal{L}(\mu, \boldsymbol{\theta})
            = \ln\mathcal{L}_{\rm main}(\mu, \boldsymbol{\theta})
              + \ln\mathcal{C}(\boldsymbol{\theta}).

        This is used internally to compute the profile likelihood ratio and for
        Hessian-based variance estimation.

        Args:
            expected (:class:`~spey.ExpectationType`): Dataset prescription.

              * :obj:`~spey.ExpectationType.observed`: Real observed data (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Post-fit expected data.
              * :obj:`~spey.ExpectationType.apriori`: Pre-fit SM expectation
                (background yields used as pseudo-data).

            data (``np.ndarray``, default ``None``): Override data.  Falls back to the
              array selected by ``expected`` when ``None``.

        Returns:
            ``Callable[[np.ndarray], float]``:
            Function of the full parameter vector
            :math:`(\mu, \theta_1, \ldots, \theta_N)` returning the scalar
            log-likelihood value.
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
        Return a callable that evaluates the Hessian of
        :math:`\ln\mathcal{L}(\mu, \boldsymbol{\theta})`.

        The Hessian matrix :math:`H_{ab} = \partial^2 \ln\mathcal{L} / \partial p_a \partial p_b`
        is computed via :func:`autograd.hessian` (exact automatic differentiation).
        Its primary use is to estimate the variance of :math:`\hat\mu` through the
        inverse of the Fisher information matrix evaluated at the best-fit point,

        .. math::

            \sigma_{\hat\mu}^2 \approx \left[H^{-1}\right]_{00},

        where index 0 corresponds to the POI :math:`\mu`.

        Args:
            expected (:class:`~spey.ExpectationType`): Dataset prescription.

              * :obj:`~spey.ExpectationType.observed`: Real observed data (default).
              * :obj:`~spey.ExpectationType.aposteriori`: Post-fit expected data.
              * :obj:`~spey.ExpectationType.apriori`: Pre-fit SM expectation.

            data (``np.ndarray``, default ``None``): Override data.

        Returns:
            ``Callable[[np.ndarray], np.ndarray]``:
            Function of the full parameter vector returning the
            :math:`(N_{\rm par} \times N_{\rm par})` Hessian matrix of
            :math:`\ln\mathcal{L}`.
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
        Return a callable that draws pseudo-data from the statistical model.

        The sampler first draws Poisson counts from the main model at the expected
        yields :math:`\lambda(\mu, \boldsymbol{\theta})`, and optionally appends
        auxiliary observations drawn from the constraint model.  This is used for
        toy-based hypothesis testing.

        Args:
            pars (``np.ndarray``): Fixed parameter vector
              :math:`(\mu, \theta_1, \ldots, \theta_N)` at which to evaluate
              expected yields before sampling.

        Returns:
            ``Callable[[int, bool], np.ndarray]``:
            A function with signature ``sampler(sample_size, include_auxiliary=True)``
            that returns an array of shape
            ``(sample_size, N_bins [+ N_auxiliary])`` containing the generated
            pseudo-data.
        """

        def sampler(sample_size: int, include_auxiliary: bool = True) -> np.ndarray:
            """
            Draw ``sample_size`` pseudo-experiments from the model.

            Args:
                sample_size (``int``): Number of pseudo-experiments to generate.
                include_auxiliary (``bool``): If ``True``, append auxiliary
                  observations drawn from the constraint model to each sample.

            Returns:
                ``np.ndarray``:
                Array of shape ``(sample_size, N_bins [+ N_auxiliary])`` with the
                generated counts.
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
        Compute the expected data vector at the given parameter point.

        Returns the expectation values :math:`\lambda_i(\mu, \boldsymbol{\theta})`
        for all bins, and optionally the auxiliary expected values from the constraint
        model (i.e. the mean of the constraint distribution, which is zero for all
        built-in backends).

        Args:
            pars (``List[float]``): Full parameter vector
              :math:`(\mu, \theta_1, \ldots, \theta_N)`.
            include_auxiliary (``bool``, default ``True``): If ``True``, append the
              expected auxiliary data from the constraint model.

        Returns:
            ``List[float]``:
            Expected bin counts (length :math:`N`), plus auxiliary data if
            ``include_auxiliary=True``.
        """
        data = self.main_model.expected_data(pars)

        if include_auxiliary:
            data = np.hstack([data, self.constraint_model.expected_data()])
        return data


class UncorrelatedBackground(DefaultPDFBase):
    r"""
    Single- or multi-bin simplified likelihood with **uncorrelated** background
    uncertainties (``default.uncorrelated_background``).

    Each bin is assigned its own independent Gaussian nuisance parameter
    :math:`\theta_i`, scaled by the absolute background uncertainty :math:`\sigma_i`.
    The full likelihood is

    .. math::

        \mathcal{L}(\mu, \boldsymbol{\theta})
        = \prod_{i=1}^{N}
          \mathrm{Poiss}\!\left(n^{\rm obs}_i \,\Big|\,
          \mu\, n^s_i \cdot f(\boldsymbol{\theta}_{\rm sig})
          + n^b_i + \theta_i \sigma_i\right)
        \cdot \prod_{i=1}^{N} \mathcal{N}(\theta_i \mid 0, 1),

    where

    * :math:`n^s_i` — signal yield in bin :math:`i`,
    * :math:`n^b_i` — expected background yield in bin :math:`i`,
    * :math:`\sigma_i` — absolute background uncertainty in bin :math:`i`
      (e.g. ``0.5`` for a yield reported as :math:`3.1 \pm 0.5`),
    * :math:`f(\boldsymbol{\theta}_{\rm sig})` — optional signal-uncertainty
      modifier (1 by default).

    The positivity constraint :math:`n^b_i + \theta_i \sigma_i \geq 0` is
    enforced during optimisation via a :class:`~scipy.optimize.NonlinearConstraint`.

    Because the bins are independent the log-likelihood factorises into a sum
    over bins, making this the fastest backend for quick estimates.

    Args:
        signal_yields (:code:`np.ndarray | Callable[[np.ndarray], np.ndarray]`): Per-bin
          signal yields :math:`\{n^s_i\}`, or a callable that accepts the extra signal
          parameters ``pars[1 : 1 + n_signal_parameters]`` and returns the per-bin
          yields as a ``np.ndarray``.
        background_yields (``List[float]``): Per-bin expected background yields
          :math:`\{n^b_i\}`.
        data (``List[int]``): Per-bin observed counts
          :math:`\{n^{\rm obs}_i\}`.
        absolute_uncertainties (``List[float]``): Absolute (not relative) background
          uncertainties :math:`\{\sigma_i\}`.  Must have the same length as the
          other array inputs.
        modifiers (``list``, default ``None``): Optional signal-uncertainty modifiers;
          see :class:`DefaultPDFBase` for the accepted format.  Not supported when
          ``signal_yields`` is callable.
        n_signal_parameters (``int``, default ``0``): Number of additional free
          parameters accepted by a callable ``signal_yields``.  Has no effect when
          ``signal_yields`` is a plain array.  See :class:`DefaultPDFBase` for the
          parameter-vector layout.
        signal_parameter_bounds (:code:`List[Tuple[Optional[float], Optional[float]]] | None`):
          Optimiser bounds for each extra signal parameter.  Each entry
          is a ``(lower, upper)`` pair; use ``None`` for an unbounded side.  When
          ``None``, every extra signal parameter receives ``(None, None)``.  Must have
          exactly ``n_signal_parameters`` entries when provided.

    .. note::

        All input lists must have the same length :math:`N` (number of bins/regions).

    Example:

    .. code:: python3

        >>> import spey
        >>> stat_wrapper = spey.get_backend('default.uncorrelated_background')
        >>> data = [1, 3]
        >>> signal = [0.5, 2.0]
        >>> background = [2.0, 2.8]
        >>> background_unc = [1.1, 0.8]
        >>> stat_model = stat_wrapper(
        ...     signal, background, data, background_unc, analysis="multi-bin", xsection=0.123
        ... )
        >>> print("CLs : %.3f" % tuple(stat_model.exclusion_confidence_level()))
    """

    name: str = "default.uncorrelated_background"
    """Name of the backend"""
    version: str = DefaultPDFBase.version
    """Version of the backend"""
    author: str = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: str = DefaultPDFBase.spey_requires
    """Spey version required for the backend"""

    def __init__(
        self,
        signal_yields: Union[List[float], Callable[[np.ndarray], np.ndarray]],
        background_yields: List[float],
        data: List[int],
        absolute_uncertainties: List[float],
        modifiers: list[Union[List[float], List[Tuple[float, float]]]] = None,
        n_signal_parameters: int = 0,
        signal_parameter_bounds: Optional[
            List[Tuple[Optional[float], Optional[float]]]
        ] = None,
    ):
        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            covariance_matrix=None,
            modifiers=modifiers,
            n_signal_parameters=n_signal_parameters,
            signal_parameter_bounds=signal_parameter_bounds,
        )

        B = np.array(absolute_uncertainties)
        nsp = self.n_signal_parameters

        self._constraint_model: ConstraintModel = ConstraintModel(
            [
                {
                    "distribution_type": "normal",
                    "args": [np.zeros(len(self.data)), np.ones(len(B))],
                    "kwargs": {"domain": slice(1 + nsp, 1 + nsp + len(B))},
                }
            ]
            + self.signal_uncertainty_configuration.get("constraint", [])
        )

        signal_unc = self.signal_uncertainty_configuration.get("lambda", lambda pars: 1.0)

        if callable(self.signal_yields):
            _sig = lambda pars: self.signal_yields(pars[1 : 1 + nsp])  # noqa: E731
        else:
            _yields = self.signal_yields
            _sig = lambda pars: _yields  # noqa: E731

        def poiss_lamb(pars: np.ndarray) -> np.ndarray:
            """Compute lambda for Main model"""
            return (
                self.background_yields
                + pars[slice(1 + nsp, 1 + nsp + len(B))] * B
                + pars[0] * _sig(pars) * signal_unc(pars)
            )

        def constraint(pars: np.ndarray) -> np.ndarray:
            """Compute the constraint term"""
            return self.background_yields + pars[slice(1 + nsp, 1 + nsp + len(B))] * B

        jac_constr = jacobian(constraint)

        self.constraints.append(
            NonlinearConstraint(constraint, 0.0, np.inf, jac=jac_constr)
        )

        self._main_model = MainModel(poiss_lamb)


class CorrelatedBackground(DefaultPDFBase):
    r"""
    Multi-bin simplified likelihood with **correlated** background uncertainties
    (``default.correlated_background``).

    This backend implements the simplified likelihood of :xref:`1809.05548`.
    Inter-bin background correlations are encoded in a user-supplied covariance
    matrix :math:`\Sigma`, which is decomposed into per-bin standard deviations
    :math:`\sigma_i = \sqrt{\Sigma_{ii}}` and a correlation matrix :math:`\rho`.
    The full likelihood is

    .. math::

        \mathcal{L}_{\rm SL}(\mu, \boldsymbol{\theta})
        = \underbrace{
            \prod_{i=1}^{N}
            \mathrm{Poiss}\!\left(n^{\rm obs}_i \,\Big|\,
            \mu\, n^s_i \cdot f(\boldsymbol{\theta}_{\rm sig})
            + n^b_i + \sigma_i \theta_i \right)
          }_{\text{main model}}
        \cdot
        \underbrace{
            \mathcal{N}(\boldsymbol{\theta} \mid \mathbf{0},\, \rho)
          }_{\text{constraint model}},

    where :math:`\rho_{ij} = \Sigma_{ij} / (\sigma_i \sigma_j)` is the
    correlation matrix.  The constraint :math:`n^b_i + \sigma_i \theta_i \geq 0`
    is enforced during optimisation.

    Args:
        signal_yields (:code:`np.ndarray | Callable[[np.ndarray], np.ndarray]`):
          Per-bin signal yields :math:`\{n^s_i\}`, or a callable that accepts
          the extra signal parameters ``pars[1 : 1 + n_signal_parameters]``
          and returns the per-bin yields as a :code:`np.ndarray`.
        background_yields (``np.ndarray``): Per-bin expected background yields
          :math:`\{n^b_i\}`.
        data (``np.ndarray``): Per-bin observed counts :math:`\{n^{\rm obs}_i\}`.
        covariance_matrix (``np.ndarray``): :math:`N \times N` background covariance
          matrix :math:`\Sigma`.  Diagonal entries are squared absolute uncertainties
          :math:`\sigma_i^2`.
        modifiers (``list``, default ``None``): Optional signal-uncertainty modifiers;
          see :class:`DefaultPDFBase` for the accepted format.  Not supported when
          ``signal_yields`` is callable.
        n_signal_parameters (``int``, default ``0``): Number of additional free
          parameters accepted by a callable ``signal_yields``.  Has no effect when
          ``signal_yields`` is a plain array.  See :class:`DefaultPDFBase` for the
          parameter-vector layout.
        signal_parameter_bounds (:code:`List[Tuple[Optional[float], Optional[float]]] | None`):
          Optimiser bounds for each extra signal parameter.  Each entry
          is a ``(lower, upper)`` pair; use ``None`` for an unbounded side.  When
          ``None``, every extra signal parameter receives ``(None, None)``.  Must have
          exactly ``n_signal_parameters`` entries when provided.

    .. note::

        All array inputs must share the same first dimension :math:`N`.  The
        ``covariance_matrix`` must be a square :math:`N \times N` matrix.

    Example:

    .. code:: python3

        >>> import spey
        >>> stat_wrapper = spey.get_backend('default.correlated_background')
        >>> signal_yields = [12.0, 11.0]
        >>> background_yields = [50.0, 52.0]
        >>> data = [51, 48]
        >>> covariance_matrix = [[3., 0.5], [0.6, 7.]]
        >>> statistical_model = stat_wrapper(
        ...     signal_yields, background_yields, data, covariance_matrix
        ... )
        >>> print(statistical_model.exclusion_confidence_level())
    """

    name: str = "default.correlated_background"
    """Name of the backend"""
    version: str = DefaultPDFBase.version
    """Version of the backend"""
    author: str = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: str = DefaultPDFBase.spey_requires
    """Spey version required for the backend"""

    def __init__(
        self,
        signal_yields: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
        background_yields: np.ndarray,
        data: np.ndarray,
        covariance_matrix: np.ndarray,
        modifiers: list[Union[List[float], List[Tuple[float, float]]]] = None,
        n_signal_parameters: int = 0,
        signal_parameter_bounds: Optional[
            List[Tuple[Optional[float], Optional[float]]]
        ] = None,
    ):
        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            covariance_matrix=covariance_matrix,
            modifiers=modifiers,
            n_signal_parameters=n_signal_parameters,
            signal_parameter_bounds=signal_parameter_bounds,
        )

        assert self.main_model is not None, "Unable to build the main model"
        assert self.constraint_model is not None, "Unable to build the constraint model"


class ThirdMomentExpansion(DefaultPDFBase):
    r"""
    Simplified likelihood with **third-moment expansion** to account for skewed
    background distributions (``default.third_moment_expansion``).

    This backend extends :class:`CorrelatedBackground` by incorporating the third
    central moment (skewness) of the background distribution, following
    :xref:`1809.05548` Sec. 2.  Given the first three moments,

    * :math:`m^{(1)}_i` — expected background yield (mean),
    * :math:`m^{(2)}_{ij}` — covariance matrix,
    * :math:`m^{(3)}_i` — diagonal elements of the third-moment tensor (skewness),

    the :math:`\lambda` function receives a quadratic correction and the correlation
    matrix is reparametrised.  Define per-bin coefficients:

    .. math::

        C_i &= -\mathrm{sign}(m^{(3)}_i)\,\sqrt{2\, m^{(2)}_{ii}}
               \cos\!\left(\frac{4\pi}{3}
               + \frac{1}{3}\arctan\!\sqrt{\frac{8\,(m^{(2)}_{ii})^3}{(m^{(3)}_i)^2} - 1}
               \right) \\[4pt]
        B_i &= \sqrt{m^{(2)}_{ii} - 2C_i^2} \\[4pt]
        A_i &= m^{(1)}_i - C_i.

    The inter-bin correlation matrix is then modified to

    .. math::

        \rho_{ij} = \frac{1}{4C_i C_j}
        \left(\sqrt{(B_i B_j)^2 + 8 C_i C_j m^{(2)}_{ij}} - B_i B_j\right),

    and the expected-count function becomes

    .. math::

        \lambda_i(\mu, \theta_i)
        = \mu\, n^s_i \cdot f(\boldsymbol{\theta}_{\rm sig})
          + A_i + B_i \theta_i + C_i \theta_i^2,

    with the constraint model

    .. math::

        \mathcal{C}(\boldsymbol{\theta})
        = \mathcal{N}(\boldsymbol{\theta} \mid \mathbf{0},\, \rho^{-1}).

    The quadratic :math:`C_i \theta_i^2` term captures the asymmetry of the
    background distribution; when :math:`m^{(3)}_i = 0` the expansion reduces to
    the standard simplified likelihood.

    Args:
        signal_yields (:code:`np.ndarray | Callable[[np.ndarray], np.ndarray]`): Per-bin
          signal yields :math:`\{n^s_i\}`, or a callable that accepts the extra signal
          parameters ``pars[1 : 1 + n_signal_parameters]`` and returns the per-bin
          yields as a ``np.ndarray``.
        background_yields (``np.ndarray``): Per-bin expected background yields
          :math:`\{m^{(1)}_i\}`.
        data (``np.ndarray``): Per-bin observed counts.
        covariance_matrix (``np.ndarray``): :math:`N \times N` covariance matrix
          :math:`\{m^{(2)}_{ij}\}`.
        third_moment (``np.ndarray``): Per-bin diagonal third-moment values
          :math:`\{m^{(3)}_i\}`.  Must have length :math:`N`.
        modifiers (``list``, default ``None``): Optional signal-uncertainty modifiers;
          see :class:`DefaultPDFBase` for the accepted format.  Not supported when
          ``signal_yields`` is callable.
        n_signal_parameters (``int``, default ``0``): Number of additional free
          parameters accepted by a callable ``signal_yields``.  Has no effect when
          ``signal_yields`` is a plain array.  See :class:`DefaultPDFBase` for the
          parameter-vector layout.
        signal_parameter_bounds (:code:`List[Tuple[Optional[float], Optional[float]]] | None`):
          Optimiser bounds for each extra signal parameter.  Each entry
          is a ``(lower, upper)`` pair; use ``None`` for an unbounded side.  When
          ``None``, every extra signal parameter receives ``(None, None)``.  Must have
          exactly ``n_signal_parameters`` entries when provided.

    .. note::

        All array inputs must share the same first dimension :math:`N`, and
        ``covariance_matrix`` must be :math:`N \times N`.

    References:
        :xref:`1809.05548`, Sec. 2.
    """

    name: str = "default.third_moment_expansion"
    """Name of the backend"""
    version: str = DefaultPDFBase.version
    """Version of the backend"""
    author: str = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: str = DefaultPDFBase.spey_requires
    """Spey version required for the backend"""
    doi: List[str] = ["10.1007/JHEP04(2019)064"]
    """Citable DOI for the backend"""
    arXiv: List[str] = ["1809.05548"]
    """arXiv reference for the backend"""

    def __init__(
        self,
        signal_yields: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
        background_yields: np.ndarray,
        data: np.ndarray,
        covariance_matrix: np.ndarray,
        third_moment: np.ndarray,
        modifiers: list[Union[List[float], List[Tuple[float, float]]]] = None,
        n_signal_parameters: int = 0,
        signal_parameter_bounds: Optional[
            List[Tuple[Optional[float], Optional[float]]]
        ] = None,
    ):
        third_moments = np.array(third_moment)

        super().__init__(
            signal_yields=signal_yields,
            background_yields=background_yields,
            data=data,
            covariance_matrix=covariance_matrix,
            modifiers=modifiers,
            n_signal_parameters=n_signal_parameters,
            signal_parameter_bounds=signal_parameter_bounds,
        )

        A, B, C, corr = third_moment_expansion(
            self.background_yields, self.covariance_matrix, third_moments, True
        )

        nsp = self.n_signal_parameters
        signal_unc = self.signal_uncertainty_configuration.get("lambda", lambda pars: 1.0)

        if callable(self.signal_yields):
            _sig = lambda pars: self.signal_yields(pars[1 : 1 + nsp])  # noqa: E731
        else:
            _yields = self.signal_yields

            def _sig(_, _y=_yields):
                return _y

        def poiss_lamb(pars: np.ndarray) -> np.ndarray:
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
                + B * pars[slice(1 + nsp, 1 + nsp + len(B))]
                + C * np.square(pars[slice(1 + nsp, 1 + nsp + len(B))])
            )
            return pars[0] * _sig(pars) * signal_unc(pars) + nI

        def constraint(pars: np.ndarray) -> np.ndarray:
            """Compute constraint term"""
            return (
                A
                + B * pars[slice(1 + nsp, 1 + nsp + len(B))]
                + C * np.square(pars[slice(1 + nsp, 1 + nsp + len(B))])
            )

        jac_constr = jacobian(constraint)

        self.constraints.append(
            NonlinearConstraint(constraint, 0.0, np.inf, jac=jac_constr)
        )

        self._main_model = MainModel(poiss_lamb)
        self._constraint_model = ConstraintModel(
            [
                {
                    "distribution_type": "multivariatenormal",
                    "args": [np.zeros(len(self.data)), corr],
                    "kwargs": {"domain": slice(1 + nsp, 1 + nsp + len(B))},
                }
            ]
            + self.signal_uncertainty_configuration.get("constraint", [])
        )


class EffectiveSigma(DefaultPDFBase):
    r"""
    Simplified likelihood with **asymmetric (effective-sigma) background uncertainties**
    (``default.effective_sigma``).

    This backend handles asymmetric background uncertainties through the
    variable-Gaussian (effective-:math:`\sigma`) approach of :xref:`physics/0406120`
    Sec. 3.6, eqs. 18–19.  Given per-bin upper and lower absolute uncertainties
    :math:`\sigma^+_i` and :math:`\sigma^-_i`, an effective width is defined that
    depends on the nuisance-parameter value:

    .. math::

        \sigma^{\rm eff}_i(\theta_i)
        = \sqrt{\sigma^+_i \sigma^-_i
                + (\sigma^+_i - \sigma^-_i)(\theta_i - n^b_i)},

    clipped from below at :math:`10^{-10}` for numerical stability.  The full
    likelihood is

    .. math::

        \mathcal{L}(\mu, \boldsymbol{\theta})
        = \prod_{i=1}^{N}
          \mathrm{Poiss}\!\left(n^{\rm obs}_i \,\Big|\,
          \mu\, n^s_i \cdot f(\boldsymbol{\theta}_{\rm sig})
          + n^b_i + \theta_i\, \sigma^{\rm eff}_i(\theta_i)\right)
        \cdot \mathcal{N}(\boldsymbol{\theta} \mid \mathbf{0},\, \rho),

    where :math:`\rho` is a user-supplied correlation matrix.

    .. note::

        The positivity constraint

        .. math::

            n^b_i + \theta_i\, \sigma^{\rm eff}_i(\theta_i) \geq 0

        is enforced via a :class:`~scipy.optimize.NonlinearConstraint` during
        optimisation.

    When :math:`\sigma^+_i = \sigma^-_i \equiv \sigma_i` the effective sigma
    reduces to the constant :math:`\sigma_i`, recovering the symmetric
    :class:`CorrelatedBackground` result.

    Args:
        signal_yields (:code:`np.ndarray | Callable[[np.ndarray], np.ndarray]`): Per-bin
          signal yields :math:`\{n^s_i\}`, or a callable that accepts the extra signal
          parameters ``pars[1 : 1 + n_signal_parameters]`` and returns the per-bin
          yields as a ``np.ndarray``.
        background_yields (``np.ndarray``): Per-bin expected background yields
          :math:`\{n^b_i\}`.
        data (``np.ndarray``): Per-bin observed counts :math:`\{n^{\rm obs}_i\}`.
        correlation_matrix (``np.ndarray``): :math:`N \times N` inter-bin
          correlation matrix :math:`\rho`.
        absolute_uncertainty_envelops (``List[Tuple[float, float]]``): Per-bin pairs
          :math:`(\sigma^+_i, \sigma^-_i)` of upper and lower absolute background
          uncertainties.  Both values are taken as absolute (sign is ignored).
        modifiers (``list``, default ``None``): Optional signal-uncertainty modifiers;
          see :class:`DefaultPDFBase` for the accepted format.  Not supported when
          ``signal_yields`` is callable.
        n_signal_parameters (``int``, default ``0``): Number of additional free
          parameters accepted by a callable ``signal_yields``.  Has no effect when
          ``signal_yields`` is a plain array.  See :class:`DefaultPDFBase` for the
          parameter-vector layout.
        signal_parameter_bounds (:code:`List[Tuple[Optional[float], Optional[float]]] | None`):
          Optimiser bounds for each extra signal parameter.  Each entry
          is a ``(lower, upper)`` pair; use ``None`` for an unbounded side.  When
          ``None``, every extra signal parameter receives ``(None, None)``.  Must have
          exactly ``n_signal_parameters`` entries when provided.

    References:
        Barlow, R., *Asymmetric Errors*, :xref:`physics/0406120` Sec. 3.6, eqs. 18–19.
    """

    name: str = "default.effective_sigma"
    """Name of the backend"""
    version: str = DefaultPDFBase.version
    """Version of the backend"""
    author: str = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: str = DefaultPDFBase.spey_requires
    """Spey version required for the backend"""
    doi: List[str] = ["10.1142/9781860948985_0013"]
    """Citable DOI for the backend"""
    arXiv: List[str] = ["physics/0406120"]
    """arXiv reference for the backend"""

    def __init__(
        self,
        signal_yields: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
        background_yields: np.ndarray,
        data: np.ndarray,
        correlation_matrix: np.ndarray,
        absolute_uncertainty_envelops: List[Tuple[float, float]],
        modifiers: list[Union[List[float], List[Tuple[float, float]]]] = None,
        n_signal_parameters: int = 0,
        signal_parameter_bounds: Optional[
            List[Tuple[Optional[float], Optional[float]]]
        ] = None,
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
            modifiers=modifiers,
            n_signal_parameters=n_signal_parameters,
            signal_parameter_bounds=signal_parameter_bounds,
        )

        A = self.background_yields
        nsp = self.n_signal_parameters

        self._constraint_model: ConstraintModel = ConstraintModel(
            [
                {
                    "distribution_type": "multivariatenormal",
                    "args": [np.zeros(len(self.data)), correlation_matrix],
                    "kwargs": {"domain": slice(1 + nsp, 1 + nsp + len(A))},
                }
            ]
            + self.signal_uncertainty_configuration.get("constraint", [])
        )

        signal_unc = self.signal_uncertainty_configuration.get("lambda", lambda pars: 1.0)

        if callable(self.signal_yields):
            _sig = lambda pars: self.signal_yields(pars[1 : 1 + nsp])  # noqa: E731
        else:
            _yields = self.signal_yields

            def _sig(_, _y=_yields):
                return _y

        # arXiv:pyhsics/0406120 eq. 18-19
        def effective_sigma(pars: np.ndarray) -> np.ndarray:
            """Compute effective sigma"""
            # clip from 1e-10 to avoid negative or zero values
            # this allows more numeric stability
            return np.sqrt(
                np.clip(
                    sigma_plus * sigma_minus
                    + (sigma_plus - sigma_minus)
                    * (pars[slice(1 + nsp, 1 + nsp + len(A))] - A),
                    1e-10,
                    None,
                )
            )

        def poiss_lamb(pars: np.ndarray) -> np.ndarray:
            """Compute lambda for Main model"""
            return (
                A
                + effective_sigma(pars) * pars[slice(1 + nsp, 1 + nsp + len(A))]
                + pars[0] * _sig(pars) * signal_unc(pars)
            )

        def constraint(pars: np.ndarray) -> np.ndarray:
            """Compute the constraint term"""
            return A + effective_sigma(pars) * pars[slice(1 + nsp, 1 + nsp + len(A))]

        jac_constr = jacobian(constraint)
        self.constraints.append(
            NonlinearConstraint(constraint, 0.0, np.inf, jac=jac_constr)
        )

        self._main_model = MainModel(poiss_lamb)
