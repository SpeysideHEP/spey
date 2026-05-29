.. _sec:plugins:

Plug-ins
========

.. contents::
    :backlinks: none
    :depth: 3

.. list-table::
   :header-rows: 1

   * - Keyword
     - Summary
   * - ``'default.uncorrelated_background'``
     - :ref:`Combination of Poisson and Gaussian PDF, assuming uncorrelated bins. <uncorrelated_background>`
   * - ``'default.correlated_background'``
     - :ref:`Combination of Poisson and Gaussian PDF, with correlated bins. <correlated_background>`
   * - ``'default.third_moment_expansion'``
     - :ref:`Simplified likelihood, extended with third moments of the background. <third_moment_expansion>`
   * - ``'default.effective_sigma'``
     - :ref:`Simplified likelihood, extended with asymmetric uncertainties.  <effective_sigma>`
   * - ``'default.poisson'``
     - :ref:`Poisson distribution, without uncertainties.  <poisson>`
   * - ``'default.normal'``
     - :ref:`Gaussian distribution.  <normal>`
   * - ``'default.multivariate_normal'``
     - :ref:`Multivariate Normal distribution.  <multinormal>`
   * - ``'pyhf'``
     - `External plug-in <https://spey-pyhf.readthedocs.io>`_ uses ``pyhf`` to construct the likelihoods.
   * - ``'pyhf.uncorrelated_background'``
     - `External plug-in <https://spey-pyhf.readthedocs.io>`_ constructs ``pyhf``-based uncorrelated likelihoods.
   * - ``'pyhf.simplify'``
     - `See doc. <https://spey-pyhf.readthedocs.io/en/main/simplify.html>`_ converts full ``pyhf`` likelihoods into simplified framework.
   * - ``'strathisla.full_nuisance_parameters'``, ``'strathisla.simple_multivariate_gaussian_eft'``, ``'strathisla.multivariate_gaussian_scaled_covariance_eft'``
     - `EFT plug-in <https://github.com/joes-git/spey-strathisla>`_ provides tools for full modeling of nuisance parameters and setting limits on Effective Field Theories. By `Joe Egan <mailto:joseph.caimin.egan@cern.ch>`_.

.. meta::
    :property=og:title: Plug-ins
    :property=og:description: Currently supported likelihood prescriptions.
    :property=og:image: https://spey.readthedocs.io/en/main/_static/spey-logo.png
    :property=og:url: https://spey.readthedocs.io/en/main/plugins.html

Spey seamlessly integrates with diverse packages that offer specific
statistical model prescriptions. Its primary objective is to centralise
these prescriptions within a unified interface, facilitating the
combination of different likelihood sources. This section
will overview the currently available plugins accessible
through the Spey interface. The string-based accessors
to the available plugins can be seen using the following command:

.. code-block:: python3

    >>> spey.AvailableBackends()
    >>> # ['default.correlated_background',
    >>> #  'default.effective_sigma',
    >>> #  'default.third_moment_expansion',
    >>> #  'default.uncorrelated_background']

where once installed without any plug-ins :func:`~spey.AvailableBackends` function
only shows the default PDFs. In the following section, we will summarise their usability.
Once we know the accessor of the plug-in, it can be called using :func:`~spey.get_backend`
function e.g.

.. code-block:: python3

    >>> pdf_wrapper = spey.get_backend('default.uncorrelated_background')

this will automatically create a wrapper around the likelihood prescription and allow `spey`
to use it properly. We will demonstrate the usage for each of the default plugins below.

.. note::

    Documentation of each plug-in has been included within the ``pdf_wrapper`` documentation.
    Hence, if one types ``pdf_wrapper?`` in the ipython command line or a jupyter notebook, it is
    possible to access the extended documentation for both the wrapper and the backend in use.

.. attention::

    :func:`~spey.get_backend` function is a wrapper between the PDF prescription and ``spey`` package.
    Once initialised, all PDF prescriptions are defined with :obj:`~spey.StatisticalModel` class
    which provides a backend agnostic interface, i.e. all PDF prescriptions will have the same functionality.

Default Plug-ins
----------------
All default plug-ins are defined using the following main likelihood structure

.. math::

    \mathcal{L}(\mu,\theta) = \underbrace{\prod_{i\in{\rm bins}}
    \mathcal{M}(n_i|\lambda_i(\mu, \theta))}_{\rm main}\cdot
    \underbrace{\prod_{j\in{\rm nui}}\mathcal{C}(\theta_j)}_{\rm constraint} \ ,

**Main model term** :math:`\mathcal{M}`: Describes how the signal hypothesis affects the observed event counts in each bin.
Typically a Poisson or Gaussian distribution where the expected mean depends on both the signal strength :math:`\mu` and
nuisance parameters :math:`\theta`.

**Constraint model term** :math:`\mathcal{C}`: Encodes our prior knowledge of systematic uncertainties through Gaussian
or multivariate distributions on the nuisance parameters. These constraints prevent the fit from indefinitely adjusting
nuisances to arbitrarily improve the likelihood.

.. _uncorrelated_background:

``'default.uncorrelated_background'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the simplest PDF where each bin is treated as statistically independent with uncorrelated
background uncertainties. The likelihood is given as

.. math::

    \mathcal{L}(\mu, \theta) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + n_b^i +
    \theta^i\sigma_b^i) \cdot \prod_{j\in{\rm nui}}\mathcal{N}(\theta^j|0, 1)\ ,

**Physical interpretation:**

- :math:`\mu`: Signal strength (parameter of interest), scales the predicted signal yields
- :math:`\theta^i`: Nuisance parameter for bin :math:`i`, representing the pull (in units of standard deviations)
  on the background estimate in that bin. A value :math:`\theta^i = 0` means the background is at its nominal estimate.
- :math:`n_s^i, n_b^i, \sigma_b^i`: Signal yield, background yield, and absolute uncertainty for bin :math:`i`
- The expected count in bin :math:`i` is :math:`\lambda_i = \mu n_s^i + n_b^i + \theta^i\sigma_b^i`
- Each bin's observation follows a Poisson distribution with mean :math:`\lambda_i`
- The Gaussian constraint :math:`\mathcal{N}(\theta^j|0,1)` enforces that nuisance parameters stay bounded near their
  nominal values; large deviations are penalized in the likelihood

**Computational details:**
Since each bin is independent and has an independent nuisance parameter, the joint likelihood factorises into a product.
The NLL is therefore a sum of individual bin contributions, which simplifies optimization.

This model can be used as follows:

.. code-block:: python3
    :linenos:

    >>> pdf_wrapper = spey.get_backend("default.uncorrelated_background")
    >>> statistical_model = pdf_wrapper(
    ...     signal_yields=[12.0, 15.0],
    ...     background_yields=[50.0,48.0],
    ...     data=[36, 33],
    ...     absolute_uncertainties=[12.0,16.0],
    ...     analysis="example",
    ...     xsection=0.123,
    ... )

**Arguments:**

 * ``signal_yields``: keyword for signal yields. It can take one or more values as a list or NumPy array.
 * ``background_yields``: keyword for background-only expectations. It can take one or more values as a list or NumPy array.
 * ``data``: keyword for observations. It can take one or more values as a list or NumPy array.
 * ``absolute_uncertainties``: keyword for absolute uncertainties (not percentage value). It can take one or more values as a list or NumPy array.
 * ``analysis`` (optional): Unique identifier for the analysis.
 * ``xsection`` (optional): Cross-section value for the signal hypothesis. Units determined by the user.

This particular example implements a two-bin histogram with uncorrelated bins. The exclusion CL
(:math:`1-CL_s`) can be computed via :func:`spey.StatisticalModel.exclusion_confidence_level` function.

.. code-block:: python3

    >>> statistical_model.exclusion_confidence_level()
    >>> # [0.9701795436411219]

For all the properties of :obj:`~spey.StatisticalModel` class, we refer the reader to the corresponding
API description.

.. _correlated_background:

``'default.correlated_background'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This plugin extends the uncorrelated case by embedding correlations between bins through a covariance matrix.
The likelihood structure is

.. math::

    \mathcal{L}(\mu, \theta) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + n_b^i +
    \theta^i\sigma_b^i) \cdot \mathcal{N}(\theta|0, \Sigma)\ ,

**Key difference from uncorrelated case:**
The constraint term changes from independent Gaussians :math:`\prod_j \mathcal{N}(\theta^j|0, 1)` to a
multivariate Gaussian :math:`\mathcal{N}(\theta|0, \Sigma)` where :math:`\Sigma` is the covariance matrix
of the nuisance parameters.

**Mathematical details:**

- The covariance matrix :math:`\Sigma` encodes correlations between bin-to-bin systematic uncertainties
- Diagonal elements :math:`\Sigma_{ii}` capture the variance of bin :math:`i`, so :math:`\sigma_b^i = \sqrt{\Sigma_{ii}}`
- Off-diagonal elements :math:`\Sigma_{ij}` with :math:`i \neq j` capture correlations: positive values indicate that upward fluctuations in bin :math:`i` are correlated with upward fluctuations in bin :math:`j`
- The multivariate Gaussian imposes: :math:`-\frac{1}{2}\theta^T \Sigma^{-1} \theta` in the log-likelihood


**Physical motivation:**
Correlated uncertainties arise from shared systematic sources. For example, if bins are in the same physics process,
they may share energy scale uncertainties, efficiency uncertainties, or other common factors.

Iterating on the same example, a correlated two-bin histogram can be defined as:

.. code-block:: python3
    :linenos:

    >>> pdf_wrapper = spey.get_backend("default.correlated_background")
    >>> statistical_model = pdf_wrapper(
    ...     signal_yields=[12.0, 15.0],
    ...     background_yields=[50.0,48.0],
    ...     data=[36, 33],
    ...     covariance_matrix=[[144.0,13.0], [25.0, 256.0]],
    ...     analysis="example",
    ...     xsection=0.123,
    ... )

which leads to the following exclusion limit

.. code-block:: python3

    >>> statistical_model.exclusion_confidence_level()
    >>> # [0.9635100547173434]

As can be seen from the two results, the correlation between histogram bins reduces the exclusion limit
as expected.

**Arguments:**

 * ``signal_yields``: keyword for signal yields. It can take one or more values as a list or NumPy array.
 * ``background_yields``: keyword for background-only expectations. It can take one or more values as a list or NumPy array.
 * ``data``: keyword for observations. It can take one or more values as a list or NumPy array.
 * ``covariance_matrix``: Covariance matrix which captures the background hypothesis's correlations and absolute uncertainty values.
   For absolute uncertainty :math:`\sigma_b`; :math:`\sigma_b = \sqrt{{\rm diag}(\Sigma)}`. The covariance matrix should be a square matrix
   and both dimensions should match the number of ``background_yields`` given as input.
 * ``analysis`` (optional): Unique identifier for the analysis.
 * ``xsection`` (optional): Cross-section value for the signal hypothesis. Units determined by the user.

.. _third_moment_expansion:

``'default.third_moment_expansion'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This plug-in implements the third-moment expansion from :cite:`Buckley:2018vdr`, which captures non-Gaussian
(asymmetric) effects in the background distribution. When background distributions are skewed, this method provides
better likelihood estimates than the quadratic approximation (Gaussian constraint).

The likelihood structure is

.. math::

    \mathcal{L}(\mu, \theta) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + \bar{n}_b^i + B_i\theta_i + S_i\theta_i^2)
     \cdot \mathcal{N}(\theta|0, \rho)\ ,

**Mathematical interpretation:**
Instead of the nominal background estimate :math:`n_b^i` with a linear uncertainty :math:`\theta^i \sigma_b^i`,
the model expands to include a quadratic term :math:`S_i\theta_i^2` to capture skewness. The coefficients are:

- :math:`\bar{n}_b^i`: Shifted nominal background (accounts for skewness)
- :math:`B_i`: Linear coefficient (analogous to standard deviation)
- :math:`S_i`: Quadratic coefficient (encodes the skewness/third moment)

The multivariate Gaussian constraint :math:`\mathcal{N}(\theta|0, \rho)` now has a correlation structure
:math:`\rho` that is computed from the full third-moment tensor to maintain consistency.

where :math:`\bar{n}_b^i,\ B_i,\ S_i` and :math:`\rho` are defined as:

.. math::

    S_i = -sign(m^{(3)}_i) \sqrt{2 diag(\Sigma)_i^2}
    \times\cos\left( \frac{4\pi}{3} +
        \frac{1}{3}\arctan\left(\sqrt{ \frac{8(diag(\Sigma)_i^2)^3}{(m^{(3)}_i)^2} - 1}\right) \right)\ ,

.. math::

    B_i = \sqrt{diag{\Sigma}_i - 2 S_i^2}\ ,

.. math::

    \bar{n}_b^i =  n_b^i - S_i\ ,

.. math::

    \rho_{ij} = \frac{1}{4S_iS_j} \left( \sqrt{(B_iB_j)^2 + 8S_iS_j\Sigma_{ij}} - B_iB_j \right)

iterating over the same example, this PDF can be accessed as follows

.. code-block:: python3
    :linenos:

    >>> pdf_wrapper = spey.get_backend("default.third_moment_expansion")
    >>> statistical_model = pdf_wrapper(
    ...     signal_yields=[12.0, 15.0],
    ...     background_yields=[50.0,48.0],
    ...     data=[36, 33],
    ...     covariance_matrix=[[144.0,13.0], [25.0, 256.0]],
    ...     third_moment=[0.5, 0.8],
    ...     analysis="example",
    ...     xsection=0.123,
    ... )

and the exclusion limit, as before, can be computed as follows

.. code-block:: python3

    >>> statistical_model.exclusion_confidence_level()
    >>> # [0.9614329616396733]

As can be seen from the result, slight skewness generated by the third moment presented in the function
reduced the exclusion limit.

**Arguments:**

 * ``signal_yields``: keyword for signal yields. It can take one or more values as a list or NumPy array.
 * ``background_yields``: keyword for background-only expectations. It can take one or more values as a list or NumPy array.
 * ``data``: keyword for observations. It can take one or more values as a list or NumPy array.
 * ``covariance_matrix``: Covariance matrix which captures the correlations and absolute uncertainty values of the background hypothesis.
   For absolute uncertainty :math:`\sigma_b`; :math:`\sigma_b = \sqrt{{\rm diag}(\Sigma)}`. The covariance matrix should be a square matrix
   and both dimensions should match the number of ``background_yields`` given as input.
 * ``third_moment``: Diagonal elements of the third moments. These can be computed using
   :func:`~spey.backends.default.third_moment.compute_third_moments` function; however this function computes third moments using
   Bifurcated Gaussian, which may not be suitable for every case.
 * ``analysis`` (optional): Unique identifier for the analysis.
 * ``xsection`` (optional): Cross-section value for the signal hypothesis. Units determined by the user.

.. _effective_sigma:

``'default.effective_sigma'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The skewness of the PDF distribution can also be captured by building an effective (theta-dependent) variance
from the asymmetric upper (:math:`\sigma^+`) and lower (:math:`\sigma^-`) uncertainty envelopes, originally
proposed in :cite:`Barlow:2004wg`. This method provides a simpler alternative to the third-moment expansion.

The effective uncertainty is

.. math::

    \sigma_{\rm eff}^i(\theta^i) = \sqrt{\sigma^+_i\sigma^-_i + (\sigma^+_i - \sigma^-_i)\theta^i}

**Physical interpretation:**
- When :math:`\theta^i = 0` (nominal), :math:`\sigma_{\rm eff}^i(0) = \sqrt{\sigma^+_i\sigma^-_i}` is the geometric mean
- For :math:`\theta^i > 0` (upward fluctuation), the effective uncertainty smoothly increases toward :math:`\sigma^+_i`
- For :math:`\theta^i < 0` (downward fluctuation), it smoothly increases toward :math:`\sigma^-_i`
- This naturally captures asymmetric uncertainties without introducing explicit quadratic terms

The Poisson model is generalized as

.. math::

    \mathcal{L}(\mu, \theta) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + n_b^i + \theta^i\sigma_{\rm eff}^i(\theta^i))
     \cdot \mathcal{N}(\theta|0, \rho)\ ,

where the expected count in bin :math:`i` depends on both the nuisance parameter :math:`\theta^i` and the effective
uncertainty :math:`\sigma_{\rm eff}^i(\theta^i)`. The correlation matrix :math:`\rho` encodes correlations between
bins' nuisance parameters.

This PDF can be utilised as follows:

.. code-block:: python3
    :linenos:

    >>> pdf_wrapper = spey.get_backend("default.effective_sigma")
    >>> statistical_model = pdf_wrapper(
    ...     signal_yields=[12.0, 15.0],
    ...     background_yields=[50.0,48.0],
    ...     data=[36, 33],
    ...     correlation_matrix=[[1., 0.06770833], [0.13020833, 1.]],
    ...     absolute_uncertainty_envelops=[(10., 15.), (13., 18.)],
    ...     analysis="example",
    ...     xsection=0.123,
    ... )

where ``absolute_uncertainty_envelops`` refers to each bin's upper and lower uncertainty envelopes.
Once again, the exclusion limit can be computed as

.. code-block:: python3

    >>> statistical_model.exclusion_confidence_level()
    >>> # [0.8567802529243093]

**Arguments:**

 * ``signal_yields``: keyword for signal yields. It can take one or more values as a list or NumPy array.
 * ``background_yields``: keyword for background-only expectations. It can take one or more values as a list or NumPy array.
 * ``data``: keyword for observations. It can take one or more values as a list or NumPy array.
 * ``correlation_matrix``: The correlation matrix should be a square matrix, and both dimensions
   should match the number of ``background_yields`` given as input. If only the covariance matrix is available,
   one can use :func:`~spey.helper_functions.covariance_to_correlation` function to convert the covariance matrix to
   a correlation matrix.
 * ``absolute_uncertainty_envelops``: This is a list of upper and lower uncertainty envelops where the first element of each
   input should be the upper uncertainty, and the second element should be the lower uncertainty envelop, e.g.,
   for background given as :math:`{n_b}_{-\sigma^-}^{+\sigma^+}` the input should be [(:math:`|\sigma^+|`, :math:`|\sigma^-|`)].
 * ``analysis`` (optional): Unique identifier for the analysis.
 * ``xsection`` (optional): Cross-section value for the signal hypothesis. Units determined by the user.

.. _poisson:

``'default.poisson'``
~~~~~~~~~~~~~~~~~~~~~~~~~

Simple Poisson implementation without any systematic uncertainties or nuisance parameters. This is the simplest
likelihood model, appropriate when background yields are known with negligible uncertainty.

.. math::

    \mathcal{L}(\mu) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + n_b^i)

**When to use:**
This model is appropriate when:
- Background yields are precisely known from Monte Carlo or sideband measurements with negligible uncertainty
- The analysis is limited by statistical fluctuations in the data, not systematic uncertainties
- Simplicity is valued over accounting for all sources of uncertainty

The likelihood depends only on the signal strength :math:`\mu` and the fixed background :math:`n_b^i`.
Each bin's observation follows a Poisson distribution with mean :math:`\lambda_i = \mu n_s^i + n_b^i`.
It can accommodate any number of bins.

.. code-block:: python3
    :linenos:

    >>> pdf_wrapper = spey.get_backend("default.poisson")
    >>> statistical_model = pdf_wrapper(
    ...     signal_yields=[12.0, 15.0],
    ...     background_yields=[50.0,48.0],
    ...     data=[36, 33],
    ...     analysis="example",
    ...     xsection=0.123,
    ... )

**Arguments:**

 * ``signal_yields``: keyword for signal yields. It can take one or more values as a list or NumPy array.
 * ``background_yields``: keyword for background-only expectations. It can take one or more values as a list or NumPy array.
 * ``data``: keyword for observations. It can take one or more values as a list or NumPy array.
 * ``analysis`` (optional): Unique identifier for the analysis.
 * ``xsection`` (optional): Cross-section value for the signal hypothesis. Units determined by the user.

.. _normal:

``'default.normal'``
~~~~~~~~~~~~~~~~~~~~~~~~~

Simple univariate Gaussian likelihood without nuisance parameters. Each bin is treated as a normally distributed
random variable.

.. math::

    \mathcal{L}(\mu) = \prod_{i\in{\rm bins}} \frac{1}{\sigma^i \sqrt{2\pi}} \exp\left[-\frac{1}{2} \left(\frac{\mu n_s^i + n_b^i - n^i}{\sigma^i} \right)^2 \right]

**When to use:**
This model is appropriate when:
- Event counts are large enough that the Poisson distribution is well-approximated by a Gaussian (large-N limit)
- Bin yields are treated as continuous rather than discrete quantities
- Background uncertainties are symmetric and well-described by a single absolute uncertainty :math:`\sigma^i`
- No correlated uncertainties between bins (use `multivariate_normal` if correlations exist)

**Mathematical details:**
For each bin, the observation :math:`n^i` is normally distributed with:
- Mean: :math:`\mu n_s^i + n_b^i` (signal-plus-background prediction)
- Standard deviation: :math:`\sigma^i` (absolute uncertainty on the background)

The log-likelihood is the sum of Gaussian log-pdfs across bins. Each bin's contribution is independent.
It can accommodate any number of yields.

.. code-block:: python3
    :linenos:

    >>> pdf_wrapper = spey.get_backend("default.normal")
    >>> statistical_model = pdf_wrapper(
    ...     signal_yields=[12.0, 15.0],
    ...     background_yields=[50.0,48.0],
    ...     data=[36, 33],
    ...     absolute_uncertainties=[20.0, 10.0],
    ...     analysis="example",
    ...     xsection=0.123,
    ... )

**Arguments:**

 * ``signal_yields``: keyword for signal yields. It can take one or more values as a list or NumPy array.
 * ``background_yields``: keyword for background-only expectations. It can take one or more values as a list or NumPy array.
 * ``data``: keyword for observations. It can take one or more values as a list or NumPy array.
 * ``absolute_uncertainties``: absolute uncertainties on the background
 * ``analysis`` (optional): Unique identifier for the analysis.
 * ``xsection`` (optional): Cross-section value for the signal hypothesis. Units determined by the user.


.. _multinormal:

``'default.multivariate_normal'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multivariate Gaussian likelihood that accounts for correlations between bin measurements. This is the proper
extension of the univariate `normal` backend when bin observations are correlated.

.. math::

    \mathcal{L}(\mu) = \frac{1}{\sqrt{(2\pi)^k {\rm det}[\Sigma] }} \exp\left[-\frac{1}{2} (\mu \mathbf{n}_s + \mathbf{n}_b - \mathbf{n})^T \Sigma^{-1} (\mu \mathbf{n}_s + \mathbf{n}_b - \mathbf{n}) \right]

**Where:**
- :math:`k` is the number of bins
- :math:`\mathbf{n}_s, \mathbf{n}_b, \mathbf{n}` are vectors of signal yields, background yields, and observations
- :math:`\Sigma` is the :math:`k \times k` covariance matrix encoding both bin variances (diagonal) and correlations (off-diagonal elements)

**When to use:**
This model is appropriate when:
- Bin measurements are correlated (e.g., from a simultaneous fit to multiple channels or detector regions)
- The uncertainty structure cannot be factorised into independent components
- You have a full covariance matrix from your analysis

**Mathematical details:**
The joint distribution is a multivariate Gaussian with:
- Mean: :math:`\mu \mathbf{n}_s + \mathbf{n}_b`
- Covariance: :math:`\Sigma`

The log-likelihood involves inverting the covariance matrix and computing the Mahalanobis distance.
It can accommodate any number of yields, though the inversion of :math:`\Sigma` becomes numerically expensive for very high dimensions.

.. code-block:: python3
    :linenos:

    >>> pdf_wrapper = spey.get_backend("default.multivariate_normal")
    >>> statistical_model = pdf_wrapper(
    ...     signal_yields=[12.0, 15.0],
    ...     background_yields=[50.0,48.0],
    ...     data=[36, 33],
    ...     covariance_matrix=[[144.0,13.0], [25.0, 256.0]],
    ...     analysis="example",
    ...     xsection=0.123,
    ... )

**Arguments:**

 * ``signal_yields``: keyword for signal yields. It can take one or more values as a list or NumPy array.
 * ``background_yields``: keyword for background-only expectations. It can take one or more values as a list or NumPy array.
 * ``data``: keyword for observations. It can take one or more values as a list or NumPy array.
 * ``covariance_matrix``: covariance matrix (square matrix)
 * ``analysis`` (optional): Unique identifier for the analysis.
 * ``xsection`` (optional): Cross-section value for the signal hypothesis. Units determined by the user.



External Plug-ins
-----------------

.. toctree::

   Spey-pyhf plug-in documentation <https://spey-pyhf.readthedocs.io>
   Spey-strathisla <https://pypi.org/project/strathisla/>
   Spey-fastprof: TBA <https://spey.readthedocs.io>

**Useful links:**

* `pyhf documentation <https://pyhf.readthedocs.io>`_ :cite:`pyhf_joss`.
* `Spey-pyhf plug-in GitHub Repository <https://github.com/SpeysideHEP/spey-pyhf>`_
* `Spey-strathisla GitHub Repository <https://github.com/joes-git/spey-strathisla>`_: Provides external plugins suitable for a. full modelling of nuisance parameters and b. setting limits on Effective Field Theories.
* ``spey-fastprof``: TBA :cite:`Berger:2023bat`.
* `fastprof documentation <https://fastprof.web.cern.ch/fastprof/>`_.
