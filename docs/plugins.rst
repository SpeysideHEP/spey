.. _sec:plugins:

Plug-ins
========

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
    >>> # ['default_pdf.correlated_background',
    >>> #  'default_pdf.effective_sigma',
    >>> #  'default_pdf.third_moment_expansion',
    >>> #  'default_pdf.uncorrelated_background']

where once installed without any plug-ins :func:`~spey.AvailableBackends` function
only shows the default PDFs. In the following section, we will summarise their usability.
Once we know the accessor of the plug-in, it can be called using :func:`~spey.get_backend` 
function e.g.

.. code-block:: python3

    >>> pdf_wrapper = spey.get_backend('default_pdf.uncorrelated_background')

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

The first term represents the primary model, and the second represents the constraint model.

``'default_pdf.uncorrelated_background'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a basic PDF where the background is assumed to be not correlated, where the PDF has been 
given as 

.. math:: 

    \mathcal{L}(\mu, \theta) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + n_b^i + 
    \theta^i\sigma_b^i) \cdot \prod_{j\in{\rm nui}}\mathcal{N}(\theta^j|0, 1)\ ,

where :math:`\mu, \theta` are the parameter of interest (signal strength) and nuisance parameters, 
the signal and background yields are given as :math:`n_s^i` and :math:`n_b^i\pm\sigma_b^i` respectively.
Additionally, absolute uncertainties are modeled as Gaussian distributions. This model can be 
used as follows

.. code-block:: python3
    :linenos:

    >>> pdf_wrapper = spey.get_backend("default_pdf.uncorrelated_background")
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

``'default_pdf.correlated_background'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This plugin embeds the correlations between each bin using a covariance matrix provided by the user
which employs the following PDF structure

.. math:: 

    \mathcal{L}(\mu, \theta) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + n_b^i + 
    \theta^i\sigma_b^i) \cdot \prod_{j\in{\rm nui}}\mathcal{N}(\theta^j|0, \rho)\ ,

Notice that the only difference between the uncorrelated case is the constraint term, which includes
correlations between each bin. Iterating on the same example, a correlated two-bin histogram can be
defined as

.. code-block:: python3
    :linenos:

    >>> pdf_wrapper = spey.get_backend("default_pdf.correlated_background")
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

``'default_pdf.third_moment_expansion'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This plug-in implements the third-moment expansion presented in :cite:`Buckley:2018vdr`, which expands the 
the main model using the diagonal elements of the third moments

.. math:: 

    \mathcal{L}(\mu, \theta) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + \bar{n}_b^i + B_i\theta_i + S_i\theta_i^2)
     \cdot \prod_{j\in{\rm nui}}\mathcal{N}(\theta^j|0, \rho)\ ,

where :math:`\bar{n}_b^i,\ B_i,\ S_i` and :math:`\rho` are defined as

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

    >>> pdf_wrapper = spey.get_backend("default_pdf.third_moment_expansion")
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
   :func:`~spey.backends.default_pdf.third_moment.compute_third_moments` function; however this function computes third moments using
   Bifurcated Gaussian, which may not be suitable for every case.
 * ``analysis`` (optional): Unique identifier for the analysis.
 * ``xsection`` (optional): Cross-section value for the signal hypothesis. Units determined by the user.


``'default_pdf.effective_sigma'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The skewness of the PDF distribution can also be captured by building an effective variance 
from the upper (:math:`\sigma^+`) and lower (:math:`\sigma^-`) uncertainty envelops as a 
the function of nuisance parameters,

.. math:: 

    \sigma_{\rm eff}^i(\theta^i) = \sqrt{\sigma^+_i\sigma^-_i + (\sigma^+_i - \sigma^-_i)(\theta^i - n_b^i)}

This method has been proposed in :cite:`Barlow:2004wg` for Gaussian models which can be 
generalised for the Poisson distribution by modifying :math:`\lambda_i(\mu, \theta)` as follows

.. math:: 

    \mathcal{L}(\mu, \theta) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + n_b^i + \theta^i\sigma_{\rm eff}^i(\theta^i))
     \cdot \prod_{j\in{\rm nui}}\mathcal{N}(\theta^j|0, \rho)\ ,

iterating over the same example, this PDF can be utilised as follows;

.. code-block:: python3
    :linenos:

    >>> pdf_wrapper = spey.get_backend("default_pdf.effective_sigma")
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

``'default_pdf.poisson'``
~~~~~~~~~~~~~~~~~~~~~~~~~

Simple Poisson implementation without uncertainties which can be described as follows;

.. math::

    \mathcal{L}(\mu) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + n_b^i)

It can take any number of yields.

.. code-block:: python3
    :linenos:

    >>> pdf_wrapper = spey.get_backend("default_pdf.poisson")
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


External Plug-ins
-----------------

.. toctree::

   Spey-pyhf plug-in documentation <https://spey-pyhf.readthedocs.io>
   Spey-fastprof: TBA <https://spey.readthedocs.io>

**Useful links:**

* `pyhf documentation <https://pyhf.readthedocs.io>`_ :cite:`pyhf_joss`.
* `Spey-pyhf plug-in GitHub Repository <https://github.com/SpeysideHEP/spey-pyhf>`_
* ``spey-fastprof``: TBA :cite:`Berger:2023bat`.
* `fastprof documentation <https://fastprof.web.cern.ch/fastprof/>`_.