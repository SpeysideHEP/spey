.. _sec:plugins:

Plug-ins
========

``spey`` works with various packages that are designed to deliver certain statistical model
prescription. The goal of the ``spey`` interface is to collect all these prescriptions under
the same roof and provide a toolset to combine different sources of likelihoods. 
In this section we will summarise currently available plugins 
which are accessible throghy `spey` interface. The string based accessors
to the available plugins can be seen using the following command:

.. code-block:: python3

    >>> spey.AvailableBackends()
    >>> # ['default_pdf.correlated_background',
    >>> #  'default_pdf.effective_sigma',
    >>> #  'default_pdf.third_moment_expansion',
    >>> #  'default_pdf.uncorrelated_background']

where once installed wihout any plug-ins :func:`~spey.AvailableBackends` function
only shows the default PDFs. In the following section we will summarize their usability.
Once we know the accessor of the plug-in it can be called using :func:`~spey.get_backend` 
function e.g.

.. code-block:: python3

    >>> pdf_wrapper = spey.get_backend('default_pdf.uncorrelated_background')

this will automatically create a wrapper around the likelihood prescription and allow `spey`
to use it properly. We will demonstrate the usage for each of the default plugins below.

.. note:: 

    Documentation of each plug-in has been included within the ``pdf_wrapper`` documentation.
    Hence, if one types ``pdf_wrapper?`` in ipython commandline or in a jupyter notebook it is
    possible to access the extended documentation for both the wrapper and the backend in use.

.. attention:: 

    :func:`~spey.get_backend` function is a wrapper between PDF prescription and ``spey`` package.
    Once initialised, all PDF prescriptions are defined with :obj:`~spey.StatisticalModel` class 
    which provides a backend agnostic interface i.e. all PDF prescriptions will have same functionality.

Default Plug-ins
----------------
All default plug-ins are defined using the following main likelihood structure

.. math:: 

    \mathcal{L}(\mu,\theta) = \underbrace{\prod_{i\in{\rm bins}} 
    \mathcal{M}(n_i|\lambda_i(\mu, \theta))}_{\rm main}\cdot 
    \underbrace{\prod_{j\in{\rm nui}}\mathcal{C}(\theta_j)}_{\rm constraint} \ ,

where first term represents the main model and the second term represents the constraint model.

``'default_pdf.uncorrelated_background'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a basic PDF where the background is assumed to be not correlated, where the PDF has been 
given as 

.. math:: 

    \mathcal{L}(\mu, \theta) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + n_b^i + 
    \theta^i\sigma_b^i) \cdot \prod_{j\in{\rm nui}}\mathcal{N}(\theta^j|0, 1)\ ,

where :math:`\mu, \theta` are the parameter of interest (signal strength) and nuisance parameters, 
the signal and background yields are given as :math:`n_s^i` and :math:`n_b^i\pm\sigma_b^i` respectively.
Additinally absolute uncertainties are modelled as Gaussian distributions. This model can be 
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

This particular example implements a two-bin histogram with uncorrelated bins. The exclusion CL 
(:math:`1-CL_s`) can be computed via :func:`spey.StatisticalModel.exclusion_confidence_level` function.

.. code-block:: python3

    >>> statistical_model.exclusion_confidence_level()
    >>> # [0.9701795436411219]

For all the properties of :obj:`~spey.StatisticalModel` class we refer the reader to the corresponding
API description.

``'default_pdf.correlated_background'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This plugin embeds the correlations between each bin using a covariance matrix provided by the user
which employs the following PDF structure

.. math:: 

    \mathcal{L}(\mu, \theta) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + n_b^i + 
    \theta^i\sigma_b^i) \cdot \prod_{j\in{\rm nui}}\mathcal{N}(\theta^j|0, \rho)\ ,

Notice that the only difference between the uncorrelated case is the constraint term which includes
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

As can be seen from the two results, correlation between histogram bins reduces the exclusion limit
as expected.

``'default_pdf.third_moment_expansion'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This plug-in implements the third moment expansion presented in :cite:`Buckley:2018vdr` which expands the 
main model using the diagonal elements of the third moments

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

As can be seen from the result, slight skewness generated by the third moments presented in the function
reduced the exclusion limit.


``'default_pdf.effective_sigma'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The skewness of the PDF distribution can also be captured by building an effective variance 
from the upper (:math:`\sigma^+`) and lower (:math:`\sigma^-`) uncertainty envelops as a 
function of nuisance parameters,

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

where ``absolute_uncertainty_envelops`` refers to the upper and lower unvertainty envelops for each bin.
Once again the exclusion limit can be computed as 

.. code-block:: python3

    >>> statistical_model.exclusion_confidence_level()
    >>> # [0.8567802529243093]

External Plug-ins
-----------------

    * ``spey-pyhf`` plugin allows pyhf's likelihood prescription to be used within ``spey``.
      for details see the `dedicated documentation <https://github.com/SpeysideHEP/spey-pyhf>`_ 
      :cite:`pyhf_joss`.
        
        * `pyhf documentation <https://pyhf.readthedocs.io>`_.
    
    * ``spey-fastprof``: TBA :cite:`Berger:2023bat`.

        * `fastprof documentation <https://fastprof.web.cern.ch/fastprof/>`_.