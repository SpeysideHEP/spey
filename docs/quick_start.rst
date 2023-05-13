.. _sec:installation:

Installation
============

``spey`` is available at `pypi <https://pypi.org>`_ , so it can be installed by running:

.. code-block:: bash

    >>> pip install spey


Python >=3.8 is required. spey heavily relies on `numpy <https://numpy.org/doc/stable/>`_, 
`scipy <https://docs.scipy.org/doc/scipy/>`_ and `autograd <https://github.com/HIPS/autograd>`_ 
which are all packaged during the installation with the necessary versions. Note that some 
versions may be restricted due to numeric stability and validation.

What is Spey?
-------------

Spey is a plug-in based statistics tool which aims to collect all likelihood prescriptions 
under one roof. This provides user the workspace to freely combine different statistical models 
and study them through a single interface. In order to achieve a module that can be used both 
with statistical model prescriptions which has been proposed in the past and will be used in the
future, Spey uses so-called plug-in system where developers can propose their own statistical 
model prescription and allow spey to use them.

.. _sec:plugins:

Plugins
-------

``spey`` works with various packages that are designed to deliver certain statistical model
prescription. The goal of the ``spey`` interface is to collect all these prescriptions under
the same roof and provide a toolset to combine different sources of likelihoods. List of plugins
are as follows;

Default plug-ins
~~~~~~~~~~~~~~~~

* ``'simplified_likelihoods'``: Main simplified likelihood backend which uses a Multivariate 
  Normal and a Poisson distributions to construct log-probability of the statistical model. 
  The Multivariate Normal distribution is constructed by the help of a covariance matrix 
  provided by the user which captures the uncertainties and background correlations between 
  each histogram bin. This statistical model has been first proposed in 
  `JHEP 04 (2019), 064 <https://doi.org/10.1007/JHEP04%282019%29064>`_. The probability 
  distribution of a simplified likelihood can be formed as follows;

  .. math:: 

        \mathcal{L}(\mu,\theta) = \underbrace{\left[\prod_i^N {\rm Poiss}\left(n^i_{obs} | \lambda_i(\mu, \theta)\right) \right]}_{\rm main\ model}
        \cdot \underbrace{\mathcal{N}(\theta | 0, \Sigma)}_{\rm constraint\ model}

  Here the first term is the so-called main model based on Poisson distribution centred around 
  :math:`\lambda_i(\mu, \theta) = \mu n^i_{sig} + \theta + n^i_{bkg}` and the second term is the 
  multivariate normal distribution centred around zero with the standard deviation of :math:`\Sigma`
  which, for multi-modal input, is covariance matrix.

* ``'simplified_likelihoods.third_moment_expansion'``: Third moment expansion follows the above 
  simplified likelihood construction and modifies the :math:`\lambda` and :math:`\Sigma`. 
  Using the expected background yields, :math:`m^{(1)}_i`, diagonal elements of the third moments, 
  :math:`m^{(3)}_i` and the covariance matrix, :math:`m^{(2)}_{ij}`, one can write a modified 
  correlation matrix and :math:`\lambda` function as follows

  .. math:: 

        C_i &= -sign(m^{(3)}_i) \sqrt{2 m^{(2)}_{ii}} \cos\left( \frac{4\pi}{3} + \frac{1}{3}\arctan\left(\sqrt{ \frac{8(m^{(2)}_{ii})^3}{(m^{(3)}_i)^2} - 1}\right) \right)
        
        B_i &= \sqrt{m^{(2)}_{ii} - 2 C_i^2}

        A_i &=  m^{(1)}_i - C_i

        \rho_{ij} &= \frac{1}{4C_iC_j} \left( \sqrt{(B_iB_j)^2 + 8C_iC_jm^{(2)}_{ij}} - B_iB_j \right)

  which further modifies :math:`\lambda_i(\mu, \theta) = \mu n^i_{sig} + A_i + B_i \theta_i + C_i \theta_i^2`
  and the multivariate normal has been modified via the inverse of the correlation matrix, 
  :math:`\mathcal{N}(\theta | 0, \rho^{-1})`. See `JHEP 04 (2019), 064 <https://doi.org/10.1007/JHEP04%282019%29064>`_
  Sec. 2 for details.

* ``'simplified_likelihoods.uncorrelated_background'``: User can use multi or single bin histograms 
  with unknown correlation structure within simplified likelihood interface. This particular 
  plug-in replaces Multivariate Normal distribution of the likelihood with a simple Normal 
  distribution to reduce the computational cost.

  .. math:: 

        \mathcal{L}_{\rm constraint}(\theta) = \prod_i^N \mathcal{N}(\theta_i | 0, \sigma_i)

* ``'simplified_likelihoods.variable_gaussian'``: Variable Gaussian method is designed to capture 
  asymetric uncertainties on the background yields. This method converts the covariance matrix in 
  to a function which takes absolute upper (:math:`\sigma^+`) and lower (:math:`\sigma^-`) envelops of the 
  background uncertainties, best fit values (:math:`\hat\theta`) and nuisance parameters 
  (:math:`\theta`) which allows the interface dynamically change the covariance 
  matrix with respect to given nuisance parameters. This implementation follows the method 
  proposed in `Ref. arXiv:physics/0406120 <https://arxiv.org/abs/physics/0406120>`_. This approach
  transforms the covariance matrix from a constant input to a function of nuisance parameters.

  .. math:: 

      \sigma^\prime &= \sqrt{\sigma^+\sigma^-  + (\sigma^+ - \sigma^-)(\theta - \hat\theta)}
      
      \Sigma(\theta) &= \sigma^\prime \otimes \rho \otimes \sigma^\prime

  which further modifies the multivariate normal distribution this new covariance matrix
  :math:`\mathcal{N}(\theta | 0, \Sigma) \to \mathcal{N}(\theta | 0, \Sigma(\theta))`. 

Third-party plug-ins
~~~~~~~~~~~~~~~~~~~~

* `spey-pyhf <https://github.com/SpeysideHEP/spey-pyhf>`_ : enables the usage of :xref:`pyhf` 
  package through ``spey`` interface. For the documentation and installation please see 
  `this link <https://github.com/SpeysideHEP/spey-pyhf>`_.

* ``spey-fastprof`` : enables the usage of ``fastprof`` through ``spey`` interface. For the 
  documentation and installation please see `this link <https://github.com/SpeysideHEP/spey-pyhf>`_.

.. _sec:quick_start:

Quick Start
===========

First one needs to choose which backend to work with. By default, spey is shipped with various types of 
`simplified_likelihood` backend which can be checked via :func:`~spey.AvailableBackends` function

.. code:: python

    >>> import spey
    >>> print(spey.AvailableBackends())
    >>> # ['simplified_likelihoods', 
    ... #  'simplified_likelihoods.third_moment_expansion', 
    ... #  'simplified_likelihoods.uncorrelated_background', 
    ... #  'simplified_likelihoods.variable_gaussian']

Using ``'simplified_likelihoods.uncorrelated_background'`` one can simply create single or multi-bin
statistical models:

.. code:: python

    >>> stat_wrapper = spey.get_backend('simplified_likelihoods.uncorrelated_background')

    >>> data = [1]
    >>> signal_yields = [0.5]
    >>> background_yields = [2.0]
    >>> background_unc = [1.1]

    >>> stat_model = stat_wrapper(
    ...     signal_yields, background_yields, data, background_unc, analysis="single_bin", xsection=0.123
    ... )

where ``data`` indicates the observed events, ``signal_yields`` and ``background_yields`` represents
yields for signal and background samples and ``background_unc`` shows the absolute uncertainties on 
the background events i.e. :math:`2.0\pm1.1` in this particular case. Note that we also introduced 
``analysis`` and ``xsection`` information which are optional where the ``analysis`` indicates a unique
identifier for the statistical model and ``xsection`` is the cross-section value of the signal which is
only used for the computation of the excluded cross section value.

During computation of any probability distribution Spey relies on so-called "expectation type". 
This can be set via :obj:`~spey.ExpectationType` which includes three different expectation mode.

* :obj:`~spey.ExpectationType.observed` : Indicates that the computation of the log-probability will be 
  achieved by fitting the statistical model on the experimental data. For the exclusion limit computation
  this will tell package to compute observed :math:`1-CL_s` values. :obj:`~spey.ExpectationType.observed`
  has been set as default through out the package.

* :obj:`~spey.ExpectationType.aposteriori`: This command will result with the same log-probability computation
  as :obj:`~spey.ExpectationType.observed`. However, expected exclusion limit will be computed by centralising
  the statistical model on the background and checking :math:`\pm1\sigma` and :math:`\pm2\sigma` fluctuations.

* :obj:`~spey.ExpectationType.apriori` : Indicates that the obseravation has never take place and the theoretical
  SM computation is the absolute truth. Thus it replaces observed values in the statistical model with the 
  background values and computes the log-probability accordingly. Similar to :obj:`~spey.ExpectationType.aposteriori`
  exclusion limit computation will return expected limits.

To compute the observed exclusion limit for the above example one can type

.. code:: python

    >>> for expectation in spey.ExpectationType:
    >>>     print(f"1-CLs ({expectation}): {stat_model.exclusion_confidence_level(expected=expectation)}")
    >>> # 1-CLs (apriori): [0.48980408984423207, 0.35671028499361224, 0.21275777462774292, 0.17543303294266588, 0.17543303294266588]
    >>> # 1-CLs (aposteriori): [0.6959976874809755, 0.5466491036450178, 0.3556261845401908, 0.2623335168616665, 0.2623335168616665]
    >>> # 1-CLs (observed): [0.40145846656558726]

Note that :obj:`~spey.ExpectationType.apriori` and :obj:`~spey.ExpectationType.aposteriori` expectation types 
resulted in a list of 5 elements which indicates :math:`-2\sigma,\ -1\sigma,\ 0,\ +1\sigma,\ +2\sigma` standard deviations.
:obj:`~spey.ExpectationType.observed` on the other hand resulted in single value which is observed exclusion limit.
Notice that the bounds on :obj:`~spey.ExpectationType.aposteriori` are slightly stronger than :obj:`~spey.ExpectationType.apriori`
this is due to the data value has been replaced with background yields, which is larger than the observations. 
:obj:`~spey.ExpectationType.apriori` is mostly used in theory collaborations to estimate the difference from the Standard Model
rather than the experimental observations.

One can play the same game using the same backend for multi-bin histograms as follows;

.. code:: python

    >>> stat_wrapper = spey.get_backend('simplified_likelihoods.uncorrelated_background')

    >>> data = [1, 3]
    >>> signal = [0.5, 2.0]
    >>> background = [2.0, 2.8]
    >>> background_unc = [1.1, 0.8]

    >>> stat_model = stat_wrapper(
    ...     signal, background, data, background_unc, analysis="multi-bin", xsection=0.123
    ... )

Note that our statistical model still represents individual bins of the histograms independently however it sums up the 
log-likelihood of each bin. Hence all bins are completely uncorrelated from each other. Computing the exclusion limits
for each :obj:`~spey.ExpectationType` will yield

.. code:: python

    >>> for expectation in spey.ExpectationType:
    >>>     print(f"1-CLs ({expectation}): {stat_model.exclusion_confidence_level(expected=expectation)}")
    >>> # 1-CLs (apriori): [0.9357315808495567, 0.8480953812080605, 0.6707336318388715, 0.40146054347432814, 0.40146054347432814]
    >>> # 1-CLs (aposteriori): [0.945840731123488, 0.8657740143137352, 0.6959070047129498, 0.41884413918205454, 0.41034502645428916]
    >>> # 1-CLs (observed): [0.7016751631249967]

It is also possible to compute :math:`1-CL_s` value with respect to the parameter of interest, :math:`\mu`.
This can be achieved by including a value for ``poi_test`` argument

.. code:: python
    :linenos:

    >>> poiUL = np.array([stat_model.exclusion_confidence_level(poi_test=p, expected=spey.ExpectationType.aposteriori) for p in np.linspace(1,5,20)])
    >>> plt.plot(np.linspace(1,5,20), poiUL[:,2], color="tab:red")
    >>> plt.fill_between(np.linspace(1,5,20), poiUL[:,1], poiUL[:,3], alpha=0.8, color="green", lw=0)
    >>> plt.fill_between(np.linspace(1,5,20), poiUL[:,0], poiUL[:,4], alpha=0.5, color="yellow", lw=0)
    >>> plt.plot([1,5], [.95,.95], color="k", ls="dashed")
    >>> plt.xlabel("$\mu$")
    >>> plt.ylabel("$1-CL_s$")
    >>> plt.xlim([1,5])
    >>> plt.ylim([.4,1.01])
    >>> plt.text(4,0.9, r"$95\%\ {\rm CL}$")
    >>> plt.show()

Here in the first line we extract :math:`1-CL_s` values per POI for :obj:`~spey.ExpectationType.aposteriori` 
expectation type and we plot specific standard deviations which provides following plot:

.. image:: ./figs/brazilian_plot.png
    :align: center
    :scale: 70
    :alt: Exclusion confidence level with respect to parameter of interest, :math:`\mu`.

The excluded value of POI can also be retreived by :func:`~spey.StatisticalModel.poi_upper_limit` function

.. code:: python

    >>> print("POI UL: %.3f" % stat_model.poi_upper_limit(expected=spey.ExpectationType.aposteriori))
    >>> # POI UL: 2.201

which is exact point where red-curve and black dashed line meets. The upper limit for the :math:`\pm1\sigma`
or :math:`\pm2\sigma` bands can be extracted by setting ``expected_pvalue`` to ``"1sigma"`` or ``"2sigma"`` 
respectively, e.g.

.. code:: python

    >>> stat_model.poi_upper_limit(expected=spey.ExpectationType.aposteriori, expected_pvalue="1sigma")
    >>> # [1.4633382034219111, 2.2009296966966683, 3.3921192489003325]

At a more lower level, one can extract the likelihood information for the statistical model by calling 
:func:`~spey.StatisticalModel.likelihood` and :func:`~spey.StatisticalModel.maximize_likelihood` functions.
By default these will return negative log-likelihood values but this can be changed via ``return_nll=False``
argument. 

.. code:: python
    :linenos:

    >>> muhat_obs, maxllhd_obs = stat_model.maximize_likelihood(return_nll=False, )
    >>> muhat_apri, maxllhd_apri = stat_model.maximize_likelihood(return_nll=False, expected=spey.ExpectationType.apriori)

    >>> poi = np.linspace(-1.4,2.2,15)

    >>> llhd_obs = np.array([stat_model.likelihood(p, return_nll=False) for p in poi])
    >>> llhd_apri = np.array([stat_model.likelihood(p, expected=spey.ExpectationType.apriori, return_nll=False) for p in poi])

Here in first two lines we extracted maximum likelihood and the POI value that maximizes the likelihood for two different
expectation type. In the following we computed likelihood distribution for various POI values which then can be plotted
as follows

.. code:: python

    >>> plt.plot(poi, llhd_obs, label=r"${\rm observed}$")
    >>> plt.plot(poi, llhd_apri, label=r"${\rm apriori}$")
    >>> plt.scatter(muhat_obs, maxllhd_obs)
    >>> plt.scatter(muhat_apri, maxllhd_apri)

.. image:: ./figs/multi_bin_llhd.png
    :align: center
    :scale: 70
    :alt: Likelihood distribution for multi-bin statistical model.

Notice the slight difference between likelihood distributions, this is because of the use of different expectation types.
The dots on the likelihood distribution represents the point where likelihood is maximized. Since for an 
:obj:`~spey.ExpectationType.apriori` likelihood distribution observed and background values are the same, the likelihood
should peak at :math:`\mu=0`.