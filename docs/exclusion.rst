Exclusion limits
================

Any statistical model in Spey comes with a posibility to compute exclusion confidence level
using three different option. Depending on the available functions in likelihood construction 
(see :ref:`this section <sec_new_plugin>` for details) one or more of these options will be 
available for the user. To see which calculators are available, one can use 
:func:`~spey.StatisticalModel.available_calculators` function.

:func:`~spey.StatisticalModel.exclusion_confidence_level` function uses ``calculator`` keyword
to choose in between ``"asymptotic"``, ``"toy"`` and ``"chi_square"`` calculators.

* ``"asymptotic"``: uses asymptotic formulae to compute p-values, see ref. :cite:`Cowan:2010js` 
  for details. This method is only available when if the likelihood construction has access to 
  expected values of the distribution, which allows one to construct Asimov data.
* ``"toy"``: This method uses the sampling functionality of the likelihood, hence expects the 
  construction to have sampling abilities. It computes p-values by sampling from signal+background
  and background only distributions.
* ``"chi_square"``: This method simply computes

  .. math::

        \chi^2(\mu) = -2 \log\frac{\mathcal{L}(\mu, \theta_\mu)}{\mathcal{L}(0,\theta_0)}
    

  and uses :math:`\chi^2`-p-value look-up tables to determine the exclusion limits. Here :math:`\mu`
  is determined by ``poi_test`` keyword, which is by default ``1.0``.

The ``expected`` keyword allows user to choose between observed and expected exclusion limit 
computations. Additionally it allows one to choose prefit expected exclusion limit as well, 
this can be enabled by choosing ``expected=spey.ExpectationType.apriori``. This option will 
disregard the experimental observations and compute the expected exclusion limit with respect to
the expected background yields i.e. simulated SM. :attr:`~spey.ExpectationType.observed` 
(:attr:`~spey.ExpectationType.aposteriori`) performs post-fit and computes observed (expected) 
exclusion confidence limits. Expected exclusion limits, for both cases, are returned with 
:math:`\pm1\sigma` and :math:`\pm2\sigma` fluctuations around the background model. Hence while 
the observe expectation limit returns one value, expected exclusion limit returns five values; 
:math:`[-2\sigma, -1\sigma, 0, 1\sigma, 2\sigma]`, respectively. However since this is not possible
for ``"chi_square"`` calculator, that option only returns one value for both.

``allow_negative_signal`` determines which test statistics to be used and limits the values that :math:`\mu`
can take during the computation of maximum likelihood. If ``allow_negative_signal=True`` algorithm
will use :math:`q_\mu` test statistic, otherwise :math:`\tilde{q}_\mu` test statistics will be used 
(see :cite:`Cowan:2010js, Araz:2023bwx` for details).

For highly complex statistical models maximising the likelihood can be tricky and might depend on the optimiser.
Spey uses Scipy to handle the optimisation and fitting tasks, all other keyword arguments that has not been mentioned
in the function descrition of :func:`~spey.StatisticalModel.exclusion_confidence_level` passed to the optimiser which
enables user to control the properties of the optimiser through the interface.