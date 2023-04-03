.. _sec:new_plugin:

Building a plugin
=================

``spey`` package has been designed to be epandable. It only needs to know certain aspects of the 
data structure that is presented and a prescription to form a likelihood function.

What a plugin provides
----------------------

A quick intro on terminology of spey plugins in this section:

  * A plugin is an external Python package that provides additional statistical model 
    prescriptions to spey.
  * Each plugin may provide one (or more) statistical model prescription, that are 
    accessible directly through spey.
  * Depending on the scope of the plugin, you may wish to provide additional (custom) 
    operations and differentiability through various autodif packages such as ``autograd``
    or ``jax``. As long as they are implemented through set of predefined function names
    spey can automatically detect and use them within the interface. 

Creating your statistical model prescription
--------------------------------------------

The first step in creating your own spey plugin is to create your statistical model interface. 
This is as simple as importing abstract base class :class:`~spey.BackendBase` from spey and 
inheriting it. The most basic implementation of a statistical model can be found below;

.. _MyStatisticalModel:
.. code-block:: python3
    :linenos:

    >>> import spey

    >>> class MyStatisticalModel(spey.BackendBase):
    >>>     name = "my_stat_model"
    >>>     version = "1.0.0"
    >>>     author = "John Smith <john.smith@smith.com>"
    >>>     spey_requires = ">=0.1.0,<0.2.0"
    
    >>>     def config(
    ...         self, allow_negative_signal: bool = True, poi_upper_bound: float = 10.0
    ...     ):
    >>>         ...

    >>>     def get_logpdf_func(
    ...         self, expected = spey.ExpectationType.observed, data = None
    ...     ):
    >>>         ...

    >>>     def get_objective_function(
    ...         self,
    ...         expected = spey.ExpectationType.observed,
    ...         data = None,
    ...         do_grad = True,
    ...     ):
    >>>         ...
    
    >>>     def generate_asimov_data(
    ...          self,
    ...          poi_asimov = 0.0,
    ...          expected = spey.ExpectationType.observed,
    ...          **kwargs,
    ...      ):
    >>>         ...

:class:`~spey.BackendBase` requires certain functionality from the statistical model to be 
implemented but let us first go through the above class structure. spey looks for certain 
metadata in order to track the version, author and name of the implementation. Additionally 
it checks compatibility with current spey version to ensure that the plugin works as it should.

.. note:: 

    The list of metadata that spey is looking for:

      * **name** (``str``): Name of the plugin.
      * **version** (``str``): Version of the plugin.
      * **author** (``str``): Author of the plugin.
      * **spey_requires** (``str``, *required*): The minimum spey version that the 
        plugin is built e.g. ``spey_requires="0.0.1"`` or ``spey_requires=">=0.3.3"``.
      * **doi** (``List[str]``): Citable DOI numbers for the plugin.
      * **arXiv** (``List[str]``): arXiv numbers for the plugin.

`MyStatisticalModel`_ class has three main functionalities namely :func:`~spey.BackendBase.config`, 
:func:`~spey.BackendBase.get_logpdf_func` and :func:`~spey.BackendBase.get_objective_function` 
(for detailed descriptions of these functions please go to the :class:`~spey.BackendBase` documentation
by clicking on them.)

* :func:`~spey.BackendBase.config`: This function returns :class:`~spey.base.model_config.ModelConfig` class
  which includes certain information about the model structure such as index of the parameter of interest 
  within the parameter list (:attr:`~spey.base.model_config.ModelConfig.poi_index`), minimum value parameter 
  of interest can take (:attr:`~spey.base.model_config.ModelConfig.minimum_poi`), suggested initialisation
  parameters for the optimiser (:attr:`~spey.base.model_config.ModelConfig.suggested_init`) and suggested 
  bounds for the parameters (:attr:`~spey.base.model_config.ModelConfig.suggested_bounds`). If 
  ``allow_negative_signal=True`` the lower bound of POI is expected to be zero, if ``False`` 
  :attr:`~spey.base.model_config.ModelConfig.minimum_poi`. ``poi_upper_bound`` is used to enforce an upper 
  bound on POI.

  .. note:: 

    suggested bounds and initialisation values should return a list with a length of number of nuissance 
    parameters and parameter of interest. Initialisation values should be a type of ``List[float, ...]`` 
    and bounds should have the type of ``List[Tuple[float, float], ...]``.

* :func:`~spey.BackendBase.get_logpdf_func`: This function returns a function that takes a NumPy array 
  as an input which indicates the fit parameters (nuisance and POI) and returns the value of natural logarithm
  of the likelihood function, :math:`\log\mathcal{L}(\mu, \theta)`. The input ``expected`` defines which data to be 
  used in the absence of ``data`` input i.e. if ``expected=spey.ExpectationType.observed`` yields of observed data 
  should be used to compute the likelihood but if ``expected=spey.ExpectationType.apriori`` background yields should
  be used. This ensures the difference between prefit and postfit likelihoods. If ``data`` is provided, it is 
  it is overwritten, this is for the case where Asimov data is in use.

* :func:`~spey.BackendBase.get_objective_function`: This function is crutial for the optimisation procedure. If 
  ``do_grad=True`` it is typically a function of :math:`-\log\mathcal{L}(\mu,\theta)` and its gradient 
  with respect to :math:`\mu` and :math:`\theta` where if ``do_grad=False`` it only returns a function of 
  :math:`-\log\mathcal{L}(\mu,\theta)`. Note that it can also return any function of the likelihood for 
  optimisation purposes, the likelihood is computed from :func:`~spey.BackendBase.get_logpdf_func` using the fit 
  parameters obtained during the optimisation. Similar to :func:`~spey.BackendBase.get_logpdf_func`, the input 
  ``expected`` defines which data to be used in the absence of ``data`` input i.e. if 
  ``expected=spey.ExpectationType.observed`` yields of observed data should be used to compute the likelihood 
  but if ``expected=spey.ExpectationType.apriori`` background yields should be used. This ensures the difference 
  between prefit and postfit likelihoods. If ``data`` is provided, it is it is overwritten, this is for the case 
  where Asimov data is in use.

  .. note::

    If gradient is not available, in case of ``do_grad=True`` this function should raise 
    :obj:`NotImplementedError` so that spey can autimatically switch to ``do_grad=False`` mode.

* :func:`~spey.BackendBase.generate_asimov_data`: This function is crutial for asymptotic hypothesis testing.
  It needs to generate Asimov data with respect to the given ``poi_asimov``, :math:`\mu_A`. As before the input 
  ``expected`` defines which data to be used in the absence of ``data`` input i.e. if 
  ``expected=spey.ExpectationType.observed`` yields of observed data should be used to compute the likelihood 
  but if ``expected=spey.ExpectationType.apriori`` background yields should be used. This ensures the difference 
  between prefit and postfit likelihoods.

Beyond the basic functionality spey also allows integration of more complex likelihood computations to be held. Prior
to calling :func:`~spey.BackendBase.get_objective_function` or :func:`~spey.BackendBase.generate_asimov_data` spey looks
for specific implementations such as :func:`~spey.BackendBase.negative_loglikelihood` or 
:func:`~spey.BackendBase.asimov_negative_loglikelihood`. If these functions are provided in the backend spey will directly
use those instead. The list of these functions can be found below and interested user can check their documentation by
clicking on the functions;

.. hlist:: 
    :columns: 2

    * :func:`~spey.BackendBase.negative_loglikelihood`
    * :func:`~spey.BackendBase.asimov_negative_loglikelihood`
    * :func:`~spey.BackendBase.minimize_negative_loglikelihood`
    * :func:`~spey.BackendBase.minimize_asimov_negative_loglikelihood`

Beyond the usage of asymptotic hypothesis testing spey also supports sampling from the statistical model which can be embeded
via :func:`~spey.BackendBase.get_sampler` function which takes fit parameters as input and returns a callable function which 
then takes number of samples as input and returns sampled outputs. Additionally, if implemented, spey can use the Hessian 
of :math:`\log\mathcal{L}(\mu, \theta)` to compute variance on :math:`\mu` which can be implemented via 
:func:`~spey.BackendBase.get_hessian_logpdf_func`.

Identifying and installing your statistical model
-------------------------------------------------

In order to add your brand new statistical model to the spey interface all you need to do is to create a ``setup.py`` file
which will create an entry point for the statistical model class. So lets assume that you have the following folder structure

.. code-block:: bash

    my_folder
    ├── my_subfolder
    │   └── mystat_model # this includes the class MyStatisticalModel
    └── setup.py

``setup.py`` file should include the following

.. code-block:: python3

    >>> from setuptools import setup
    >>> stat_model_list = ["mystat_model = my_subfolder.mystat_model:MyStatisticalModel"]
    >>> setup(entry_points={"spey.backend.plugins": stat_model_list})

where

* ``stat_model_list`` is a list of statistical model s you would like to register.
* ``mystat_model`` is the short name for statistical model 
* ``my_subfolder.mystat_model`` is the path to your statistical model class, `MyStatisticalModel`_.

Note that ``stat_model_list`` can include as many implementation as desired. After this step is complete all one needs to do
is ``pip install -e .`` and :func:`~spey.AvailableBackends` function should include ``mystat_model`` as well;

.. code-block:: python3

    >>> import spey
    >>> spey.AvailableBackends() # ['simplified_likelihoods', 'mystat_model']
    >>> spey.get_backend_metadata("mystat_model")
    >>> # {'name': 'my_stat_model',
    ... #  'author': 'John Smith <john.smith@smith.com>',
    ... #  'version': '1.0.0',
    ... #  'spey_requires': '>=0.1.0,<0.2.0',
    ... #  'doi': [],
    ... #  'arXiv': []}