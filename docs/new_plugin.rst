.. _sec_new_plugin:

Building a plugin
=================

``spey`` package has been designed to be expandable. It only needs to know certain aspects of the 
data structure that is presented and a prescription to form a likelihood function.

What a plugin provides
----------------------

A quick intro on the terminology of spey plugins in this section:

  * A plugin is an external Python package that provides additional statistical model 
    prescriptions to spey.
  * Each plugin may provide one (or more) statistical model prescriptions  
    accessible directly through Spey.
  * Depending on the scope of the plugin, you may wish to provide additional (custom) 
    operations and differentiability through various autodif packages such as ``autograd``
    or ``jax``. As long as they are implemented through predefined function names,
    Spey can automatically detect and use them within the interface. 

Creating your Statistical Model Prescription
--------------------------------------------

The first step in creating your Spey plugin is to create your statistical model interface. 
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

    >>>     def __init__(self, ...)
    >>>         ...

    >>>     @property
    >>>     def is_alive(self):
    >>>         ...

    >>>     def config(
    ...         self, allow_negative_signal: bool = True, poi_upper_bound: float = 10.0
    ...     ):
    >>>         ...

    >>>     def get_logpdf_func(
    ...         self, expected = spey.ExpectationType.observed, data = None
    ...     ):
    >>>         ...
    
    >>>     def expected_data(self, pars):
    >>>         ...


:class:`~spey.BackendBase` requires certain functionality from the statistical model to be 
implemented, but let us first go through the above class structure. Spey looks for specific 
metadata to track the version, author and name of the implementation. Additionally, 
it checks compatibility with the current Spey version to ensure that the plugin works as it should.

.. note:: 

    The list of metadata that Spey is looking for:

      * **name** (``str``): Name of the plugin.
      * **version** (``str``): Version of the plugin.
      * **author** (``str``): Author of the plugin.
      * **spey_requires** (``str``): The minimum spey version that the 
        plugin needs, e.g. ``spey_requires="0.0.1"`` or ``spey_requires=">=0.3.3"``.
      * **doi** (``List[str]``): Citable DOI numbers for the plugin.
      * **arXiv** (``List[str]``): arXiv numbers for the plugin.

`MyStatisticalModel`_ class has four main functionalities namely :func:`~spey.BackendBase.is_alive`, 
:func:`~spey.BackendBase.config`, :func:`~spey.BackendBase.get_logpdf_func`,  and 
:func:`~spey.BackendBase.expected_data`(for detailed descriptions of these functions please go to the 
:class:`~spey.BackendBase` documentation by clicking on them.)

* :func:`~spey.BackendBase.is_alive`: This function returns a boolean indicating that the statistical model 
  has at least one signal bin with a non-zero yield.

* :func:`~spey.BackendBase.config`: This function returns :class:`~spey.base.model_config.ModelConfig` class
  which includes certain information about the model structure, such as the index of the parameter of interest 
  within the parameter list (:attr:`~spey.base.model_config.ModelConfig.poi_index`), minimum value parameter 
  of interest can take (:attr:`~spey.base.model_config.ModelConfig.minimum_poi`), suggested initialisation
  parameters for the optimiser (:attr:`~spey.base.model_config.ModelConfig.suggested_init`) and suggested 
  bounds for the parameters (:attr:`~spey.base.model_config.ModelConfig.suggested_bounds`). If 
  ``allow_negative_signal=True`` the lower bound of POI is expected to be zero; if ``False`` 
  :attr:`~spey.base.model_config.ModelConfig.minimum_poi`. ``poi_upper_bound`` is used to enforce an upper 
  bound on POI.

  .. note:: 

    Suggested bounds and initialisation values should return a list with a length of the number of nuisance 
    parameters and parameters of interest. Initialisation values should be a type of ``List[float, ...]`` 
    and bounds should have the type of ``List[Tuple[float, float], ...]``.

* :func:`~spey.BackendBase.get_logpdf_func`: This function returns a function that takes a NumPy array 
  as an input which indicates the fit parameters (nuisance, :math:`\theta`, and POI, :math:`\mu`) and returns the
  value of the natural logarithm of the likelihood function, :math:`\log\mathcal{L}(\mu, \theta)`. The input 
  ``expected`` defines which data to be used in the absence of ``data`` input, i.e. if 
  ``expected=spey.ExpectationType.observed`` yields of observed data should be used to compute the likelihood, but 
  if ``expected=spey.ExpectationType.apriori`` background yields should be used. This ensures the difference between 
  prefit and postfit likelihoods. If ``data`` is provided, it is overwritten; this is for the case where Asimov 
  data is in use.

* :func:`~spey.BackendBase.expected_data` (optional): This function is crutial for **asymptotic** hypothesis testing.
  This function is used to generate the expected value of the data with the given fit parameters, i.e. :math:`\theta`
  and :math:`\mu`. If this function does not exist, exclusion limits can still be computed using ``chi_square`` calculator.
  see :func:`~spey.base.hypotest_base.HypothesisTestingBase.exclusion_confidence_level`.

Other available functions that can be implemented are shown in the table below.

.. list-table:: 
    :header-rows: 1
    
    * - Functions and Properties
      - Explanation
    * - :func:`~spey.BackendBase.get_objective_function`
      - Returns the objective function and/or its gradient.
    * - :func:`~spey.BackendBase.get_hessian_logpdf_func` 
      - Returns Hessian of the log-probability
    * - :func:`~spey.BackendBase.get_sampler` 
      - Returns a function to sample from the likelihood distribution.

.. attention:: 
    
    A simple example implementation can be found in the `example-plugin repository <https://github.com/SpeysideHEP/example-plugin>`_
    which implements

    .. math:: 

        \mathcal{L}(\mu) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + n_b^i)



Identifying and installing your statistical model
-------------------------------------------------

In order to add your brand new statistical model to the spey interface, all you need to do is to create a ``setup.py`` file
which will create an entry point for the statistical model class. So let's assume that you have the following folder structure

.. code-block:: bash

    my_folder
    ├── my_subfolder
    │   ├── __init__.py
    │   └── mystat_model.py # this includes class MyStatisticalModel
    └── setup.py

The ``setup.py`` file should include the following

.. code-block:: python3

    >>> from setuptools import setup
    >>> stat_model_list = ["my_stat_model = my_subfolder.mystat_model:MyStatisticalModel"]
    >>> setup(entry_points={"spey.backend.plugins": stat_model_list})

where

* ``stat_model_list`` is a list of statistical models you would like to register.
* ``my_stat_model`` is the short name for a statistical model. This should be the same as the ``name`` attribute
  of the class. Spey will identify the backend with this name.
* ``my_subfolder.mystat_model`` is the path to your statistical model class, `MyStatisticalModel`_.

Note that ``stat_model_list`` can include as many implementations as desired. After this step is complete, all one needs to do
is ``pip install -e .`` and :func:`~spey.AvailableBackends` function will include ``mystat_model`` as well.

Citing Plug-ins
---------------

Since other users can build plug-ins, they are given a metadata accessor to extract proper information
to cite them. :func:`~spey.get_backend_metadata` function allows the user to extract name, author, version, DOI and
arXiv number to be used in academic publications. This information can be accessed as follows

.. code-block:: python3

    >>> import spey
    >>> spey.get_backend_metadata("mystat_model")
    >>> # {'name': 'my_stat_model',
    ... #  'author': 'John Smith <john.smith@smith.com>',
    ... #  'version': '1.0.0',
    ... #  'spey_requires': '>=0.1.0,<0.2.0',
    ... #  'doi': [],
    ... #  'arXiv': []}