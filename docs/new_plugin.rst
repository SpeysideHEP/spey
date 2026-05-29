.. _sec_new_plugin:

Building a plugin
=================

.. meta::
    :property=og:title: Building a plugin
    :property=og:description: A Spey plug-in allows the usage of custom likelihood prescriptions for inference.
    :property=og:image: https://spey.readthedocs.io/en/main/_static/spey-logo.png
    :property=og:url: https://spey.readthedocs.io/en/main/new_plugin.html

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
metadata to track the implementation's version, author and name. Additionally,
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

* :func:`~spey.BackendBase.get_logpdf_func`: Returns a callable that computes the log-likelihood for any parameter vector.
  Mathematically, this function should return :math:`\log\mathcal{L}(\mu, \theta)` where the input array contains both
  the POI (:math:`\mu`) and nuisance parameters (:math:`\theta`). Behind the scenes, Spey uses this function within
  an optimization loop:

  .. math::

      (\hat{\mu}, \hat{\theta}) = \arg\min_{\mu, \theta} \left[ -\log\mathcal{L}(\mu, \theta) \right]

  The ``expected`` argument determines which data to use in the likelihood computation:
  if ``expected=spey.ExpectationType.observed``, use actual experimental data; if ``expected=spey.ExpectationType.apriori``,
  use background yields as the "observed" data. This ensures the function correctly computes both fit and Asimov likelihoods.
  If ``data`` is provided explicitly, it overrides the default data selection (used for Asimov data in hypothesis testing).

* :func:`~spey.BackendBase.expected_data` (optional): This function is crutial for **asymptotic** hypothesis testing.
  This function is used to generate the expected value of the data with the given fit parameters, i.e. :math:`\theta`
  and :math:`\mu`. If this function does not exist, exclusion limits can still be computed using ``chi_square`` calculator.
  see :func:`~spey.base.hypotest_base.HypothesisTestingBase.exclusion_confidence_level`.

Other available functions that can be implemented are shown in the table below. These are optional optimizations
that improve computational efficiency or enable advanced features.

.. list-table::
    :header-rows: 1

    * - Functions and Properties
      - Mathematical Purpose
      - Use Case
    * - :func:`~spey.BackendBase.get_objective_function`
      - Returns :math:`f(\vec{p}) = -\log\mathcal{L}(\vec{p})` and optionally its gradient :math:`\nabla f`.
        Enables first-order optimization methods that use analytical gradients instead of numerical differentiation.
      - Significant speedup for high-dimensional fits; essential for Automatic Differentiation backends
    * - :func:`~spey.BackendBase.get_hessian_logpdf_func`
      - Returns the Hessian matrix :math:`H_{ij} = \frac{\partial^2 \log\mathcal{L}}{\partial p_i \partial p_j}`.
        The inverse Hessian at the maximum is the Fisher information matrix, used to estimate parameter uncertainties.
      - Accurate uncertainty estimation via :func:`~spey.StatisticalModel.sigma_mu`; required for confidence intervals
    * - :func:`~spey.BackendBase.get_sampler`
      - Returns a function that generates pseudo-datasets by sampling from the likelihood distribution at given parameter values.
        Enables toy Monte Carlo hypothesis testing (see :func:`~spey.StatisticalModel.exclusion_confidence_level` with ``calculator='toy'``).
      - Toy-based exclusion limits; empirical p-value computation when asymptotic approximations are insufficient

.. attention::

    A simple example implementation can be found in the `example-plugin repository <https://github.com/SpeysideHEP/example-plugin>`_
    which implements

    .. math::

        \mathcal{L}(\mu) = \prod_{i\in{\rm bins}}{\rm Poiss}(n^i|\mu n_s^i + n_b^i)

In order to make this model recognised by Spey, the class must be registered as an entry point or by a decorator. The former is
explained in the next section, while the latter can be done by using the :func:`~spey.register_backend` decorator as follows;

.. code-block:: python3
    :linenos:

    >>> import spey

    >>> @spey.register_backend
    >>> class MyStatisticalModel(spey.BackendBase):
    >>>     name = "my_stat_model"
    >>>     ...
    >>>     # rest of the implementation
    >>>     ...

Notice that this method does not require a ``setup.py`` file, but the statistical model will only be
available if the module is imported before calling :func:`~spey.AvailableBackends`. Hence if the goal is to create a package that
can be installed and used as a plugin, the entry point method is preferred.

Identifying and installing your statistical model
-------------------------------------------------

To register your statistical model with Spey, you need to create an entry point. Modern Python projects use ``pyproject.toml``
(recommended), while legacy projects may use ``setup.py``. Both approaches are shown below.

**Folder structure** (same for both methods):

.. code-block:: bash

    my_folder
    ├── my_subfolder
    │   ├── __init__.py
    │   └── mystat_model.py # this includes class MyStatisticalModel
    ├── pyproject.toml    # Modern approach (recommended)
    └── setup.py          # Legacy approach (optional)

Using ``pyproject.toml`` (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The modern, PEP 517/518 compliant approach uses ``pyproject.toml``:

.. code-block:: toml

    [build-system]
    requires = ["setuptools>=64"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "my-spey-plugin"
    version = "1.0.0"
    description = "A custom Spey statistical model"
    requires-python = ">=3.8"
    dependencies = ["spey>=0.1.0"]

    [project.entry-points."spey.backend.plugins"]
    "my_stat_model" = "my_subfolder.mystat_model:MyStatisticalModel"

    [tool.setuptools.packages.find]
    where = ["."]

**Key components:**

* ``[build-system]``: Specifies that the project uses setuptools with PEP 517 backend
* ``[project]``: Standard project metadata (name, version, dependencies)
* ``[project.entry-points."spey.backend.plugins"]``: The section where plugins are registered
  - Key (left of ``=``): The name Spey will use to identify your backend (must match the ``name`` class attribute)
  - Value (right of ``=``): The module path and class name in the format ``"module.path:ClassName"``
* ``[tool.setuptools.packages.find]``: Automatically discovers packages in the current directory

After writing ``pyproject.toml``, install with: ``pip install -e .``

The Spey package will automatically discover your plugin, and :func:`~spey.AvailableBackends` will list ``"my_stat_model"``.

Using ``setup.py`` (Legacy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer the legacy approach or need maximum compatibility with older tools:

.. code-block:: python3

    from setuptools import setup

    stat_model_list = ["my_stat_model = my_subfolder.mystat_model:MyStatisticalModel"]

    setup(
        name="my-spey-plugin",
        version="1.0.0",
        description="A custom Spey statistical model",
        py_modules=["my_subfolder"],
        install_requires=["spey>=0.1.0"],
        entry_points={"spey.backend.plugins": stat_model_list}
    )

**Parameters:**

* ``stat_model_list`` is a list of statistical models to register (can include multiple backends)
* ``"my_stat_model"`` is the backend identifier (must match the class's ``name`` attribute)
* ``"my_subfolder.mystat_model:MyStatisticalModel"`` is the module path and class name

After writing ``setup.py``, install with: ``pip install -e .``

Both methods achieve the same result—after installation, your plugin is immediately available through Spey.
Choose ``pyproject.toml`` for new projects unless you have specific legacy requirements.

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
