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

The first step in creating your own spey plugin is to create your statistical model interface and 
data structure. The data structure needs to inherit :obj:`~spey.DataBase` and the inderface
needs to inherit :obj:`~spey.BackendBase` class to ensure certain specifications
within the plugin. Beyond providing required abstract functions, user is free to shape the classes
to their liking. An example of a basic data structure is shown below;

.. code-block:: python3

    >>> import spey
    
    >>> class MyData(spey.DataBase):
    >>>     def __init__(self, *args, **kwargs):
    >>>         """Construct statistical model with signal, background, and data yields."""
    >>>         ...
    
    >>>     @property
    >>>     def isAlive(self) -> bool:
    >>>         """Return ``True`` if at least on bin has non zero signal yields"""
    >>>         ...
    
    >>>     def config(
    ...         self, allow_negative_signal: bool = True, poi_upper_bound: float = 40.0
    ...     ) -> spey.base.model_config.ModelConfig:
    >>>         """Return model configuration"""
    >>>         ...

* **__init__** : Initialisation typically takes information about signal, background, observed
  yields, information regarding the correlation structure etc. However, it can be structured 
  according to the needs of the interface.
* :attr:`~spey.DataBase.isAlive`: Simply indicates that there is at least one bin in the 
  histogram with non zero signal yields. This allows algorithm to terminate early to avoid 
  unnecessary CPU consumption.
* :func:`~spey.DataBase.config`: Model configuration returns a 
  :class:`~spey.base.model_config.ModelConfig` class which includes information about index of 
  the paramtere of interest within the list of fit parameters, allows **minimum_poi** information 
  to pass along the optimiser, proposes suggested initialisation and boundaries to the fit 
  parameters for the optimiser. The latter information is always possible to alter during the 
  execution of the code by providing ``init_pars`` and ``par_bounds`` arguments.

The rest of the functions that are used through :obj:`~spey.DataBase` can be seen through the 
documentation of the class itself and default return values can be altered by overwriting the 
function itself through inheritance.

.. code-block:: python3

    >>> import spey
    >>> from .mydata import MyData

    >>> class MyStatisticalModel(spey.BackendBase):
    >>>     name = "truncated_gaussian"
    >>>     version = "1.0.0"
    >>>     author = "John Smith"
    >>>     spey_requires = ">=0.0.1"
    >>>     datastructure = MyData

.. note:: 

    The list of metadata that spey is looking for:

      * **name** (``str``, *required*): Name of the plugin.
      * **version** (``str``, *required*): Version of the plugin.
      * **author** (``str``, *required*): Author of the plugin.
      * **spey_requires** (``str``, *required*): The minimum spey version that the 
        plugin is built e.g. ``spey_requires="0.0.1"`` or ``spey_requires=">=0.3.3"``.
      * **doi** (``List[str]``): Citable DOI numbers for the plugin.
      * **arXiv** (``List[str]``): arXiv numbers for the plugin.
      * **datastructure** (``Callable`` or :obj:`~spey.DataBase` , *required*): Container 
        that includes and describes the relation of the input data with the statistical model.