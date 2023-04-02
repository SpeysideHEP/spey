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
inheriting it:

.. code-block:: python3
    :linenos:

    >>> from spey import BackendBase

    >>> class MyStatisticalModel(BackendBase):
    >>>     name = "my_stat_model"
    >>>     version = "1.0.0"
    >>>     author = "John Smith <john.smith@smith.com>"
    >>>     spey_requires = ">=0.1.0,<0.2.0"

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
