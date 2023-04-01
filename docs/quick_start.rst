.. _sec:installation:

Installation
============

``spey`` is available at `pypi <https://pypi.org>`_ , so it can be installed by running:

.. code-block:: bash

    $ pip install spey


Python >=3.8 is required. spey heavily relies on `numpy <https://numpy.org/doc/stable/>`_, 
`scipy <https://docs.scipy.org/doc/scipy/>`_ and `autograd <https://github.com/HIPS/autograd>`_ 
which are all packaged during the installation with the necessary versions. Note that some 
versions may be restricted due to numeric stability and validation.

.. _sec:plugins:

Plugins
-------

``spey`` works with various packages that are designed to deliver certain statistical model
prescriptions. The goal of the ``spey`` interface is to collect all these prescriptions under
the same roof and provide a toolset to combine different sources of likelihoods. List of plugins
are as follows;

  * `spey-pyhf <https://github.com/SpeysideHEP/spey-pyhf>`_ : enables the usage of :xref:`pyhf` 
    package through ``spey`` interface.
  * ``spey-fastprof`` : enables the usage of ``fastprof`` through ``spey`` interface.

By default, ``spey`` is shipped with ``"simplified_likelihoods"`` backend.

.. _sec:quick_start:

Quick Start
===========