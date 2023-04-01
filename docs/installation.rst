.. _sec:installation:

Installation
============

spey is available at pypi, so it can be installed by running:

.. code-block:: bash

    $ pip install spey


Python >=3.8 is required. spey heavily relies on `numpy <https://numpy.org/doc/stable/>`_, `scipy <https://docs.scipy.org/doc/scipy/>`_ and `autograd <https://github.com/HIPS/autograd>`_ which are all packaged during the installation with the necessary versions. Note that some versions may be restricted due to numeric stability and validation.

Plugins
-------

spey is developed to be expanded with other packages that deals with hypothesis testing such as `pyhf <https://pyhf.readthedocs.io/>`_. `pyhf <https://pyhf.readthedocs.io/>`_ integration has been supported via respective plugin which can be installed by following command:

.. code-block:: bash
    
    $ pip install spey-pyhf


