Description of all functions and classes
========================================

Top-Level
---------

.. currentmodule:: spey

.. autosummary:: 
    :toctree: _generated/
    :nosignatures:

    spey.ExpectationType
    spey.AvailableBackends
    spey.get_backend
    spey.get_backend_metadata
    spey.get_uncorrelated_nbin_statistical_model
    spey.get_correlated_nbin_statistical_model
    spey.statistical_model_wrapper

Main Classes
------------

.. autoclass:: spey.StatisticalModel
    :members:
    :undoc-members:
    :show-inheritance:

Base Classes
------------

.. autoclass:: spey.BackendBase
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: spey.base.hypotest_base.HypothesisTestingBase
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: spey.base.model_config.ModelConfig
    :members:
    :undoc-members:
    :show-inheritance:

Statistical Model Combiner
--------------------------

.. autoclass:: spey.StatisticsCombiner
    :members:
    :undoc-members:
    :show-inheritance:

Hypothesis testing
------------------

.. autosummary:: 
    :toctree: _generated/
    :nosignatures:

    spey.hypothesis_testing.test_statistics.qmu
    spey.hypothesis_testing.test_statistics.qmu_tilde
    spey.hypothesis_testing.test_statistics.q0
    spey.hypothesis_testing.test_statistics.get_test_statistic
    spey.hypothesis_testing.test_statistics.compute_teststatistics
    spey.hypothesis_testing.upper_limits.find_poi_upper_limit
    spey.hypothesis_testing.upper_limits.find_root_limits
    spey.hypothesis_testing.utils.compute_confidence_level
    spey.hypothesis_testing.utils.pvalues
    spey.hypothesis_testing.utils.expected_pvalues
    spey.hypothesis_testing.asymptotic_calculator.AsymptoticTestStatisticsDistribution

Default Backends
----------------

Simplified Likelihoods
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary:: 
    :toctree: _generated/
    :nosignatures:

    spey.backends.simplifiedlikelihood_backend.interface.SimplifiedLikelihoodInterface
    spey.backends.simplifiedlikelihood_backend.sldata.SLData

Exceptions
----------

.. automodule:: spey.system.exceptions
    :members:
    :undoc-members:
    :show-inheritance: