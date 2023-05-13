Description of all functions and classes
========================================

Top-Level
---------

.. currentmodule:: spey

.. autosummary:: 
    :toctree: _generated/

    ExpectationType
    AvailableBackends
    get_backend
    get_backend_metadata
    get_uncorrelated_nbin_statistical_model
    get_correlated_nbin_statistical_model
    statistical_model_wrapper
    helper_functions.correlation_to_covariance
    helper_functions.covariance_to_correlation

Main Classes
------------

.. autoclass:: spey.StatisticalModel
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Base Classes
------------

.. autoclass:: spey.BackendBase
    :members:
    :undoc-members:

.. autoclass:: spey.base.hypotest_base.HypothesisTestingBase
    :members:
    :undoc-members:

.. autoclass:: spey.base.model_config.ModelConfig
    :members:
    :undoc-members:

Statistical Model Combiner
--------------------------

.. autoclass:: spey.UnCorrStatisticsCombiner
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Hypothesis testing
------------------

.. currentmodule:: spey.hypothesis_testing

.. autosummary:: 
    :toctree: _generated/

    test_statistics.qmu
    test_statistics.qmu_tilde
    test_statistics.q0
    test_statistics.get_test_statistic
    test_statistics.compute_teststatistics
    upper_limits.find_poi_upper_limit
    upper_limits.find_root_limits
    utils.pvalues
    utils.expected_pvalues
    distributions.AsymptoticTestStatisticsDistribution
    distributions.EmpricTestStatisticsDistribution
    asymptotic_calculator.compute_asymptotic_confidence_level
    toy_calculator.compute_toy_confidence_level

Default Backends
----------------

Simplified Likelihoods
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: spey.backends.simplifiedlikelihood_backend

.. autosummary:: 
    :toctree: _generated/

    UncorrelatedBackground
    SimplifiedLikelihoods
    ThirdMomentExpansion
    VariableGaussian
    sldata.SLData
    distributions.Poisson
    distributions.Normal
    distributions.MultivariateNormal
    distributions.MainModel
    distributions.ConstraintModel

Exceptions
----------

.. currentmodule:: spey.system.exceptions

.. autosummary:: 
    :toctree: _generated/
    :nosignatures:

    AnalysisQueryError
    MethodNotAvailable
    NegativeExpectedYields
    PluginError
    UnknownCrossSection
    UnknownTestStatistics
    CalculatorNotAvailable
