Description of all functions and classes
========================================

.. meta::
    :property=og:title: Description of all functions and classes
    :property=og:image: https://spey.readthedocs.io/en/main/_static/spey-logo.png
    :property=og:url: https://spey.readthedocs.io/en/main/api.html

Top-Level
---------

.. currentmodule:: spey

.. autosummary::
    :toctree: _generated/

    version
    about
    check_updates
    ExpectationType
    register_backend
    AvailableBackends
    set_optimiser
    get_backend
    get_backend_metadata
    get_backend_bibtex
    cite
    reset_backend_entries
    statistical_model_wrapper
    helper_functions.correlation_to_covariance
    helper_functions.covariance_to_correlation
    helper_functions.merge_correlated_bins
    optimizer.core.fit
    set_log_level

Main Classes
------------

.. autoclass:: spey.StatisticalModel
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Base Classes
------------

Backend Base
~~~~~~~~~~~~

.. autoclass:: spey.BackendBase
    :members:
    :undoc-members:

Hypothesis Testing Base
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: spey.base.hypotest_base.HypothesisTestingBase
    :members:
    :undoc-members:

Model Configuration
~~~~~~~~~~~~~~~~~~~

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
    upper_limits.ComputerWrapper
    upper_limits.find_poi_upper_limit
    upper_limits.find_root_limits
    utils.pvalues
    utils.expected_pvalues
    distributions.AsymptoticTestStatisticsDistribution
    distributions.EmpricTestStatisticsDistribution
    asymptotic_calculator.compute_asymptotic_confidence_level
    toy_calculator.compute_toy_confidence_level

Gradient Tools
--------------

.. currentmodule:: spey.math

.. autosummary::
    :toctree: _generated/

    value_and_grad
    hessian

Default Backends
----------------

Distributions
~~~~~~~~~~~~~

.. currentmodule:: spey.backends.distributions

.. autosummary::
    :toctree: _generated/

    Poisson
    Normal
    MultivariateNormal
    MainModel
    ConstraintModel

Default PDFs
~~~~~~~~~~~~

.. currentmodule:: spey.backends.default_pdf

.. autosummary::
    :toctree: _generated/

    DefaultPDFBase
    UncorrelatedBackground
    CorrelatedBackground
    ThirdMomentExpansion
    EffectiveSigma
    third_moment.third_moment_expansion
    third_moment.compute_third_moments

Simple PDFs
~~~~~~~~~~~

.. autosummary::
    :toctree: _generated/

    simple_pdf.Poisson
    simple_pdf.Gaussian
    simple_pdf.MultivariateNormal

Developer Tools
---------------

Exceptions
~~~~~~~~~~

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
    CanNotFindRoots
    UnknownComputer
    CombinerNotAvailable

Logging
~~~~~~~

.. currentmodule:: spey.system.logger

.. autosummary::
    :toctree: _generated/
    :nosignatures:

    disable_logging
    capture_logs
