Default Backends
----------------

Distributions
~~~~~~~~~~~~~

.. currentmodule:: spey.backends.distributions

.. autosummary::
    :toctree: ../_generated/

    Poisson
    Normal
    MultivariateNormal
    MainModel
    ConstraintModel

Default PDFs
~~~~~~~~~~~~

.. currentmodule:: spey.backends.default_pdf

.. autosummary::
    :toctree: ../_generated/

    DefaultPDFBase
    UncorrelatedBackground
    CorrelatedBackground
    ThirdMomentExpansion
    EffectiveSigma
    third_moment.third_moment_expansion
    third_moment.compute_third_moments
    uncertainty_synthesizer.signal_uncertainty_synthesizer

Simple PDFs
~~~~~~~~~~~

.. autosummary::
    :toctree: ../_generated/

    simple_pdf.Poisson
    simple_pdf.Gaussian
    simple_pdf.MultivariateNormal
