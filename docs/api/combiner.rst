Statistical Model Combiners
---------------------------

Statistical models are combined through two plug-ins, retrieved like any other backend
via :func:`~spey.get_backend`; the result is an ordinary :class:`~spey.StatisticalModel`.

* ``default.uncorrelated_combiner`` (:class:`~spey.combiner.UncorrelatedStatisticsCombiner`):
  independent analyses that share only their parameter of interest.
* ``default.correlated_combiner`` (:class:`~spey.combiner.CorrelatedStatisticsCombiner`):
  analyses that share any declared parameters.

Both are built on :class:`~spey.combiner.combiner_core.CombinerBase`. See
:doc:`../comb` for a worked example.

Uncorrelated Statistical Model Combiner
---------------------------------------

.. automodule:: spey.combiner.uncorrelated_statistics_combiner

.. autoclass:: spey.combiner.UncorrelatedStatisticsCombiner
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Correlated Statistical Model Combiner
-------------------------------------

.. automodule:: spey.combiner.correlated_statistics_combiner

.. autoclass:: spey.combiner.CorrelatedStatisticsCombiner
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Combiner Core
-------------

.. automodule:: spey.combiner.combiner_core

.. autoclass:: spey.combiner.combiner_core.CombinerBase
    :members:
    :private-members: _build_parameter_map, _select_poi, _allocate_slots, _local_parameter_names
    :show-inheritance:

Deprecated: UnCorrStatisticsCombiner
------------------------------------

.. automodule:: spey.combiner.deprecated_combiner

.. autoclass:: spey.UnCorrStatisticsCombiner
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
