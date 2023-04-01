.. _sec:introduction:

``spey``: Smooth statistics combination for LHC reinterpretation studies
========================================================================

In recent years LHC analysis results have started to ship with more and more 
information regarding the statistical model used within the collaboration. 
Some collaborations have provided correlation information between the histogram 
bins, whereas others have provided full statistical model information in specific 
formats. Hence, a wide range of packages have been developed to make use of the 
output provided by the experimental collaborations. Although this has significantly
improved the accuracy of the simple recasts, lately, there has been a push towards
going beyond the scope of a single analysis and figuring out a way to combine 
different analyses and experiments that might constrain the same phase space. 
This requires collecting different prescriptions of statistical models under one 
roof, performing hypothesis testing for a combination of different statistical model 
prescriptions, and perhaps going beyond to cover the possibility of different types 
of prescriptions in the future.

Bibliography
~~~~~~~~~~~~

.. bibliography:: bib/references.bib
   :filter: docname in docnames
   :style: plain
   :keyprefix: intro-
   :labelprefix: intro-