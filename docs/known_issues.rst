Known Issues
============

* It has been observed that the Scipy version and its compiler, which is used for maximising and profiling the likelihood can have a slight effect on the results.

* NumPy v2.0.0 does not work with autograd, which is used in "default_pdf.XXX" likelihoods. The dependencies are reformulated accordingly.