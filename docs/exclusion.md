---
myst:
  html_meta:
    "property=og:title": "Exclusion limits"
    "property=og:description": "How does the exclusion limits work in spey"
    "property=og:image": "https://spey.readthedocs.io/en/main/_static/spey-logo.png"
    "property=og:url": "https://spey.readthedocs.io/en/main/exclusion.html"
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.8.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Exclusion limits

Any Spey statistical model can compute the exclusion confidence level
using three options. Depending on the available functions in likelihood construction
(see {ref}`this section <sec_new_plugin>` for details), one or more of these options will be
available for the user. One can use
{func}`~spey.StatisticalModel.available_calculators` function to see which calculators are available.

{func}`~spey.StatisticalModel.exclusion_confidence_level` function uses ``calculator`` keyword
to choose in between ``"asymptotic"``, ``"toy"`` and ``"chi_square"`` calculators.

* ``"asymptotic"``: uses asymptotic formulae to compute p-values, see ref. {cite}`Cowan:2010js`
  for details. This method is only available if the likelihood construction has access to
  the expected values of the distribution, which allows one to construct Asimov data. Hence the test statistic is constructed with the Asimov likelihood.
* ``"toy"``: This method uses the sampling functionality of the likelihood, hence expects the
  construction to have sampling abilities. It computes p-values by sampling from signal+background
  and background-only distributions.
* ``"chi_square"``: This method compares $\chi^2(\mu)$ distributions,

  $$

        \chi^2(\mu) = -2 \log\frac{\mathcal{L}(\mu, \theta_\mu)}{\mathcal{L}(\hat{\mu},\hat{\theta}_{\mu})},
  $$

  for signal-like $\chi^2(\mu=1)$ and background like $\chi^2(\mu=0)$ to compute the p-values for the model.

The `expected` keyword allows users to select between computing observed or expected exclusion limits. It also supports the calculation of prefit expected exclusion limits, which can be enabled by setting `expected=spey.ExpectationType.apriori`. This option ignores experimental data and computes the expected exclusion limit based solely on the simulated Standard Model (SM) background yields. On the other hand, {attr}`~spey.ExpectationType.observed` (for observed limits) and {attr}`~spey.ExpectationType.aposteriori` (for post-fit expected limits) compute the exclusion confidence limits after fitting the model.

In both expected cases, the exclusion limits are returned with $\pm1\sigma$ and $\pm2\sigma$ variations around the background model, resulting in five values: $[-2\sigma, -1\sigma, 0, 1\sigma, 2\sigma]$. However, the observed exclusion limit returns a single value. The ``"chi_square"`` calculator is an exception—it only provides one value for both observed and expected limits.

The `allow_negative_signal` keyword controls which test statistic is used and restricts the values that $\mu$ (the signal strength) can take when computing the maximum likelihood. When `allow_negative_signal=True`, the $q_\mu$ test statistic is applied; otherwise, the $\tilde{q}_\mu$ statistic is used (for further details, see {cite}`Cowan:2010js, Araz:2023bwx`).

For complex statistical models, optimizing the likelihood can be challenging and depends on the choice of optimizer. Spey uses SciPy for optimization and fitting tasks. Any additional keyword arguments not explicitly covered in the `exclusion_confidence_level` function description are passed directly to the optimizer, allowing users to customize its behavior through the interface.

Below we compare the exclusion limits computed with each approach. This comparisson uses normal distribution for the likelihood (`default_pdf.normal`) background yields are set to $n_b$, uncertainties are shown with $\sigma$ and observations are given with $n$.

```{figure} ./figs/comparisson_observed.png
---
width: 100%
figclass: caption
alt: exclusion limit calculator comparisson
name: fig2
---
exclusion limit calculator comparisson for observed p-values
```
