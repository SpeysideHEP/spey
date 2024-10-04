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

Any Spey statistical model can compute the exclusion confidence level using three different calculators. The availability of these options depends on the functions used in the likelihood construction (see {ref}`this section <sec_new_plugin>` for details). To check which calculators are available for a given model, users can call the {func}`~spey.StatisticalModel.available_calculators` function.

The {func}`~spey.StatisticalModel.exclusion_confidence_level` function allows users to select a calculator through the ``calculator`` keyword, with three options:

- **"asymptotic"**: Uses asymptotic formulae to calculate p-values (see ref. {cite}`Cowan:2010js` for details). This method requires access to the expected values of the distribution to construct Asimov data, forming the test statistic based on the Asimov likelihood. This is the most common method used in LHC analyses.
- **"toy"**: Relies on the sampling functionality of the likelihood, requiring the construction to support sampling. It computes p-values by drawing samples from both the signal-plus-background and background-only distributions.
- **"chi_square"**: Compares $\chi^2(\mu)$ distributions,

  $$
  \chi^2(\mu) = -2 \log\frac{\mathcal{L}(\mu, \theta_\mu)}{\mathcal{L}(\hat{\mu},\hat{\theta}_{\mu})},
  $$

  where $\chi^2(\mu=1)$ represents signal-like behavior and $\chi^2(\mu=0)$ represents background-like behavior, to compute the p-values. This method was widely used during the Tevatron era.

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
