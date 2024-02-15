---
myst:
  html_meta:
    "property=og:title": "Signal Uncertainties"
    "property=og:description": "Modules to extend the likelihood with signal uncertainties."
    "property=og:image": "https://spey.readthedocs.io/en/main/_static/spey-logo.png"
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

# Signal Uncertainties

Let us assume that we have a set of signal uncertainties (such as scale and PDF uncertainties). We can extend the likelihood prescription to include these uncertainties as a new set of nuisance parameters.

```{note}
:class: dropdown

Note that theoretical uncertainties have different interpretations, we can interpret them similar to experimental uncertainties, as we will do in this tutorial, or they can be interpreted as uncertainties on the cross-section. In the latter case one should compute the limits by changing the signal yields with respect to the change in cross section.
```

```{code-cell} ipython3
:tags: [hide-cell]
import spey
import numpy as np
import matplotlib.pyplot as plt
```

We can add uncorrelated signal uncertainties just like in uncorrelated background case

$$
    \mathcal{L}(\mu, \theta) = \prod_{i\in{\rm bins}}{\rm Poiss}(n_i|\mu (n^{(s)}_i + \theta_i^{(s)}\sigma^{(s)}_i) + n^{(b)}_i + \theta_i^{(b)}\sigma^{(b)}_i) \cdot \prod_{j\in{\rm nui}}\mathcal{N}(\theta_j^{(b)}|0, 1) \cdot \prod_{j\in{\rm nui}}\mathcal{N}(\theta^{(s)}_j|0, 1)\ ,
$$

where $(s)$ superscript indicates signal and $(b)$ indicates background.

```{code-cell} ipython3
pdf_wrapper = spey.get_backend("default_pdf.uncorrelated_background")
statistical_model_sigunc = pdf_wrapper(
    signal_yields=[12.0, 15.0],
    background_yields=[50.0, 48.0],
    data=[36, 33],
    absolute_uncertainties=[12.0, 16.0],
    signal_uncertainty_configuration={"absolute_uncertainties": [3.0, 4.0]},
)
```

Similarly, we can construct signal uncertainties using ``"absolute_uncertainty_envelops"`` keyword which accepts upper and lower uncertainties as ``[(upper, lower)]``. We can also add a correlation matrix with ``"correlation_matrix"`` keyword and third moments with ``"third_moments"`` keyword. Notice that these are completely independent of background. Now we can simply compute the limits as follows

```{code-cell} ipython3
print(f"1 - CLs: {statistical_model_sigunc.exclusion_confidence_level()[0]:.5f}")
print(f"POI upper limit: {statistical_model_sigunc.poi_upper_limit():.5f}")
```

```python
1 - CLs: 0.96607
POI upper limit: 0.88808
```

Let us also check the $\chi^2$ distribution with respect to POI which we expect the distribution should get wider with signal uncertainties. For this comparison we first need to define the model without signal uncertainties:

```{code-cell} ipython3
statistical_model = pdf_wrapper(
    signal_yields=[12.0, 15.0],
    background_yields=[50.0, 48.0],
    data=[36, 33],
    absolute_uncertainties=[12.0, 16.0],
)
```

Using ``statistical_model`` and ``statistical_model_sigunc`` we can compute the $\chi^2$ distribution

```{code-cell} ipython3
:tags: [hide-cell]
poi = np.linspace(-3,2,20)
plt.plot(poi, [statistical_model.chi2(p) for p in poi], color="b", label="no signal uncertainties")
plt.plot(poi, [statistical_model_sigunc.chi2(p) for p in poi], color="r", label="with signal uncertainties")
plt.legend()
plt.xlabel("$\mu$")
plt.ylabel("$\chi^2(\mu)$")
plt.show()
```

```{figure} ../figs/sig_unc_chi2.png
---
width: 60%
figclass: caption
alt: chi-square distribution
name: fig1
---
$\chi^2(\mu)$ distribution comparisson for statistical model with and without signal uncertainties.
```
