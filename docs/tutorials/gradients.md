---
myst:
  html_meta:
    "property=og:title": "Gradient of a Statistical Model"
    "property=og:description": "Modules to compute gradient and Hessian of negative log-probabilities"
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

# Gradient of a Statistical Model

````{margin}
```{versionadded} 0.1.6
```

```{note}
In previous versions gradient and Hessian was limited to internal computations only.
```
````

With version 0.1.6, Spey includes additional functionalities to extract gradient and Hessian information directly from the statistical model. The gradient and Hessian are defined as follows

$$
{\rm Gradient} = -\frac{d\log\mathcal{L}(\theta)}{d\theta}\quad , \quad {\rm Hessian} = -\frac{d^2\log\mathcal{L}(\theta)}{d\theta_i d\theta_j}\quad , \quad \mu ,\theta_i \in \theta \ .
$$

In order to access this information we will use `spey.math` module.

```{code-cell} ipython3
:tags: [hide-cell]
import spey
from spey.math import value_and_grad, hessian
import numpy as np
np.random.seed(14)
```

{py:func}`spey.math.value_and_grad` returns a function that computes negative log-likelihood and its gradient for a given statistical model and {py:func}`spey.math.hessian` returns a function that computes Hessian of negative log-likelihood.

Let us examine this on ``"default_pdf.uncorrelated_background"``:

```{code-cell} ipython3
pdf_wrapper = spey.get_backend("default_pdf.uncorrelated_background")

data = [36, 33]
signal_yields = [12.0, 15.0]
background_yields = [50.0, 48.0]
background_unc = [12.0, 16.0]

stat_model = pdf_wrapper(
    signal_yields=signal_yields,
    background_yields=background_yields,
    data=data,
    absolute_uncertainties=background_unc,
)
```

Here we constructed a two-bin statistical model with observations $36,\ 33$, signal yields $12,\ 15$ and background yields $50\pm12,\ 48\pm16$. We can construct the function that will return negative log probability and its gradient as follows

```{code-cell} ipython3
neg_logprob = value_and_grad(stat_model)
```

Notice that this function constructs a negative log-probability for the observed statistical model using the default data that we provided earlier. This can be changed using ``expected`` and ``data`` keywords. Now we can choose nuisance parameters and execute the function:

```{code-cell} ipython3
nui = np.random.uniform(0,1,(3,))
neg_logprob(nui)
```

```python
(27.81902589793928, array([13.29067478,  6.17223275,  9.28814191]))
```

For this particular model, we have only two nuisance parameters, $\theta_i$, and signal strength, $\mu$, due to the structure of the statistical model. For Hessian, we can use the same formulation:

```{code-cell} ipython3
hess = hessian(stat_model)
hess(nui)
```

```python
array([[ 2.74153126,  1.21034187,  1.63326868],
       [ 1.21034187,  2.21034187, -0.        ],
       [ 1.63326868, -0.        ,  2.74215326]])
```
