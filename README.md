# madstats
A universal statistics package for LHC reinterpretation

## Installation
`make install` or `pip install -e .`

## Usage

### Using with JSON-HistFactory type input
```python
import madstats, json
import numpy as np
import matplotlib.pyplot as plt

with open("test/signal_test.json", "r") as f:
    signal = json.load(f)

with open("test/background_test.json", "r") as f:
    background = json.load(f)

stat_model = madstats.get_multi_region_statistical_model(
    analysis="atlas_susy_2018_31",
    signal=signal,
    background=background,
    xsection=0.000207244,
)
print(stat_model)
# Out: StatisticalModel(analysis='atlas_susy_2018_31', xsection=2.072e-04 [pb], backend=pyhf)

print(f"1 - CLs : {stat_model.exclusion_confidence_level()}")
# Out: 1 - CLs : 0.4133782692578205

print(f"Expected exclusion cross-section at 95% CLs : {stat_model.s95exp}")
# Out: Expected exclusion cross-section at 95% CLs : 0.0013489112699333228

print(f"Observed exclusion cross-section at 95% CLs : {stat_model.s95obs}")
# Out: Observed exclusion cross-section at 95% CLs : 0.0010071155210690825

print(f"Upper limit on POI : {stat_model.computeUpperLimitOnMu()}")
# Out: Upper limit on POI : 4.859564190370204

mu = np.linspace(-2,2,25)
nll = np.array([stat_model.likelihood(mu=m, return_nll=True, allow_negative_signal=True) for m in mu])

plt.plot(mu, nll)
plt.title("atlas - susy - 2018 - 31".upper(), fontsize=20)
plt.xlabel("$\mu$")
plt.ylabel("negative log-likelihood")
plt.show()
```
![Likelihood plot for pyhf backend](./docs/figs/pyhf_test.png)
