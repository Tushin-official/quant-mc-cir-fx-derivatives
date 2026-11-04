# CIR / FX Range Accrual — Monte Carlo research prototype

Three correlated factors: RUB short rate, USD short rate and **RUB per USD** spot.
This educational implementation estimates a RUB-settled range-accrual coupon under
explicit model assumptions. It is **not a market-calibrated production pricer**.

## Quick start

Python 3.11+:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m examples.validate
```

The tests and example use synthetic inputs and require no market-data downloads.
`docs/validation.json` contains an actual seeded run, including runtime versions.
Dependencies are unpinned; numerical output can vary across library/platform versions.

```python
import numpy as np
from source.simulation import CIRParams, simulate_paths
from source.pricing import RangeAccrualPricer

paths = simulate_paths(
    domestic=CIRParams(1.2, 0.06, 0.10),
    foreign=CIRParams(1.0, 0.03, 0.08),
    sigma_fx=0.20, corr=np.eye(3),
    r_domestic_0=0.06, r_foreign_0=0.03, spot=100,
    horizon=1, dt=1/252, n_sims=10000, seed=2026,
)
result = RangeAccrualPricer(90, 110, notional=1_000_000,
                          coupon_rate=0.10, accrual_years=1).price_from_paths(paths)
print(result['price'], result['stderr'], result['ci95'])
```

These are illustrative inputs, not calibrated quotes. `assumed_q` means that the
caller supplies model parameters under the stated measures; it does not certify them.

## Product and model

- One coupon paid in RUB at the final grid date. No principal repayment.
- Coupon = notional × annual coupon rate × accrual years × weighted fraction of
  observation dates inside inclusive barriers. Default coupon and accrual factors
  are both one, preserving the original normalized payoff.
- Default observations are all grid points **after** time zero. Use `obs_indices`
  and `obs_weights` for an explicit schedule; weights are normalized.
- Stochastic RUB discounting uses left-point integration on the actual time grid.
- FX uses an exponential update. Rates use full-truncation Euler with latent states;
  this has discretization bias and is not exact CIR transition sampling.
- Correlation order is `(RUB rate, USD rate, FX)`. Valid PSD matrices, including
  singular ones, are supported; invalid matrices are rejected rather than repaired.
- USD rate parameters are specified under the USD measure. Simulation under the
  RUB measure includes the foreign-rate quanto drift correction.

See [model specification](docs/model.md) for equations, units and limitations.

## Validation

The test suite covers analytic constant-rate range accrual, CIR bond price and mean,
discounted foreign-account martingale, foreign-rate measure correction, deterministic
limits, observation schedules, input checks, seeds, stressed positivity and the legacy API.
Monte Carlo standard errors use independent **path-level** discounted payoffs.
The approximate 95% interval measures sampling error only, not model or discretization error.

The example reports two separate experiments: sample-size sensitivity with fixed
monthly observation dates, and CIR bond time-step sensitivity against an analytic price.
It does not assert monotonic convergence from one random realization.

## Legacy notebooks and historical estimation

`source/core.py` retains `Preprocess` and the original data-loading interface.
Historical OLS/bootstrap estimates are **physical-measure estimates**, not market
risk-neutral calibration. `Preprocess.simulate()` now emits a warning and labels
its output `historical_proxy`; its discounted values are sensitivity scenarios.
Unconstrained OLS can produce invalid CIR estimates; these now raise a clear error.
A bootstrap draw that cannot be fitted also fails explicitly rather than being silently
excluded. A constrained estimator and bootstrap failure-rate analysis remain future work.

`dt` in historical estimation assumes equally spaced observations. The old workflow
joins trading-day observations and does not estimate a calendar-time model with irregular
intervals. Correlations of raw increments are proxies, not fitted diffusion correlations.
These limitations preclude interpreting this route as an independently validated calibration.

Existing `test.ipynb` and `test_en.ipynb` are historical exploratory notebooks. Their
stored outputs were **not rerun** and reflect the previous implementation. Use the
module example and numerical report for current behavior. The corrected pricer intentionally
changes old prices through discounting and exclusion of time zero.

## Scope remaining

Market curve fitting and volatility calibration, date/calendar conventions, multiple coupon
payments, funding/collateral and cross-currency basis, Greeks with error analysis, broad
stress tests, and independent review are outside the validated scope. No trading-performance
or production-readiness claim is made.
