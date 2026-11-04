# Model specification and review notes

Let S be RUB per USD. Domestic currency d is RUB; foreign currency f is USD.
Times are year fractions; rates and volatilities are decimal annualized quantities.
All paths returned by the new simulator are generated under the domestic measure Qd.
Inputs are model assumptions. No calibration to tradable instruments is performed.

## Dynamics

Under Qd:

```
dr_d = k_d (theta_d - r_d) dt + sigma_d sqrt(r_d) dW_d
dr_f = [k_f (theta_f - r_f) - rho_fS sigma_f sigma_S sqrt(r_f)] dt
       + sigma_f sqrt(r_f) dW_f
dS/S = (r_d - r_f) dt + sigma_S dW_S
```

The foreign CIR coefficients k_f, theta_f, sigma_f refer to the foreign measure Qf.
The density for Qf relative to Qd is proportional to S B_f / B_d, whose diffusion
under Qd is sigma_S dW_S. Thus dW_f under Qf equals dW_f under Qd minus
rho_fS sigma_S dt; substitution gives the negative quanto correction above.
If all foreign coefficients were instead defined directly under Qd, this correction
would be inappropriate. The API uses one convention only.

No market price of risk is inferred from historical data. The legacy path simply
substitutes physical estimates for the risk-neutral coefficients as a labelled scenario.

## Numerical scheme

For each rate, retain latent x and set r = max(x, 0). Update latent x with drift
and diffusion evaluated at r; report max(x_new, 0). This is full truncation,
not resetting latent x to zero before every step. For FX, use

```
S_new = S * exp((r_d - r_f - sigma_S^2/2) h + sigma_S sqrt(h) Z_S)
```

The number of steps is ceil(T / requested_dt); actual steps are T / n_steps.
All simulation and discount operations use the actual grid. There is no silent
mismatch between a linspace grid and an unrelated integration step.

The continuous-time CIR nonnegativity property does not make Euler exact.
The Feller condition 2 k theta >= sigma^2 is not enforced: zero can be attainable,
and high volatility/low mean cases demand a finer-grid bias study. The foreign
process under Qd additionally has the quanto drift and is not a plain CIR process.

## Payoff, estimator and uncertainty

At final T, pay N c A sum_j(w_j 1{L <= S(t_j) <= U}) / sum_j(w_j).
Here A is explicitly supplied accrual years, c the annual coupon rate; there is no
principal repayment. Observations must be unique increasing grid indices > 0.
Discount each path by exp(-sum_i r_d(t_i) [t_(i+1)-t_i]).
The estimator is the mean of discounted payoffs. Its SE is sample standard deviation
(ddof=1) divided by sqrt(number of independent paths). The normal 95% interval is
approximate; rare events and small samples can give poor coverage.

## Independent benchmarks

With constant rates, log S(t) is normal with mean log S0 + (r_d-r_f-sigma_S^2/2)t
and variance sigma_S^2 t. Hence the expected range coupon is a weighted sum of
normal CDF differences times exp(-r_d T). No independence across observation
dates is required for this expectation.

For the domestic CIR bond, gamma=sqrt(k^2+2 sigma^2),
B=2(exp(gamma T)-1)/[(gamma+k)(exp(gamma T)-1)+2 gamma],
A=[2 gamma exp((k+gamma)T/2)/((gamma+k)(exp(gamma T)-1)+2 gamma)]^(2k theta/sigma^2),
and P(0,T)=A exp(-B r0). This benchmark tests stochastic discounting separately.

The discounted foreign account S(T) exp(integral(r_f-r_d)dt) has mean S0 in the
model. That test alone does NOT identify the foreign quanto correction: the rate
difference cancels under either foreign-rate specification. A separate one-step
foreign-rate drift test checks the measure correction's sign and magnitude.

## Known limitations

See README for historical estimation, missing market calibration and old notebook
outputs. Passing these tests supports the listed numerical cases; it does not establish
correctness for every parameter regime or suitability for live valuation.
