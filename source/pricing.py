"""RUB-settled single-payment range accrual."""
import warnings
import numpy as np

class RangeAccrualPricer:
    """Pay N * coupon_rate * accrual_years * weighted occupancy at final T.

    No principal repayment. Inclusive barriers; t=0 excluded.
    Defaults preserve N times occupancy, now discounted to time zero.
    """
    def __init__(self, L=None, U=None, notional=None, *, coupon_rate=1.0,
                 accrual_years=1.0):
        self.L = -np.inf if L is None else float(L)
        self.U = np.inf if U is None else float(U)
        if np.isnan([self.L, self.U]).any() or self.L > self.U:
            raise ValueError("Require L <= U")
        if notional is None or not np.isfinite([notional, coupon_rate, accrual_years]).all():
            raise ValueError("Finite notional, coupon and accrual required")
        if min(notional, coupon_rate, accrual_years) < 0:
            raise ValueError("Notional, coupon and accrual must be nonnegative")
        self.notional, self.coupon_rate, self.accrual_years = notional, coupon_rate, accrual_years

    def price_from_paths(self, sim_result, obs_indices=None, obs_weights=None):
        s = np.asarray(sim_result["S_paths"], dtype=float)
        r = np.asarray(sim_result["r_rub_paths"], dtype=float)
        t = np.asarray(sim_result["t_grid"], dtype=float)
        if (s.ndim != 2 or s.shape[0] < 2 or s.shape[1] < 2
                or r.shape != s.shape or t.shape != (s.shape[1],)
                or not all(np.isfinite(a).all() for a in (s, r, t))
                or t[0] != 0 or np.any(np.diff(t) <= 0) or np.any(s <= 0)):
            raise ValueError("Invalid paths or time grid; at least two paths required")
        if sim_result.get("measure") not in (None, "domestic_q"):
            raise ValueError("Pricing requires domestic-measure paths")
        if sim_result.get("fx_quote") not in (None, "RUB_per_USD"):
            raise ValueError("Expected RUB per USD")
        basis = sim_result.get("parameter_basis", "unspecified")
        if basis != "assumed_q":
            warnings.warn("Historical or unspecified parameters: scenario PV, not market-calibrated fair value",
                          UserWarning, stacklevel=2)
        idx = np.arange(1, len(t)) if obs_indices is None else np.asarray(obs_indices)
        if (idx.ndim != 1 or len(idx) == 0 or idx.dtype.kind not in "iu"
                or np.any(idx < 1) or np.any(idx >= len(t)) or np.any(np.diff(idx) <= 0)):
            raise ValueError("Observation indices must be unique, increasing integers in [1, n_steps]")
        weights = np.ones(len(idx)) if obs_weights is None else np.asarray(obs_weights, dtype=float)
        if (weights.shape != idx.shape or not np.isfinite(weights).all()
                or np.any(weights < 0) or weights.sum() <= 0):
            raise ValueError("Weights must be finite, nonnegative and have positive sum")
        w = ((s[:, idx] >= self.L) & (s[:, idx] <= self.U)) @ (weights / weights.sum())
        payoff = self.notional*self.coupon_rate*self.accrual_years*w
        # Left quadrature matches the rate used in each FX evolution step.
        discount = np.exp(-(r[:, :-1] @ np.diff(t)))
        pv = discount*payoff
        if not np.isfinite(pv).all():
            raise FloatingPointError("Non-finite present values")
        price = float(pv.mean())
        stderr = float(pv.std(ddof=1)/np.sqrt(len(pv)))
        return dict(price=price, stderr=stderr,
                    ci95=(price-1.959963984540054*stderr, price+1.959963984540054*stderr),
                    w_paths=w, payoff_paths=payoff, pv_paths=pv,
                    discount_paths=discount, parameter_basis=basis)

