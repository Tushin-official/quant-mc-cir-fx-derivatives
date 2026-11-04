"""Domestic-measure CIR/FX scenarios. S is RUB per USD."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CIRParams:
    kappa: float
    theta: float
    sigma: float

    def __post_init__(self):
        if not np.isfinite([self.kappa, self.theta, self.sigma]).all():
            raise ValueError("CIR parameters must be finite")
        if self.kappa <= 0 or self.theta < 0 or self.sigma < 0:
            raise ValueError("Require kappa > 0, theta >= 0, sigma >= 0")

def correlation_factor(corr):
    """Accept PSD correlations, including singular matrices."""
    corr = np.asarray(corr, dtype=float)
    if (corr.shape != (3, 3) or not np.isfinite(corr).all()
            or not np.allclose(corr, corr.T, rtol=0, atol=1e-12)
            or not np.allclose(np.diag(corr), 1, rtol=0, atol=1e-12)):
        raise ValueError("Expected a finite symmetric 3x3 correlation matrix")
    values, vectors = np.linalg.eigh(corr)
    if values.min() < -1e-12:
        raise ValueError("Correlation matrix must be positive semidefinite")
    return vectors @ np.diag(np.sqrt(np.maximum(values, 0)))

def simulate_paths(*, domestic, foreign, sigma_fx, corr, r_domestic_0,
                   r_foreign_0, spot, horizon=1.0, dt=1/252,
                   n_sims=10000, seed=None, parameter_basis="assumed_q"):
    """Full-truncation Euler rates and log-Euler FX under domestic Q.

    Foreign CIR parameters are defined under foreign Q. Domestic Q adds
    -rho(f,FX)*sigma_f*sigma_fx*sqrt(r_f) to the foreign rate drift.
    assumed_q denotes supplied scenario assumptions, NOT market calibration.
    historical_proxy explicitly labels legacy historical sensitivity scenarios.
    """
    if not isinstance(domestic, CIRParams) or not isinstance(foreign, CIRParams):
        raise TypeError("domestic and foreign must be CIRParams")
    if parameter_basis not in {"assumed_q", "historical_proxy"}:
        raise ValueError("Unsupported parameter basis")
    if (not np.isfinite([sigma_fx, r_domestic_0, r_foreign_0, spot, horizon, dt]).all()
            or min(sigma_fx, r_domestic_0, r_foreign_0) < 0
            or min(spot, horizon, dt) <= 0):
        raise ValueError("Invalid initial values, volatility or time grid")
    if isinstance(n_sims, bool) or not isinstance(n_sims, (int, np.integer)) or n_sims < 2:
        raise ValueError("n_sims must be an integer >= 2")
    factor = correlation_factor(corr)
    corr = np.asarray(corr)
    steps = int(np.ceil(horizon / dt))
    times = np.linspace(0, horizon, steps + 1)
    rng = np.random.default_rng(seed)
    rd = np.empty((n_sims, steps + 1))
    rf = np.empty_like(rd)
    fx = np.empty_like(rd)
    rd[:, 0], rf[:, 0], fx[:, 0] = r_domestic_0, r_foreign_0, spot
    # Retain latent negative states: full truncation is not projected Euler.
    xd = np.full(n_sims, r_domestic_0, dtype=float)
    xf = np.full(n_sims, r_foreign_0, dtype=float)
    for i, h in enumerate(np.diff(times)):
        z = rng.standard_normal((n_sims, 3)) @ factor.T
        pd, pf = np.maximum(xd, 0), np.maximum(xf, 0)
        fx[:, i+1] = fx[:, i] * np.exp((pd-pf-0.5*sigma_fx**2)*h
                                               + sigma_fx*np.sqrt(h)*z[:, 2])
        xd += domestic.kappa*(domestic.theta-pd)*h + domestic.sigma*np.sqrt(pd*h)*z[:, 0]
        quanto = corr[1, 2]*foreign.sigma*sigma_fx*np.sqrt(pf)
        xf += (foreign.kappa*(foreign.theta-pf)-quanto)*h + foreign.sigma*np.sqrt(pf*h)*z[:, 1]
        rd[:, i+1], rf[:, i+1] = np.maximum(xd, 0), np.maximum(xf, 0)
    if not all(np.isfinite(a).all() for a in (rd, rf, fx)):
        raise FloatingPointError("Non-finite simulated paths; check parameters")
    return dict(t_grid=times, S_paths=fx, r_rub_paths=rd, r_usd_paths=rf,
                dt=horizon/steps, measure="domestic_q", fx_quote="RUB_per_USD",
                parameter_basis=parameter_basis, seed=seed)
