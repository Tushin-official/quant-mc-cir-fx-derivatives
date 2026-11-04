"""Deterministic-seed numerical report; run from repository root as a module."""
import json
import platform
import numpy as np
from scipy.special import ndtr
from source.simulation import CIRParams, simulate_paths
from source.pricing import RangeAccrualPricer


def run():
    report = dict(python=platform.python_version(), numpy=np.__version__, seed=2026,
                  convention='RUB per USD; monthly observations; payment at one year',
                  sample_size_study=[], time_step_study=[])
    for n in (2000, 8000, 32000):
        s = simulate_paths(domestic=CIRParams(1,.06,0), foreign=CIRParams(1,.03,0),
                           sigma_fx=.2, corr=np.eye(3), r_domestic_0=.06,
                           r_foreign_0=.03, spot=100, dt=1/12, n_sims=n, seed=2026)
        t=s['t_grid'][1:]
        prob=ndtr((np.log(1.1)-.01*t)/(.2*np.sqrt(t)))-ndtr((np.log(.9)-.01*t)/(.2*np.sqrt(t)))
        exact=float(100*np.exp(-.06)*prob.mean())
        p=RangeAccrualPricer(90,110,100).price_from_paths(s)
        report['sample_size_study'].append(dict(n=n,price=p['price'],stderr=p['stderr'],
                                               ci95=p['ci95'],analytic=exact,error=p['price']-exact))
    # CIR zero-coupon bond: no changing observation schedule confounder.
    k,th,vol,r0=1.2,.05,.1,.08
    g=np.sqrt(k*k+2*vol*vol)
    den=(g+k)*np.expm1(g)+2*g
    exact=float((2*g*np.exp((k+g)/2)/den)**(2*k*th/vol**2)*np.exp(-2*np.expm1(g)/den*r0))
    for steps in (12,52,252):
        s=simulate_paths(domestic=CIRParams(k,th,vol),foreign=CIRParams(1,.03,0),
                         sigma_fx=.2,corr=np.eye(3),r_domestic_0=r0,r_foreign_0=.03,
                         spot=100,dt=1/steps,n_sims=32000,seed=2026)
        p=RangeAccrualPricer(notional=1).price_from_paths(s)
        report['time_step_study'].append(dict(steps=steps,n=32000,price=p['price'],
                                             stderr=p['stderr'],analytic=exact,error=p['price']-exact))
    print(json.dumps(report,indent=2))

if __name__=='__main__': run()
