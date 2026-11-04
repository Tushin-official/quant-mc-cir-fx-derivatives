import unittest
import numpy as np
from scipy.special import ndtr
from source.simulation import CIRParams, simulate_paths, correlation_factor
from source.pricing import RangeAccrualPricer
from source.core import Preprocess


def scenario(**overrides):
    kw = dict(domestic=CIRParams(1,.06,0), foreign=CIRParams(1,.03,0),
              sigma_fx=.2, corr=np.eye(3), r_domestic_0=.06, r_foreign_0=.03,
              spot=100., dt=1/12, horizon=1., n_sims=2000, seed=42)
    kw.update(overrides)
    return simulate_paths(**kw)

class EngineTests(unittest.TestCase):
    def test_discount_coupon(self):
        r = RangeAccrualPricer(notional=100,coupon_rate=.1,accrual_years=.5).price_from_paths(scenario())
        self.assertAlmostEqual(r['price'],5*np.exp(-.06),places=12)
        self.assertLess(r['stderr'],1e-13)

    def test_zero_occupancy(self):
        r = RangeAccrualPricer(200,300,100).price_from_paths(scenario(sigma_fx=0))
        self.assertEqual(r['price'],0)

    def test_observations(self):
        s = scenario(n_sims=2,dt=.5)
        s['S_paths'][:] = [100,200,100]
        p = RangeAccrualPricer(99,101,100)
        self.assertAlmostEqual(p.price_from_paths(s)['w_paths'][0],.5)
        self.assertAlmostEqual(p.price_from_paths(s,[1,2],[1,3])['w_paths'][0],.75)
        self.assertEqual(p.price_from_paths(s,[2])['w_paths'][0],1)

    def test_noninteger_grid(self):
        s = scenario(horizon=.95,dt=.3,sigma_fx=0)
        self.assertEqual(s['t_grid'][-1],.95)
        np.testing.assert_allclose(s['S_paths'][:,-1],100*np.exp(.03*.95),rtol=1e-14)
        self.assertAlmostEqual(RangeAccrualPricer(notional=100).price_from_paths(s)['price'],100*np.exp(-.06*.95),places=12)

    def test_seed_global_rng(self):
        np.random.seed(3)
        expected = np.random.random(4)
        np.random.seed(3)
        a,b = scenario(),scenario()
        np.testing.assert_array_equal(a['S_paths'],b['S_paths'])
        np.testing.assert_array_equal(np.random.random(4),expected)
        self.assertFalse(np.array_equal(a['S_paths'],scenario(seed=2)['S_paths']))

    def test_correlations(self):
        for c in [np.eye(3),np.ones((3,3)),np.array([[1,.2,-.1],[.2,1,.4],[-.1,.4,1]])]:
            f=correlation_factor(c)
            np.testing.assert_allclose(f@f.T,c,atol=1e-12)
        for c in [np.array([[1,2,0],[2,1,0],[0,0,1]]),np.zeros((3,3)),np.full((3,3),np.nan)]:
            with self.assertRaises(ValueError): correlation_factor(c)

    def test_inputs(self):
        for kw in [dict(n_sims=1),dict(n_sims=2.5),dict(dt=0),dict(spot=-1),dict(horizon=np.nan),dict(sigma_fx=-1)]:
            with self.assertRaises(ValueError): scenario(**kw)
        with self.assertRaises(ValueError): CIRParams(-1,.1,.2)
        p,s=RangeAccrualPricer(notional=100),scenario()
        for idx in [[0],[1,1],[2,1],[1.5],[],[100]]:
            with self.assertRaises(ValueError): p.price_from_paths(s,idx)
        with self.assertRaises(ValueError): p.price_from_paths(s,[1],[-1])
        with self.assertRaises(ValueError): RangeAccrualPricer(2,1,100)
        with self.assertRaises(ValueError): p.price_from_paths({**s,'measure':'physical_p'})
        with self.assertRaises(ValueError): p.price_from_paths({**s,'fx_quote':'USD_per_RUB'})

    def test_analytic_range_accrual(self):
        s=scenario(n_sims=50000)
        t=s['t_grid'][1:]
        prob=ndtr((np.log(1.1)-(.03-.5*.2**2)*t)/(.2*np.sqrt(t)))-ndtr((np.log(.9)-(.03-.5*.2**2)*t)/(.2*np.sqrt(t)))
        exact=100*np.exp(-.06)*prob.mean()
        p=RangeAccrualPricer(90,110,100).price_from_paths(s)
        self.assertLess(abs(p['price']-exact),5*p['stderr'])

    def test_cir_bond_mean(self):
        k,th,vol,r0=1.2,.05,.1,.08
        s=scenario(domestic=CIRParams(k,th,vol),r_domestic_0=r0,dt=1/252,n_sims=20000)
        g=np.sqrt(k*k+2*vol*vol)
        den=(g+k)*np.expm1(g)+2*g
        B=2*np.expm1(g)/den
        A=(2*g*np.exp((k+g)/2)/den)**(2*k*th/vol**2)
        p=RangeAccrualPricer(notional=1).price_from_paths(s)
        self.assertLess(abs(p['price']-A*np.exp(-B*r0)),5*p['stderr']+1e-4)
        x=s['r_rub_paths'][:,-1]
        self.assertLess(abs(x.mean()-(th+(r0-th)*np.exp(-k))),5*x.std()/np.sqrt(len(x))+1e-4)

    def test_foreign_account_martingale(self):
        c=np.array([[1,.2,-.1],[.2,1,.4],[-.1,.4,1]])
        s=scenario(domestic=CIRParams(1,.06,.1),foreign=CIRParams(1,.03,.1),corr=c,n_sims=20000,dt=.01)
        x=s['S_paths'][:,-1]*np.exp((s['r_usd_paths'][:,:-1]-s['r_rub_paths'][:,:-1])@np.diff(s['t_grid']))
        self.assertLess(abs(x.mean()-100),5*x.std(ddof=1)/np.sqrt(len(x)))

    def test_quanto_drift(self):
        s=scenario(foreign=CIRParams(1,.03,.3),corr=np.ones((3,3)),dt=.1,horizon=.1,n_sims=100000,r_foreign_0=.2)
        exact=.2+((.03-.2)-.3*.2*np.sqrt(.2))*.1
        x=s['r_usd_paths'][:,-1]
        self.assertLess(abs(x.mean()-exact),5*x.std(ddof=1)/np.sqrt(len(x)))

    def test_stress_positivity(self):
        s=scenario(domestic=CIRParams(.1,.001,1),foreign=CIRParams(.1,.001,1),dt=.1)
        self.assertTrue((s['r_rub_paths']>=0).all())
        self.assertTrue((s['r_usd_paths']>=0).all())
        self.assertTrue((s['S_paths']>0).all())

    def test_legacy(self):
        p=Preprocess(None,None,None,None,None,None,None,None,None,None,.1,10,1.,seed=42)
        p.params=dict(cir_rub={k:[v] for k,v in zip(('kappa','theta','sigma'),(1,.06,0))},cir_usd={k:[v] for k,v in zip(('kappa','theta','sigma'),(1,.03,0))},fx={'sigma':[.2]},corr_matrix=np.eye(3))
        p.start_values=dict(r_rub_0=.06,r_usd_0=.03,S_0=100)
        with self.assertWarns(UserWarning): p.simulate()
        with self.assertWarns(UserWarning): r=RangeAccrualPricer(notional=100).price_from_paths(p.get_simulation_result())
        self.assertEqual(r['parameter_basis'],'historical_proxy')
        self.assertAlmostEqual(r['price'],100*np.exp(-.06))

if __name__=='__main__': unittest.main()
