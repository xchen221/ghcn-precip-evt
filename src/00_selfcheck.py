"""Step 0: check evtlib on simulated data with known parameters, and against scipy.

Sign conventions: scipy.stats.genextreme has c = -xi; scipy.stats.genpareto has c = +xi.
"""

import numpy as np
from scipy import stats

import config as cfg
import evtlib as ev

rng = np.random.default_rng(cfg.SEED)
true = {"mu": 50.0, "sigma": 15.0, "xi": 0.15}
x = ev.gev_rvs(140, true["mu"], true["sigma"], true["xi"], rng)

ours = ev.gev_fit_mle(x)
c, loc, scale = stats.genextreme.fit(x, -ours["xi"], loc=ours["mu"], scale=ours["sigma"])
print("GEV on one simulated sample (n=140), truth mu=50 sigma=15 xi=0.15")
print(f"  evtlib MLE      mu={ours['mu']:.3f} sigma={ours['sigma']:.3f} xi={ours['xi']:.4f}")
print(f"  scipy MLE       mu={loc:.3f} sigma={scale:.3f} xi=-c={-c:.4f}")
lm = ev.gev_fit_lmom(x)
print(f"  evtlib L-mom    mu={lm['mu']:.3f} sigma={lm['sigma']:.3f} xi={lm['xi']:.4f}")
assert abs(ours["xi"] + c) < 1e-3 and abs(ours["mu"] - loc) < 1e-2, "MLE disagrees with scipy"

# Return level: closed form vs scipy quantile with the sign flipped.
z100 = ev.gev_return_level(100, ours["mu"], ours["sigma"], ours["xi"])
z100_scipy = stats.genextreme.ppf(0.99, -ours["xi"], ours["mu"], ours["sigma"])
print(f"  100-yr level: evtlib {z100:.3f}, scipy {z100_scipy:.3f}")
assert abs(z100 - z100_scipy) < 1e-6

# Sampling behaviour of the two estimators over repeated samples.
reps = np.array([[ev.gev_fit_mle(xb)["xi"], ev.gev_fit_lmom(xb)["xi"]]
                 for xb in (ev.gev_rvs(140, **true, rng=rng) for _ in range(300))])
print("\nxi over 300 simulated samples of n=140 (truth 0.15):")
for name, col in zip(("MLE", "L-moments"), reps.T):
    print(f"  {name:10s} mean={col.mean():.3f}  sd={col.std():.3f}")

# GPD: evtlib vs scipy.stats.genpareto (same sign convention for the shape).
y = stats.genpareto.rvs(0.1, scale=10.0, size=400, random_state=rng)
g = ev.gpd_fit_mle(y)
c2, _, s2 = stats.genpareto.fit(y, floc=0)
print(f"\nGPD (truth sigma=10 xi=0.1): evtlib sigma={g['sigma']:.3f} xi={g['xi']:.4f}; "
      f"scipy sigma={s2:.3f} xi=c={c2:.4f}")
assert abs(g["xi"] - c2) < 2e-3
print("\nAll checks passed.")
