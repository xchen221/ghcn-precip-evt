"""Step 3: goodness of fit for the stationary GEV.

Graphical: probability (PP), quantile (QQ), return-level and density plots (Coles 2001, Fig. 3.5 layout).
Formal: Anderson-Darling A^2 (tail-weighted) and Kolmogorov-Smirnov D. Because the
parameters are estimated from the same data, p-values come from a parametric
bootstrap that refits the GEV on every replicate; the textbook KS p-value (which
assumes known parameters) is printed for contrast.

Outputs: results/03_gof.csv, figures/03_gev_diagnostics.png
"""

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg
import evtlib as ev
from common import C1, C2, FIGURES, INK, INK2, PROCESSED, RESULTS, plt

rng = np.random.default_rng(cfg.SEED + 3)
am = pd.read_csv(PROCESSED / "annual_max.csv", index_col="year")["annual_max_mm"]
x = np.sort(am.to_numpy())
n = len(x)

fitters = {
    "MLE": ev.gev_fit_mle,
    "L-moments": ev.gev_fit_lmom,
}
cdf = lambda v, f: ev.gev_cdf(v, f["mu"], f["sigma"], f["xi"])  # noqa: E731
rvs = lambda m, f, r: ev.gev_rvs(m, f["mu"], f["sigma"], f["xi"], r)  # noqa: E731

rows = []
for name, fitter in fitters.items():
    g = ev.gof_parametric_bootstrap(x, cdf, rvs, fitter, cfg.N_BOOT, rng)
    rows.append({"fit": name, **g})
    print(f"GEV ({name}): A2={g['A2']:.3f} (bootstrap p={g['p_A2']:.3f}); "
          f"KS D={g['D']:.4f} (bootstrap p={g['p_D']:.3f}; known-parameter p={g['D_naive_p']:.3f})")

# Same tests against the Gumbel model, to see whether GoF tests can tell the two apart.
gum_fit = lambda v: ev.gev_fit_gumbel(v)  # noqa: E731
gum_rvs = lambda m, f, r: ev.gev_rvs(m, f["mu"], f["sigma"], 0.0, r)  # noqa: E731
g = ev.gof_parametric_bootstrap(x, cdf, gum_rvs, gum_fit, cfg.N_BOOT, rng)
rows.append({"fit": "Gumbel (xi=0)", **g})
print(f"Gumbel: A2={g['A2']:.3f} (bootstrap p={g['p_A2']:.3f}); "
      f"KS D={g['D']:.4f} (bootstrap p={g['p_D']:.3f}; known-parameter p={g['D_naive_p']:.3f})")
pd.DataFrame(rows).to_csv(RESULTS / "03_gof.csv", index=False)

# ------------------------------------------------------------ diagnostic plots
mle = ev.gev_fit_mle(x)
mu, sigma, xi = mle["mu"], mle["sigma"], mle["xi"]
pp = np.arange(1, n + 1) / (n + 1)  # Weibull plotting positions, as in Coles (2001)
boot = np.load(RESULTS / "02_boot_mle.npy")  # bootstrap refits from step 2: mu, sigma, xi, ...

fig, ax = plt.subplots(2, 2, figsize=(9.5, 8))
a = ax[0, 0]
a.plot([0, 1], [0, 1], color=INK2, lw=1)
a.plot(pp, cdf(x, mle), "o", ms=4, color=C1)
a.set(xlabel="empirical", ylabel="model", title="Probability plot")

a = ax[0, 1]
q_model = ev.gev_ppf(pp, mu, sigma, xi)
a.plot([x.min(), x.max()], [x.min(), x.max()], color=INK2, lw=1)
a.plot(q_model, x, "o", ms=4, color=C1)
a.set(xlabel="model quantile (mm)", ylabel="empirical (mm)", title="Quantile plot")

a = ax[1, 0]
Tg = np.geomspace(1.05, 500, 200)
band = np.array([ev.gev_return_level(Tg, *b[:3]) for b in boot])
lo, hi = np.percentile(band, [2.5, 97.5], axis=0)
a.fill_between(Tg, lo, hi, color=C1, alpha=0.18, lw=0, label="95% bootstrap band")
a.plot(Tg, ev.gev_return_level(Tg, mu, sigma, xi), color=C1, label="GEV (MLE)")
gum = ev.gev_fit_gumbel(x)
a.plot(Tg, ev.gev_return_level(Tg, gum["mu"], gum["sigma"], 0.0),
       color=C2, ls="--", lw=1.5, label="Gumbel (ξ = 0)")
a.plot(1 / (1 - pp), x, "o", ms=4, color=INK, label="observed")
a.set_xscale("log")
a.set(xlabel="return period (years)", ylabel="return level (mm/day)", title="Return-level plot")
a.axvline(n, color=INK2, lw=0.8, ls=":")
a.text(n * 1.05, a.get_ylim()[0] + 5, "record length", color=INK2, fontsize=8)
a.legend(loc="upper left")

a = ax[1, 1]
grid = np.linspace(x.min() * 0.8, x.max() * 1.1, 300)
pdf = stats.genextreme.pdf(grid, -xi, mu, sigma)  # scipy shape c = -xi
a.hist(x, bins=18, density=True, color="#c9c8c3", edgecolor="white")
a.plot(grid, pdf, color=C1)
a.set(xlabel="annual max daily precip (mm)", ylabel="density", title="Density")

fig.suptitle(f"{cfg.STATION_NAME}: stationary GEV diagnostics (n={n})", y=1.0)
fig.tight_layout()
fig.savefig(FIGURES / "03_gev_diagnostics.png")
print(f"Saved {FIGURES / '03_gev_diagnostics.png'}")
