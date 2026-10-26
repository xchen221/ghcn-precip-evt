"""Step 2: fit a GEV to annual maxima by MLE and L-moments; return levels with CIs.

CIs for return levels, three ways:
  delta method      MLE +/- 1.96 se, se from the observed information (symmetric)
  profile likelihood  {z: 2 (l_max - l_p(z)) <= chi2_1(0.95)}; main result
  parametric bootstrap  percentiles of refitted return levels
Also: likelihood-ratio test of xi = 0 (Gumbel) and a coverage-rule sensitivity check.

Outputs: data/processed/annual_max.csv, results/02_*.csv, figures/02_profiles.png
"""

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg
import evtlib as ev
from common import C1, C2, FIGURES, INK2, PROCESSED, RESULTS, plt

T = np.array(cfg.RETURN_PERIODS, float)
rng = np.random.default_rng(cfg.SEED)

yearly = pd.read_csv(RESULTS / "01_yearly_coverage.csv", index_col="year")


def annual_max(min_cov):
    keep = yearly[yearly["coverage"] >= min_cov]
    return keep["raw_max_mm"]


am = annual_max(cfg.MIN_COVERAGE)
am.rename("annual_max_mm").to_csv(PROCESSED / "annual_max.csv")
x = am.to_numpy()
n = len(x)
print(f"Annual maxima: n={n} years ({am.index.min()}-{am.index.max()}), coverage >= {cfg.MIN_COVERAGE}")
print(f"  mean={x.mean():.1f} mm  sd={x.std(ddof=1):.1f}  min={x.min():.1f}  max={x.max():.1f}\n")

# ---------------------------------------------------------------- parameter estimates
mle = ev.gev_fit_mle(x)
lmom = ev.gev_fit_lmom(x)
gum = ev.gev_fit_gumbel(x)
se = np.sqrt(np.diag(ev.gev_mle_cov(x, mle)))
xi_lo, xi_hi = ev.gev_profile_xi(x, mle)

lr = 2 * (mle["loglik"] - gum["loglik"])
p_lr = stats.chi2.sf(lr, 1)

params = pd.DataFrame({
    "MLE": [mle["mu"], mle["sigma"], mle["xi"], mle["loglik"]],
    "MLE_se": [se[0], se[1], se[2], np.nan],
    "L-moments": [lmom["mu"], lmom["sigma"], lmom["xi"], ev.gev_loglik(x, lmom["mu"], lmom["sigma"], lmom["xi"])],
    "Gumbel_MLE": [gum["mu"], gum["sigma"], 0.0, gum["loglik"]],
}, index=["mu", "sigma", "xi", "loglik"])
params.to_csv(RESULTS / "02_gev_params.csv")
print("GEV parameters (xi > 0 = heavy upper tail):")
print(params.round(4).to_string())
print(f"\nProfile 95% CI for xi: [{xi_lo:.3f}, {xi_hi:.3f}]")
print(f"LR test xi=0 (Gumbel) vs GEV: D={lr:.2f}, p={p_lr:.4f}  (chi2_1)")
print(f"MLE converged: {mle['converged']}\n")

# ------------------------------------------------------------------ return levels
T_grid = np.unique(np.concatenate([T, np.geomspace(1.05, 200, 60)]))
boot_mle = ev.gev_parametric_bootstrap(mle, n, cfg.N_BOOT, T_grid, rng)
boot_lm = ev.gev_parametric_bootstrap(lmom, n, cfg.N_BOOT, T_grid, rng, fitter=ev.gev_fit_lmom)
np.save(RESULTS / "02_boot_mle.npy", boot_mle)
idx = np.searchsorted(T_grid, T)

rows = []
for k, Tk in enumerate(T):
    z_d, se_d = ev.gev_return_level_delta(Tk, x, mle)
    z_p, lo_p, hi_p = ev.gev_profile_return_level(x, mle, Tk)
    b = boot_mle[:, 3 + idx[k]]
    bl = boot_lm[:, 3 + idx[k]]
    z_l = ev.gev_return_level(Tk, lmom["mu"], lmom["sigma"], lmom["xi"])
    rows += [
        (Tk, "MLE + delta", z_d, z_d - 1.96 * se_d, z_d + 1.96 * se_d),
        (Tk, "MLE + profile", z_p, lo_p, hi_p),
        (Tk, "MLE + bootstrap", z_p, *np.percentile(b, [2.5, 97.5])),
        (Tk, "L-mom + bootstrap", z_l, *np.percentile(bl, [2.5, 97.5])),
    ]
rl = pd.DataFrame(rows, columns=["T_years", "method", "estimate_mm", "lo95_mm", "hi95_mm"])
rl["width_mm"] = rl["hi95_mm"] - rl["lo95_mm"]
rl.to_csv(RESULTS / "02_return_levels.csv", index=False)
print(f"Return levels (mm/day) with 95% CIs, bootstrap B={cfg.N_BOOT}:")
print(rl.round(1).to_string(index=False))

# Coverage-rule sensitivity: refit on the strict sample.
x99 = annual_max(cfg.SENS_COVERAGE).to_numpy()
mle99 = ev.gev_fit_mle(x99)
sens = []
for Tk in T:
    z99, lo99, hi99 = ev.gev_profile_return_level(x99, mle99, Tk)
    z90 = rl.query("T_years == @Tk and method == 'MLE + profile'")["estimate_mm"].item()
    sens.append((Tk, z90, z99, lo99, hi99, 100 * (z99 - z90) / z90))
sens = pd.DataFrame(sens, columns=["T_years", f"z_cov{cfg.MIN_COVERAGE}", f"z_cov{cfg.SENS_COVERAGE}",
                                   "lo95_strict", "hi95_strict", "pct_change"])
sens.to_csv(RESULTS / "02_sensitivity_coverage.csv", index=False)
print(f"\nSensitivity: coverage >= {cfg.SENS_COVERAGE} (n={len(x99)}): "
      f"mu={mle99['mu']:.2f} sigma={mle99['sigma']:.2f} xi={mle99['xi']:.3f}")
print(sens.round(2).to_string(index=False))

# ---------------------------------------------------------------- profile figure
crit = stats.chi2.ppf(0.95, 1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
dev_xi = ev.gev_profile_deviance_xi(x, mle)
xs = np.linspace(mle["xi"] - 0.25, mle["xi"] + 0.3, 81)
a1.plot(xs, [dev_xi(v) for v in xs], color=C1)
a1.axhline(crit, color=INK2, lw=1, ls="--")
a1.axvline(0, color=INK2, lw=0.8)
a1.set_ylim(0, 3 * crit)
a1.set_xlabel("shape ξ")
a1.set_ylabel("profile deviance")
a1.set_title(f"ξ: MLE {mle['xi']:.3f}, 95% CI [{xi_lo:.3f}, {xi_hi:.3f}]")

T100 = T.max()
r100 = rl.query("T_years == @T100").set_index("method")
z_hat = r100.loc["MLE + profile", "estimate_mm"]
dev_z = ev.gev_profile_deviance_rl(x, mle, T100)
zs = np.linspace(0.75 * z_hat, 1.9 * z_hat, 81)
a2.plot(zs, [dev_z(v) for v in zs], color=C1, label="profile likelihood")
a2.axhline(crit, color=INK2, lw=1, ls="--")
lo_d, hi_d = r100.loc["MLE + delta", ["lo95_mm", "hi95_mm"]]
a2.plot([lo_d, hi_d], [crit, crit], "|-", color=C2, ms=12, lw=2, label="delta-method 95% CI")
a2.set_ylim(0, 3 * crit)
a2.set_xlabel(f"{int(T100)}-yr return level (mm/day)")
a2.set_title(f"{int(T100)}-yr level: {z_hat:.0f} mm, profile CI "
             f"[{r100.loc['MLE + profile', 'lo95_mm']:.0f}, {r100.loc['MLE + profile', 'hi95_mm']:.0f}]")
a2.legend(loc="upper right")
fig.suptitle("Dashed line: χ²₁ 95% cutoff (3.84)", y=1.02, fontsize=9, color=INK2)
fig.savefig(FIGURES / "02_profiles.png")
print(f"\nSaved {FIGURES / '02_profiles.png'}")
