"""Step 4b: GPD fit to declustered daily exceedances; return levels comparable to the GEV.

Return level x_T: the level with annual exceedance probability 1/T. With independent
cluster peaks arriving as a Poisson process (rate lam per year) and GPD excesses,
  lam * P(X > x_T | X > u) = -log(1 - 1/T),
which is the same definition the GEV return levels use.

CIs: profile likelihood (rate lam treated as known) and a parametric bootstrap that
also resamples the number of clusters ~ Poisson, so rate uncertainty is included.

Outputs: results/04b_*.csv, figures/04b_pot_diagnostics.png
"""

import numpy as np
import pandas as pd

import config as cfg
import evtlib as ev
from common import C1, C2, C3, FIGURES, INK, INK2, PROCESSED, RESULTS, daily_series, plt, to_obs_precision

T = np.array(cfg.RETURN_PERIODS, float)
rng = np.random.default_rng(cfg.SEED + 4)
s, n_years = daily_series(cfg.MIN_COVERAGE)
v = s.to_numpy()


def pot_analysis(u):
    peaks = v[ev.decluster_runs(v, u, cfg.RUN_LENGTH)]
    y = peaks - u
    fit = ev.gpd_fit_mle(y)
    lam = len(y) / n_years
    se = np.sqrt(np.diag(ev.gpd_mle_cov(y, fit)))
    boot = np.empty((cfg.N_BOOT, len(T)))
    for b in range(cfg.N_BOOT):
        nb = rng.poisson(len(y))
        fb = ev.gpd_fit_mle(ev.gpd_rvs(nb, fit["sigma"], fit["xi"], rng))
        boot[b] = ev.pot_return_level(T, u, fb["sigma"], fb["xi"], nb / n_years)
    rows = []
    for k, Tk in enumerate(T):
        x_hat, lo, hi = ev.pot_profile_return_level(y, fit, u, lam, Tk)
        blo, bhi = np.percentile(boot[:, k], [2.5, 97.5])
        rows.append({"u_mm": u, "T_years": Tk, "estimate_mm": x_hat,
                     "profile_lo95": lo, "profile_hi95": hi, "boot_lo95": blo, "boot_hi95": bhi})
    summary = {"u_mm": u, "n_clusters": len(y), "rate_per_year": lam,
               "sigma": fit["sigma"], "sigma_se": se[0], "xi": fit["xi"], "xi_se": se[1],
               "converged": fit["converged"]}
    return y, fit, lam, summary, rows


print(f"Effective record: {n_years:.1f} years; run length {cfg.RUN_LENGTH} days\n")
results, summaries, rl_rows = {}, [], []
for u in [cfg.POT_THRESHOLD] + cfg.POT_SENS_THRESHOLDS:
    y, fit, lam, summ, rows = pot_analysis(u)
    results[u] = (y, fit, lam)
    summaries.append(summ)
    rl_rows += rows
summ = pd.DataFrame(summaries)
rl = pd.DataFrame(rl_rows)
summ.to_csv(RESULTS / "04b_gpd_params.csv", index=False)
rl.to_csv(RESULTS / "04b_return_levels.csv", index=False)
print("GPD fits:")
print(summ.round(4).to_string(index=False))
print(f"\nPOT return levels (mm/day), bootstrap B={cfg.N_BOOT}:")
print(rl.round(1).to_string(index=False))

# GEV vs POT at the main threshold.
gev = pd.read_csv(RESULTS / "02_return_levels.csv").query("method == 'MLE + profile'")
main = rl.query("u_mm == @cfg.POT_THRESHOLD").set_index("T_years")
cmp = gev.set_index("T_years")[["estimate_mm", "lo95_mm", "hi95_mm"]].rename(
    columns=lambda c: "GEV_" + c)
cmp["POT_estimate_mm"] = main["estimate_mm"]
cmp["POT_lo95_mm"] = main["profile_lo95"]
cmp["POT_hi95_mm"] = main["profile_hi95"]
cmp["POT_vs_GEV_pct"] = 100 * (cmp["POT_estimate_mm"] / cmp["GEV_estimate_mm"] - 1)
cmp.to_csv(RESULTS / "04b_gev_vs_pot.csv")
print(f"\nGEV (annual maxima) vs POT (u = {cfg.POT_THRESHOLD:g} mm), profile 95% CIs:")
print(cmp.round(1).to_string())

# ------------------------------------------------------------- GoF at main threshold
# Observations are recorded to 0.01 inch, so 1183 peaks take only ~230 distinct values.
# With this many points AD/KS detect the ties, so a second null distribution rounds each
# simulated sample to the same precision (and drops values that round down to u).
y, fit, lam = results[cfg.POT_THRESHOLD]
u = cfg.POT_THRESHOLD
gpd_cdf = lambda z, f: ev.gpd_cdf(z, f["sigma"], f["xi"])  # noqa: E731


def rvs_rounded(m, f, r):
    z = to_obs_precision(u + ev.gpd_rvs(m, f["sigma"], f["xi"], r)) - u
    return z[z > 0]


gof_rows = []
for null, rvs in (("continuous", lambda m, f, r: ev.gpd_rvs(m, f["sigma"], f["xi"], r)),
                  ("rounded to 0.01 in", rvs_rounded)):
    gof = ev.gof_parametric_bootstrap(y, gpd_cdf, rvs, ev.gpd_fit_mle, cfg.N_BOOT, rng)
    gof_rows.append({"u_mm": u, "null": null, **gof})
    print(f"\nGPD GoF (u = {u:g}), {null} null: A2={gof['A2']:.3f} (bootstrap p={gof['p_A2']:.3f}); "
          f"KS D={gof['D']:.4f} (bootstrap p={gof['p_D']:.3f}; known-parameter p={gof['D_naive_p']:.3f})")
pd.DataFrame(gof_rows).to_csv(RESULTS / "04b_gof.csv", index=False)
F = np.sort(gpd_cdf(y, fit))
ys_ = np.sort(y)
i = np.arange(1, len(y) + 1)
k = np.argmax(np.maximum(i / len(y) - F, F - (i - 1) / len(y)))
print(f"Distinct peak values: {len(np.unique(y))} of {len(y)}; "
      f"KS distance is largest at {u + ys_[k]:.1f} mm (just above the threshold)")

# ------------------------------------------------------------------------ figure
ys = np.sort(y)
m = len(ys)
pp = np.arange(1, m + 1) / (m + 1)
fig, ax = plt.subplots(2, 2, figsize=(9.5, 8))
a = ax[0, 0]
a.plot([0, 1], [0, 1], color=INK2, lw=1)
a.plot(pp, ev.gpd_cdf(ys, fit["sigma"], fit["xi"]), "o", ms=3, color=C1)
a.set(xlabel="empirical", ylabel="model", title=f"GPD probability plot (u = {cfg.POT_THRESHOLD:g} mm)")
a = ax[0, 1]
qm = cfg.POT_THRESHOLD + ev.gpd_ppf(pp, fit["sigma"], fit["xi"])
a.plot([qm.min(), qm.max()], [qm.min(), qm.max()], color=INK2, lw=1)
a.plot(qm, cfg.POT_THRESHOLD + ys, "o", ms=3, color=C1)
a.set(xlabel="model quantile (mm)", ylabel="empirical (mm)", title="GPD quantile plot")

a = ax[1, 0]
Tg = np.geomspace(1.5, 500, 200)
am = np.sort(pd.read_csv(PROCESSED / "annual_max.csv")["annual_max_mm"].to_numpy())
gpar = pd.read_csv(RESULTS / "02_gev_params.csv", index_col=0)["MLE"]
a.plot(Tg, ev.gev_return_level(Tg, gpar["mu"], gpar["sigma"], gpar["xi"]), color=C1,
       label="GEV, annual maxima")
for u, color, ls in zip([cfg.POT_THRESHOLD] + cfg.POT_SENS_THRESHOLDS, [C2, C3, INK2], ["-", "--", ":"]):
    yu, fu, lu = results[u]
    a.plot(Tg, ev.pot_return_level(Tg, u, fu["sigma"], fu["xi"], lu), color=color, ls=ls, lw=1.6,
           label=f"GPD, u = {u:g} mm")
a.plot(1 / (1 - np.arange(1, len(am) + 1) / (len(am) + 1)), am, "o", ms=3.5, color=INK,
       label="observed annual max")
a.set_xscale("log")
a.set_xlim(1.5, 500)
a.set_ylim(50, None)
a.set(xlabel="return period (years)", ylabel="return level (mm/day)", title="Return levels: GEV vs POT")
a.legend(loc="upper left", fontsize=8)

a = ax[1, 1]
T100 = T.max()
labels, est, lo, hi = ["GEV"], [], [], []
g100 = cmp.loc[T100]
est.append(g100["GEV_estimate_mm"]); lo.append(g100["GEV_lo95_mm"]); hi.append(g100["GEV_hi95_mm"])
for u in [cfg.POT_THRESHOLD] + cfg.POT_SENS_THRESHOLDS:
    r = rl.query("u_mm == @u and T_years == @T100").iloc[0]
    labels.append(f"GPD u={u:g}")
    est.append(r["estimate_mm"]); lo.append(r["profile_lo95"]); hi.append(r["profile_hi95"])
pos = np.arange(len(labels))
est, lo, hi = map(np.array, (est, lo, hi))
a.errorbar(pos, est, yerr=[est - lo, hi - est], fmt="o", ms=7, color=C1, ecolor=INK2, elinewidth=1.5)
for p_, e_ in zip(pos, est):
    a.text(p_ + 0.08, e_, f"{e_:.0f}", va="center", fontsize=9, color=INK)
a.set_xticks(pos)
a.set_xticklabels(labels)
a.set_xlim(-0.5, len(labels) - 0.3)
a.set(ylabel="100-yr return level (mm/day)", title="100-yr level, profile 95% CIs")
fig.suptitle(f"{cfg.STATION_NAME}: peaks over threshold (runs declustering, r = {cfg.RUN_LENGTH} d)", y=1.0)
fig.tight_layout()
fig.savefig(FIGURES / "04b_pot_diagnostics.png")
print(f"\nSaved {FIGURES / '04b_pot_diagnostics.png'}")
