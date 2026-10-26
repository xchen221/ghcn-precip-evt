"""Step 4a: threshold and declustering diagnostics for peaks over threshold (POT).

For a threshold u to be valid, excesses over u should follow a GPD. Two checks:
  mean residual life  E[X - u | X > u] is linear in u above a valid threshold
  parameter stability  xi and the modified scale sigma* = sigma_u - xi * u stay
                       constant (within CIs) above a valid threshold
Daily values cluster in multi-day storms, so exceedances are declustered with the
runs method before fitting; the table shows how the run length changes cluster counts.

Outputs: results/04a_threshold_table.csv, figures/04a_threshold_choice.png
"""

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg
import evtlib as ev
from common import C1, FIGURES, INK2, RESULTS, daily_series, plt

s, n_years = daily_series(cfg.MIN_COVERAGE)
v = s.to_numpy()
print(f"Daily series: {np.isfinite(v).sum()} valid days = {n_years:.1f} effective years")

# ------------------------------------------------------------ mean residual life
vals = v[np.isfinite(v)]
mrl = ev.mean_residual_life(vals, np.arange(5, 121, 1.0))

# -------------------------------------------- parameter stability, declustered
rows = []
for u in np.arange(20, 82, 2.0):
    counts = {r: len(ev.decluster_runs(v, u, r)) for r in (1, 2, 3)}
    peaks = v[ev.decluster_runs(v, u, cfg.RUN_LENGTH)]
    y = peaks - u
    fit = ev.gpd_fit_mle(y)
    V = ev.gpd_mle_cov(y, fit)
    grad = np.array([1.0, -u])  # sigma* = sigma - xi * u
    rows.append({
        "u_mm": u,
        "n_exceed_days": int(np.sum(vals > u)),
        "clusters_r1": counts[1], "clusters_r2": counts[2], "clusters_r3": counts[3],
        "per_year_r2": counts[2] / n_years,
        "xi": fit["xi"], "xi_se": np.sqrt(V[1, 1]),
        "sigma_star": fit["sigma"] - fit["xi"] * u, "sigma_star_se": np.sqrt(grad @ V @ grad),
    })
tab = pd.DataFrame(rows)
tab["theta_r2"] = tab["clusters_r2"] / tab["n_exceed_days"]  # extremal-index estimate
tab.to_csv(RESULTS / "04a_threshold_table.csv", index=False)
print("\nThreshold table (GPD fits use the configured run length):")
print(tab.round(3).to_string(index=False))

# ----------------------------------------------------------------------- figure
z = stats.norm.ppf(0.975)
fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
a = ax[0]
a.fill_between(mrl[:, 0], mrl[:, 1] - mrl[:, 2], mrl[:, 1] + mrl[:, 2], color=C1, alpha=0.18, lw=0)
a.plot(mrl[:, 0], mrl[:, 1], color=C1)
a.set(ylabel="mean excess (mm)", title="Mean residual life (all valid days)")
a = ax[1]
a.errorbar(tab["u_mm"], tab["sigma_star"], yerr=z * tab["sigma_star_se"], fmt="o", ms=4,
           color=C1, ecolor=INK2, elinewidth=1, capsize=0)
a.set(ylabel="modified scale σ*", title="Parameter stability (declustered peaks)")
a = ax[2]
a.errorbar(tab["u_mm"], tab["xi"], yerr=z * tab["xi_se"], fmt="o", ms=4,
           color=C1, ecolor=INK2, elinewidth=1, capsize=0)
a.axhline(0, color=INK2, lw=0.8)
a.set(ylabel="shape ξ", xlabel="threshold u (mm/day)")
ax2 = a.secondary_xaxis("top")
ax2.set_xticks(tab["u_mm"][::5])
ax2.set_xticklabels([f"{c}" for c in tab["clusters_r2"][::5]], color=INK2, fontsize=8)
ax2.set_xlabel("number of clusters", color=INK2, fontsize=8)
for a in ax:
    a.set_xlim(5, 121)
fig.tight_layout()
fig.savefig(FIGURES / "04a_threshold_choice.png")
print(f"\nSaved {FIGURES / '04a_threshold_choice.png'}")
