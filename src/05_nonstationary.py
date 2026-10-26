"""Step 5: non-stationary GEV with global mean surface temperature (GMST) as covariate.

M0: stationary GEV (step 2).  M1: mu(t) = mu0 + mu1 * GMST'(t), sigma and xi constant.
GMST' is the NASA GISTEMP v4 annual global anomaly (deg C vs 1951-1980), smoothed with a
4-year trailing mean to damp ENSO noise (as in World Weather Attribution studies).
Test: likelihood ratio D = 2 (l1 - l0) ~ chi2_1. Diagnostics: residuals transformed to
the standard Gumbel scale (Coles 2001, sec. 6.2.3). CIs: parametric bootstrap from M1.

Outputs: data/raw/GLB.Ts+dSST.csv, results/05_*.csv, figures/05_nonstationary.png
"""

import hashlib
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg
import evtlib as ev
from common import C1, C2, FIGURES, INK, INK2, PROCESSED, RAW, RESULTS, plt

GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
gis_csv = RAW / "GLB.Ts+dSST.csv"
if not gis_csv.exists():
    urllib.request.urlretrieve(GISTEMP_URL, gis_csv)
    with open(RAW / "DOWNLOAD_INFO.txt", "a") as f:
        f.write(f"\nurl: {GISTEMP_URL}\n"
                f"downloaded_utc: {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}\n"
                f"bytes: {gis_csv.stat().st_size}\n"
                f"sha256: {hashlib.sha256(gis_csv.read_bytes()).hexdigest()}\n")

gis = pd.read_csv(gis_csv, skiprows=1, na_values="***", index_col="Year")["J-D"].dropna()
gmst = gis.rolling(4).mean().rename("gmst_4yr")  # mean of years t-3 .. t

T = np.array(cfg.RETURN_PERIODS, float)
rng = np.random.default_rng(cfg.SEED + 5)
yearly = pd.read_csv(RESULTS / "01_yearly_coverage.csv", index_col="year")


def sample(min_cov):
    am = yearly.loc[yearly["coverage"] >= min_cov, "raw_max_mm"]
    df = pd.concat([am, gmst], axis=1, join="inner")
    return df["raw_max_mm"].to_numpy(), df["gmst_4yr"].to_numpy(), df.index.to_numpy()


def fit_pair(x, t):
    m0 = ev.gev_fit_mle(x)
    m1 = ev.gev_fit_ns(x, t, start=m0)
    lr = 2 * (m1["loglik"] - m0["loglik"])
    return m0, m1, lr, stats.chi2.sf(lr, 1)


def z_ns(Tk, m1, t):
    """T-year level (annual exceedance prob 1/T) in a climate with smoothed GMST' = t."""
    return ev.gev_return_level(Tk, m1["mu0"] + m1["mu1"] * t, m1["sigma"], m1["xi"])


x, t, years = sample(cfg.MIN_COVERAGE)
m0, m1, lr, p = fit_pair(x, t)
print(f"n={len(x)} years ({years.min()}-{years.max()}); smoothed GMST' from {t.min():.2f} to {t.max():.2f} C")
print(f"M0 stationary: mu={m0['mu']:.2f} sigma={m0['sigma']:.2f} xi={m0['xi']:.3f} loglik={m0['loglik']:.2f}")
print(f"M1 mu0={m1['mu0']:.2f} ({m1['se']['mu0']:.2f})  mu1={m1['mu1']:.2f} ({m1['se']['mu1']:.2f}) mm/K  "
      f"sigma={m1['sigma']:.2f} ({m1['se']['sigma']:.2f})  xi={m1['xi']:.3f} ({m1['se']['xi']:.3f})  "
      f"loglik={m1['loglik']:.2f}  converged={m1['converged']}")
print(f"LR test M1 vs M0: D={lr:.2f}, p={p:.4f};  AIC M0={-2*m0['loglik']+6:.1f}  M1={-2*m1['loglik']+8:.1f}")

# Return levels in an early (1900) and a recent (2025) climate, with bootstrap CIs from M1.
t_ref = {"1900": gmst.loc[1900], "2025": gmst.loc[cfg.LAST_YEAR]}
boot = np.empty((cfg.N_BOOT, 2 + 2 * len(T)))
for b in range(cfg.N_BOOT):
    xb = ev.gev_rvs(len(x), m1["mu0"] + m1["mu1"] * t, m1["sigma"], m1["xi"], rng)
    fb = ev.gev_fit_ns(xb, t, start={"mu": m1["mu0"], "sigma": m1["sigma"], "xi": m1["xi"]})
    boot[b, :2] = fb["mu1"], fb["xi"]
    boot[b, 2:2 + len(T)] = z_ns(T, fb, t_ref["1900"])
    boot[b, 2 + len(T):] = z_ns(T, fb, t_ref["2025"])
rows = []
for k, Tk in enumerate(T):
    z_a, z_b = z_ns(Tk, m1, t_ref["1900"]), z_ns(Tk, m1, t_ref["2025"])
    ba, bb = boot[:, 2 + k], boot[:, 2 + len(T) + k]
    rows.append({
        "T_years": Tk,
        "z_1900_mm": z_a, "z_1900_lo95": np.percentile(ba, 2.5), "z_1900_hi95": np.percentile(ba, 97.5),
        "z_2025_mm": z_b, "z_2025_lo95": np.percentile(bb, 2.5), "z_2025_hi95": np.percentile(bb, 97.5),
        "change_mm": z_b - z_a,
        "change_lo95": np.percentile(bb - ba, 2.5), "change_hi95": np.percentile(bb - ba, 97.5),
        "stationary_mm": ev.gev_return_level(Tk, m0["mu"], m0["sigma"], m0["xi"]),
    })
rl = pd.DataFrame(rows)
rl.to_csv(RESULTS / "05_return_levels_by_climate.csv", index=False)
mu1_lo, mu1_hi = np.percentile(boot[:, 0], [2.5, 97.5])
print(f"\nSmoothed GMST': 1900 = {t_ref['1900']:.2f} C, {cfg.LAST_YEAR} = {t_ref['2025']:.2f} C")
print(f"mu1 bootstrap 95% CI: [{mu1_lo:.1f}, {mu1_hi:.1f}] mm/K")
z100_0 = z_ns(100, m1, 0.0)
print(f"100-yr level change per K of GMST: {m1['mu1']:.1f} mm = {100 * m1['mu1'] / z100_0:.1f}% "
      f"of the 100-yr level at GMST' = 0 ({z100_0:.0f} mm)")
print(f"\nReturn levels (mm/day) in the 1900 vs {cfg.LAST_YEAR} climate, bootstrap B={cfg.N_BOOT}:")
print(rl.round(1).to_string(index=False))

# Sensitivity: strict coverage sample (drops most of the gappy 1890-1948 years).
xs, ts, _ = sample(cfg.SENS_COVERAGE)
_, m1s, lrs, ps = fit_pair(xs, ts)
print(f"\nSensitivity, coverage >= {cfg.SENS_COVERAGE} (n={len(xs)}): mu1={m1s['mu1']:.2f} "
      f"({m1s['se']['mu1']:.2f}) mm/K, LR D={lrs:.2f}, p={ps:.4f}")

# Influence: leave one year out, refit both models.
jk = []
for i in range(len(x)):
    keep = np.arange(len(x)) != i
    _, m1j, lrj, pj = fit_pair(x[keep], t[keep])
    jk.append((years[i], m1j["mu1"], pj))
jk = pd.DataFrame(jk, columns=["dropped_year", "mu1", "p"])
jk.to_csv(RESULTS / "05_jackknife.csv", index=False)
print(f"\nLeave-one-year-out: mu1 range [{jk['mu1'].min():.2f}, {jk['mu1'].max():.2f}] mm/K, "
      f"LR p range [{jk['p'].min():.4f}, {jk['p'].max():.4f}]")
print("Most influential years:")
print(jk.assign(d=(jk["mu1"] - m1["mu1"]).abs()).nlargest(4, "d")[["dropped_year", "mu1", "p"]]
      .round(4).to_string(index=False))

pd.DataFrame([
    {"sample": f"coverage>={cfg.MIN_COVERAGE}", "n": len(x), "mu0": m1["mu0"], "mu1": m1["mu1"],
     "mu1_se": m1["se"]["mu1"], "mu1_boot_lo95": mu1_lo, "mu1_boot_hi95": mu1_hi,
     "sigma": m1["sigma"], "xi": m1["xi"], "loglik_M1": m1["loglik"], "loglik_M0": m0["loglik"],
     "LR": lr, "p": p},
    {"sample": f"coverage>={cfg.SENS_COVERAGE}", "n": len(xs), "mu0": m1s["mu0"], "mu1": m1s["mu1"],
     "mu1_se": m1s["se"]["mu1"], "sigma": m1s["sigma"], "xi": m1s["xi"], "loglik_M1": m1s["loglik"],
     "LR": lrs, "p": ps},
]).to_csv(RESULTS / "05_ns_params.csv", index=False)

# ------------------------------------------------------------------------ figure
fig = plt.figure(figsize=(10, 7.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
a = fig.add_subplot(gs[0, :])
order = np.argsort(years)
a.plot(years, x, "o", ms=3.5, color=INK, label="annual max")
a.plot(years[order], (m1["mu0"] + m1["mu1"] * t)[order], color=C1, lw=1.8, label="μ(t), M1")
a.plot(years[order], z_ns(100, m1, t)[order], color=C2, lw=1.8, label="100-yr level, M1")
a.axhline(ev.gev_return_level(100, m0["mu"], m0["sigma"], m0["xi"]), color=C2, ls=":", lw=1.4,
          label="100-yr level, stationary")
a.set(ylabel="mm/day", title=f"Location tied to smoothed GMST (LR test p = {p:.3f})")
a.legend(loc="upper left", ncol=2, fontsize=8)

a = fig.add_subplot(gs[1, 0])
resid = np.sort(np.log1p(m1["xi"] * (x - m1["mu0"] - m1["mu1"] * t) / m1["sigma"]) / m1["xi"])
pp = np.arange(1, len(x) + 1) / (len(x) + 1)
gq = -np.log(-np.log(pp))
a.plot([gq.min(), gq.max()], [gq.min(), gq.max()], color=INK2, lw=1)
a.plot(gq, resid, "o", ms=3.5, color=C1)
a.set(xlabel="standard Gumbel quantile", ylabel="transformed residual", title="M1 residual quantile plot")

a = fig.add_subplot(gs[1, 1])
Tg = np.geomspace(1.5, 300, 120)
for key, color in (("1900", C1), (str(cfg.LAST_YEAR), C2)):
    a.plot(Tg, z_ns(Tg, m1, t_ref[key if key == "1900" else "2025"]), color=color,
           label=f"{key} climate (GMST' = {t_ref[key if key == '1900' else '2025']:.2f} C)")
a.set_xscale("log")
a.set(xlabel="return period (years)", ylabel="return level (mm/day)", title="Return levels by climate (M1)")
a.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "05_nonstationary.png")
print(f"\nSaved {FIGURES / '05_nonstationary.png'}")
