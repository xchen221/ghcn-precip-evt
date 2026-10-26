"""Step 1: download GHCN-Daily, quality-control PRCP, and tabulate yearly coverage.

Outputs
  data/raw/<station>.csv, data/raw/DOWNLOAD_INFO.txt  cached download + provenance
  data/processed/prcp_daily.csv                      QC'd daily precipitation (mm)
  results/01_yearly_coverage.csv                     valid-day fraction and raw max per year
  figures/01_annual_max_coverage.png
"""

import hashlib
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as cfg
from common import C1, C2, FIGURES, INK2, PROCESSED, RAW, RESULTS, plt

raw_csv = RAW / f"{cfg.STATION_ID}.csv"
if not raw_csv.exists():
    print(f"Downloading {cfg.DATA_URL}")
    urllib.request.urlretrieve(cfg.DATA_URL, raw_csv)
    sha = hashlib.sha256(raw_csv.read_bytes()).hexdigest()
    (RAW / "DOWNLOAD_INFO.txt").write_text(
        f"url: {cfg.DATA_URL}\n"
        f"downloaded_utc: {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}\n"
        f"bytes: {raw_csv.stat().st_size}\n"
        f"sha256: {sha}\n"
    )
print((RAW / "DOWNLOAD_INFO.txt").read_text())

df = pd.read_csv(raw_csv, usecols=["DATE", "PRCP", "PRCP_ATTRIBUTES"],
                 dtype={"PRCP_ATTRIBUTES": str}, parse_dates=["DATE"])
df = df.dropna(subset=["PRCP"])

# PRCP_ATTRIBUTES = "MFLAG,QFLAG,SFLAG,OBSTIME"; PRCP is in tenths of mm.
flags = df["PRCP_ATTRIBUTES"].fillna(",,,").str.split(",", expand=True)
df["mflag"], df["qflag"], df["sflag"] = flags[0], flags[1], flags[2]
df["prcp_mm"] = df["PRCP"] / 10.0

print("Quality flags on PRCP (blank = passed all NCEI checks):")
print(df["qflag"].replace("", "(blank)").value_counts().to_string(), "\n")
failed = df[df["qflag"] != ""]
if len(failed):
    print("Values removed by QC:")
    print(failed[["DATE", "prcp_mm", "qflag"]].to_string(index=False), "\n")

daily = df.loc[df["qflag"] == "", ["DATE", "prcp_mm", "mflag", "sflag"]]
daily = daily.rename(columns={"DATE": "date"}).set_index("date")
daily.to_csv(PROCESSED / "prcp_daily.csv")

# Coverage on the full calendar, so days absent from the file count as missing.
first_year = daily.index.year.min()
cal = pd.date_range(f"{first_year}-01-01", f"{cfg.LAST_YEAR}-12-31", freq="D")
s = daily["prcp_mm"].reindex(cal)
g = s.groupby(s.index.year)
summer_fall = s[(s.index.month >= 6) & (s.index.month <= 10)]
yearly = pd.DataFrame({
    "n_valid": g.count(),
    "n_days": g.size(),
    "raw_max_mm": g.max(),
    "date_of_max": g.idxmax(),
    "missing_jun_oct": summer_fall.isna().groupby(summer_fall.index.year).sum(),
})
yearly["coverage"] = yearly["n_valid"] / yearly["n_days"]
yearly.index.name = "year"
yearly.to_csv(RESULTS / "01_yearly_coverage.csv")

print(f"Years {first_year}-{cfg.LAST_YEAR}: {len(yearly)}")
for thr in (1.0, 0.99, 0.95, 0.90, 0.80):
    print(f"  coverage >= {thr:.2f}: {(yearly['coverage'] >= thr).sum()} years")
print("\nYears with coverage < 0.99:")
cols = ["n_valid", "n_days", "coverage", "missing_jun_oct", "raw_max_mm"]
print(yearly.loc[yearly["coverage"] < 0.99, cols].round(3).to_string())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
full = yearly["coverage"] >= 0.99
ax1.plot(yearly.index[full], yearly["raw_max_mm"][full], "o", ms=4, color=C1,
         label="coverage ≥ 99%")
ax1.plot(yearly.index[~full], yearly["raw_max_mm"][~full], "o", ms=5, color=C2,
         label="coverage < 99%")
ax1.set_ylabel("Annual max daily precip (mm)")
ax1.set_title(f"{cfg.STATION_NAME} ({cfg.STATION_ID}): annual maximum 1-day precipitation")
ax1.legend(loc="upper left")
ax2.bar(yearly.index, yearly["coverage"] * 100, width=0.8, color=INK2)
ax2.set_ylim(np.floor(yearly["coverage"].min() * 10) * 10, 100.5)
ax2.set_ylabel("Valid days (%)")
ax2.set_xlabel("Year")
fig.savefig(FIGURES / "01_annual_max_coverage.png")
print(f"\nSaved {FIGURES / '01_annual_max_coverage.png'}")
