"""Paths and plot style shared by the step scripts."""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
for d in (RAW, PROCESSED, FIGURES, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# Series colors in a fixed order, plus text and grid colors.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "lines.linewidth": 2,
    "legend.frameon": False,
})


def daily_series(min_coverage):
    """QC'd daily precipitation on a full calendar; years below min_coverage set to NaN.

    Returns the series and the effective number of years of valid data.
    """
    yearly = pd.read_csv(RESULTS / "01_yearly_coverage.csv", index_col="year")
    good = yearly.index[yearly["coverage"] >= min_coverage]
    daily = pd.read_csv(PROCESSED / "prcp_daily.csv", index_col="date", parse_dates=True)["prcp_mm"]
    s = daily.reindex(pd.date_range(f"{good.min()}-01-01", f"{good.max()}-12-31", freq="D"))
    s[~s.index.year.isin(good)] = np.nan
    return s, s.notna().sum() / 365.25


def to_obs_precision(x_mm):
    """Round to how GHCN-Daily stores this station's PRCP: 0.01 inch, kept as integer tenths of mm."""
    return np.round(np.round(np.asarray(x_mm) / 0.254) * 2.54) / 10
