"""Analysis choices for the Blue Hill extreme-precipitation study.

Each value is a decision made during the analysis; the comment records why.
"""

STATION_ID = "USC00190736"  # Blue Hill COOP, Milton MA; daily record from 1885-02-01
STATION_NAME = "Blue Hill, MA"
DATA_URL = (
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/"
    f"access/{STATION_ID}.csv"
)

LAST_YEAR = 2025  # 2026 is still in progress, so its annual maximum is not final

# Minimum fraction of valid days for a year to enter the block-maxima sample.
# Missing days can only lower an annual maximum, so incomplete years bias return
# levels low; dropping years costs sample size. Main analysis uses the common 90%
# rule (139 of 140 years); the 99% rule (123 years) is a sensitivity check.
MIN_COVERAGE = 0.90
SENS_COVERAGE = 0.99

RETURN_PERIODS = [2, 10, 25, 50, 100]  # years; 100 is within the ~140-yr record length
# Peaks over threshold (chosen from figures/04a_threshold_choice.png): 30 mm is the
# lowest threshold above which xi and sigma* are stable and the mean residual life is
# roughly linear; 40 and 50 mm are reported as threshold sensitivity.
POT_THRESHOLD = 30.0
POT_SENS_THRESHOLDS = [40.0, 50.0]
# Runs declustering: exceedances separated by fewer than 2 non-exceedance days are one
# event. Cluster counts differ by about 3% between 1 and 3 days at u = 30 mm (1200 vs 1160).
RUN_LENGTH = 2

N_BOOT = 2000  # parametric-bootstrap replicates for CIs and GoF p-values
SEED = 20261023
