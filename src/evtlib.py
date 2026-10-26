"""Extreme-value tools written out explicitly (no EVT package), following Coles (2001).

Shape convention: xi > 0 is a heavy (Frechet-type) upper tail, xi = 0 Gumbel,
xi < 0 a bounded tail. scipy.stats.genextreme uses c = -xi, so any comparison
with scipy must flip the sign.
"""

import numpy as np
from scipy import optimize, special, stats

XI_EPS = 1e-6  # |xi| below this is treated as the Gumbel / exponential limit


def _expm1_ratio(xi, y):
    """(1 - y**(-xi)) / xi, with the xi -> 0 limit log(y)."""
    if abs(xi) < XI_EPS:
        return np.log(y)
    return -np.expm1(-xi * np.log(y)) / xi


# ----------------------------------------------------------------------------- GEV

def gev_loglik(x, mu, sigma, xi):
    """GEV log-likelihood; mu may be an array (non-stationary location)."""
    if sigma <= 0:
        return -np.inf
    z = (np.asarray(x) - mu) / sigma
    if abs(xi) < XI_EPS:
        return float(np.sum(-np.log(sigma) - z - np.exp(-z)))
    t = 1.0 + xi * z
    if np.any(t <= 0):
        return -np.inf  # an observation lies outside the support
    return float(np.sum(-np.log(sigma) - (1 + 1 / xi) * np.log(t) - t ** (-1 / xi)))


def gev_cdf(x, mu, sigma, xi):
    z = (np.asarray(x, float) - mu) / sigma
    if abs(xi) < XI_EPS:
        return np.exp(-np.exp(-z))
    t = np.maximum(1.0 + xi * z, 0.0)
    with np.errstate(divide="ignore"):
        return np.exp(-t ** (-1 / xi))


def gev_ppf(p, mu, sigma, xi):
    """Quantile function; the T-year return level is gev_ppf(1 - 1/T)."""
    y = -np.log(np.asarray(p, float))
    if abs(xi) < XI_EPS:
        return mu - sigma * np.log(y)
    return mu + sigma / xi * (y ** (-xi) - 1.0)


def gev_return_level(T, mu, sigma, xi):
    return gev_ppf(1.0 - 1.0 / np.asarray(T, float), mu, sigma, xi)


def gev_rvs(n, mu, sigma, xi, rng):
    return gev_ppf(rng.uniform(size=n), mu, sigma, xi)


def gev_fit_lmom(x):
    """L-moment (= probability-weighted-moment) estimator, Hosking (1990)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    i = np.arange(n)
    b0 = x.mean()
    b1 = np.sum(i / (n - 1) * x) / n
    b2 = np.sum(i * (i - 1) / ((n - 1) * (n - 2)) * x) / n
    l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    t3 = l3 / l2  # L-skewness
    c = 2 / (3 + t3) - np.log(2) / np.log(3)
    k = 7.8590 * c + 2.9554 * c**2  # Hosking's k = -xi
    g = special.gamma(1 + k)
    sigma = l2 * k / ((1 - 2.0 ** (-k)) * g)
    mu = l1 - sigma * (1 - g) / k
    return {"mu": mu, "sigma": sigma, "xi": -k, "t3": t3}


def gev_fit_mle(x, start=None):
    """MLE over (mu, log sigma, xi), started from the L-moment estimate."""
    x = np.asarray(x, float)
    if start is None:
        s = gev_fit_lmom(x)
        start = (s["mu"], s["sigma"], s["xi"])
    th0 = np.array([start[0], np.log(start[1]), start[2]])

    def nll(th):
        ll = gev_loglik(x, th[0], np.exp(th[1]), th[2])
        return -ll if np.isfinite(ll) else 1e10

    res = optimize.minimize(nll, th0, method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000})
    mu, sigma, xi = res.x[0], np.exp(res.x[1]), res.x[2]
    return {"mu": mu, "sigma": sigma, "xi": xi, "loglik": -res.fun, "converged": res.success}


def gev_fit_gumbel(x):
    """MLE with xi fixed at 0 (the null model for testing xi = 0)."""
    x = np.asarray(x, float)
    m, s = np.mean(x), np.std(x)
    th0 = np.array([m - 0.5772 * s * np.sqrt(6) / np.pi, np.log(s * np.sqrt(6) / np.pi)])
    res = optimize.minimize(lambda th: -gev_loglik(x, th[0], np.exp(th[1]), 0.0), th0,
                            method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10})
    return {"mu": res.x[0], "sigma": np.exp(res.x[1]), "xi": 0.0, "loglik": -res.fun}


def numerical_hessian(f, th, rel_step=1e-4):
    th = np.asarray(th, float)
    k = len(th)
    h = rel_step * np.maximum(np.abs(th), 1.0)
    H = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            def fij(di, dj):
                t = th.copy()
                t[i] += di
                t[j] += dj
                return f(t)
            H[i, j] = (fij(h[i], h[j]) - fij(h[i], -h[j]) - fij(-h[i], h[j])
                       + fij(-h[i], -h[j])) / (4 * h[i] * h[j])
    return H


def gev_mle_cov(x, fit):
    """Inverse observed information for (mu, sigma, xi)."""
    th = [fit["mu"], fit["sigma"], fit["xi"]]
    H = numerical_hessian(lambda t: -gev_loglik(x, t[0], t[1], t[2]), th)
    return np.linalg.inv(H)


def gev_return_level_delta(T, x, fit):
    """Return level and its delta-method standard error, se = sqrt(g' V g) (Coles 2001, eq. 3.10).

    The gradient g of z_T with respect to (mu, sigma, xi) is taken numerically.
    """
    th = np.array([fit["mu"], fit["sigma"], fit["xi"]])
    z = gev_return_level(T, *th)
    grad = np.empty(3)
    for i in range(3):
        h = 1e-5 * max(abs(th[i]), 1.0)
        tp, tm = th.copy(), th.copy()
        tp[i] += h
        tm[i] -= h
        grad[i] = (gev_return_level(T, *tp) - gev_return_level(T, *tm)) / (2 * h)
    V = gev_mle_cov(x, fit)
    return z, float(np.sqrt(grad @ V @ grad))


def _profile_ci(dev, center, lo_guess, hi_guess, level):
    """Values where the profile deviance crosses the chi2(1) critical value."""
    crit = stats.chi2.ppf(level, 1)
    f = lambda v: dev(v) - crit  # noqa: E731
    bounds = []
    for guess in (lo_guess, hi_guess):
        step, edge = guess - center, guess
        for _ in range(60):  # widen the bracket until the deviance exceeds crit
            if f(edge) > 0:
                break
            edge += step
            step *= 1.5
        else:
            bounds.append(np.nan)
            continue
        bounds.append(optimize.brentq(f, *sorted((center, edge)), xtol=1e-6))
    return tuple(bounds)


def gev_profile_deviance_rl(x, fit, T):
    """Profile deviance 2 * (l_max - l_p(z_T)) for the T-year return level (Coles 2001, sec. 3.3.3).

    Reparameterise mu = z_T + sigma * (1 - y**(-xi)) / xi with y = -log(1 - 1/T),
    then for each fixed z_T maximise the likelihood over (sigma, xi).
    """
    x = np.asarray(x, float)
    y = -np.log(1 - 1 / T)
    ls = np.log(fit["sigma"])
    starts = [(ls, fit["xi"]), (ls, 0.0), (ls + 0.5, fit["xi"]), (ls - 0.5, fit["xi"] + 0.2)]

    def prof_loglik(zt):
        def nll(th):
            sigma, xi = np.exp(th[0]), th[1]
            ll = gev_loglik(x, zt + sigma * _expm1_ratio(xi, y), sigma, xi)
            return -ll if np.isfinite(ll) else 1e10
        # Several starts: far from z_hat the MLE start can sit outside the support.
        best = min((optimize.minimize(nll, th0, method="Nelder-Mead",
                                      options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 5000})
                    for th0 in starts), key=lambda r: r.fun)
        return -best.fun

    return lambda zt: 2 * (fit["loglik"] - prof_loglik(zt))


def gev_profile_return_level(x, fit, T, level=0.95):
    """Profile-likelihood CI for the T-year return level."""
    z_hat = gev_return_level(T, fit["mu"], fit["sigma"], fit["xi"])
    se = gev_return_level_delta(T, x, fit)[1]
    dev = gev_profile_deviance_rl(x, fit, T)
    lo, hi = _profile_ci(dev, z_hat, z_hat - se, z_hat + se, level)
    return z_hat, lo, hi


def gev_profile_deviance_xi(x, fit):
    """Profile deviance for the shape xi, maximising over (mu, sigma)."""
    x = np.asarray(x, float)

    def prof_loglik(xi):
        def nll(th):
            ll = gev_loglik(x, th[0], np.exp(th[1]), xi)
            return -ll if np.isfinite(ll) else 1e10
        r = optimize.minimize(nll, [fit["mu"], np.log(fit["sigma"])], method="Nelder-Mead",
                              options={"xatol": 1e-8, "fatol": 1e-10})
        return -r.fun

    return lambda xi: 2 * (fit["loglik"] - prof_loglik(xi))


def gev_profile_xi(x, fit, level=0.95):
    """Profile-likelihood CI for the shape xi."""
    dev = gev_profile_deviance_xi(x, fit)
    return _profile_ci(dev, fit["xi"], fit["xi"] - 0.05, fit["xi"] + 0.05, level)


def gev_parametric_bootstrap(fit, n, B, T, rng, fitter=gev_fit_mle):
    """Refit B samples simulated from the fitted GEV; returns params and return levels."""
    out = np.empty((B, 3 + len(T)))
    for b in range(B):
        xb = gev_rvs(n, fit["mu"], fit["sigma"], fit["xi"], rng)
        fb = fitter(xb)
        out[b, :3] = fb["mu"], fb["sigma"], fb["xi"]
        out[b, 3:] = gev_return_level(T, fb["mu"], fb["sigma"], fb["xi"])
    return out


# ----------------------------------------------------------------------- goodness of fit

def ad_ks(u):
    """Anderson-Darling A^2 and Kolmogorov-Smirnov D from fitted-CDF values u = F(x)."""
    u = np.clip(np.sort(np.asarray(u, float)), 1e-12, 1 - 1e-12)
    n = len(u)
    i = np.arange(1, n + 1)
    a2 = -n - np.mean((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1])))
    d = max(np.max(i / n - u), np.max(u - (i - 1) / n))
    return a2, d


def gof_parametric_bootstrap(x, cdf, rvs, fitter, B, rng):
    """AD and KS p-values when parameters are estimated from the same data.

    Tabulated AD/KS critical values assume known parameters and are too lenient
    here, so each bootstrap sample is refitted before computing its statistic.
    cdf(x, fit) -> F(x); rvs(n, fit, rng) -> sample; fitter(x) -> fit.
    """
    fit = fitter(x)
    a2_obs, d_obs = ad_ks(cdf(x, fit))
    a2_b, d_b = np.empty(B), np.empty(B)
    for b in range(B):
        xb = rvs(len(x), fit, rng)
        a2_b[b], d_b[b] = ad_ks(cdf(xb, fitter(xb)))
    return {
        "A2": a2_obs, "p_A2": (1 + np.sum(a2_b >= a2_obs)) / (B + 1),
        "D": d_obs, "p_D": (1 + np.sum(d_b >= d_obs)) / (B + 1),
        "D_naive_p": stats.kstwo.sf(d_obs, len(x)),  # known-parameter KS p-value, for contrast
    }


# ----------------------------------------------------------------------------- GPD

def gpd_loglik(y, sigma, xi):
    """GPD log-likelihood for exceedances y = x - u > 0."""
    if sigma <= 0:
        return -np.inf
    y = np.asarray(y, float)
    if abs(xi) < XI_EPS:
        return float(-len(y) * np.log(sigma) - np.sum(y) / sigma)
    t = 1.0 + xi * y / sigma
    if np.any(t <= 0):
        return -np.inf
    return float(-len(y) * np.log(sigma) - (1 + 1 / xi) * np.sum(np.log(t)))


def gpd_cdf(y, sigma, xi):
    y = np.asarray(y, float)
    if abs(xi) < XI_EPS:
        return 1 - np.exp(-y / sigma)
    return 1 - np.maximum(1 + xi * y / sigma, 0.0) ** (-1 / xi)


def gpd_ppf(p, sigma, xi):
    p = np.asarray(p, float)
    if abs(xi) < XI_EPS:
        return -sigma * np.log1p(-p)
    return sigma / xi * ((1 - p) ** (-xi) - 1)


def gpd_rvs(n, sigma, xi, rng):
    return gpd_ppf(rng.uniform(size=n), sigma, xi)


def gpd_fit_mle(y):
    y = np.asarray(y, float)
    m, v = y.mean(), y.var()
    xi0 = 0.5 * (1 - m**2 / v)  # method-of-moments start
    th0 = np.array([np.log(max(0.5 * m * (m**2 / v + 1), 1e-3)), xi0])

    def nll(th):
        ll = gpd_loglik(y, np.exp(th[0]), th[1])
        return -ll if np.isfinite(ll) else 1e10

    res = optimize.minimize(nll, th0, method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000})
    return {"sigma": np.exp(res.x[0]), "xi": res.x[1], "loglik": -res.fun,
            "converged": res.success}


def gpd_mle_cov(y, fit):
    H = numerical_hessian(lambda t: -gpd_loglik(y, t[0], t[1]), [fit["sigma"], fit["xi"]])
    return np.linalg.inv(H)


def pot_return_level(T, u, sigma, xi, rate):
    """Level with annual exceedance probability 1/T, for a Poisson(rate per year)
    process of independent exceedances of u with GPD(sigma, xi) excesses."""
    r = rate / -np.log1p(-1 / np.asarray(T, float))  # expected exceedances per "event-year"
    if abs(xi) < XI_EPS:
        return u + sigma * np.log(r)
    return u + sigma / xi * (r**xi - 1)


def pot_profile_return_level(y, fit, u, rate, T, level=0.95):
    """Profile CI for the POT return level, treating the exceedance rate as known.

    For fixed x_T, sigma = xi * (x_T - u) / (r**xi - 1); maximise over xi only.
    """
    y = np.asarray(y, float)
    r = rate / -np.log1p(-1 / T)
    x_hat = pot_return_level(T, u, fit["sigma"], fit["xi"], rate)

    def prof_loglik(xt):
        def nll(xi):
            s = (xt - u) / np.log(r) if abs(xi) < XI_EPS else xi * (xt - u) / np.expm1(xi * np.log(r))
            ll = gpd_loglik(y, s, xi)
            return -ll if np.isfinite(ll) else 1e10
        res = optimize.minimize_scalar(nll, bounds=(-1.0, 1.5), method="bounded",
                                       options={"xatol": 1e-8})
        return -res.fun

    dev = lambda xt: 2 * (fit["loglik"] - prof_loglik(xt))  # noqa: E731
    lo, hi = _profile_ci(dev, x_hat, 0.9 * x_hat, 1.1 * x_hat, level)
    return x_hat, lo, hi


def mean_residual_life(x, thresholds, level=0.95):
    """Mean excess over u with a normal-approximation CI, for each candidate u."""
    x = np.asarray(x, float)
    zc = stats.norm.ppf(0.5 + level / 2)
    rows = []
    for u in thresholds:
        e = x[x > u] - u
        if len(e) < 2:
            break
        rows.append((u, e.mean(), zc * e.std(ddof=1) / np.sqrt(len(e)), len(e)))
    return np.array(rows)


def decluster_runs(values, u, run):
    """Runs declustering: a cluster ends after `run` consecutive non-exceedances.

    `values` is a daily series (NaN = missing, which breaks a run of exceedances
    the same way as a non-exceedance). Returns the index positions of cluster peaks.
    """
    v = np.asarray(values, float)
    exc = np.flatnonzero(v > u)
    if len(exc) == 0:
        return exc
    peaks, start = [], 0
    for k in range(1, len(exc) + 1):
        if k == len(exc) or exc[k] - exc[k - 1] > run:
            block = exc[start:k]
            peaks.append(block[np.argmax(v[block])])
            start = k
    return np.array(peaks)


# ---------------------------------------------------------------- non-stationary GEV

def gev_fit_ns(x, t, start):
    """GEV with mu(t) = mu0 + mu1 * t; sigma and xi constant."""
    x, t = np.asarray(x, float), np.asarray(t, float)

    def nll(th):
        ll = gev_loglik(x, th[0] + th[1] * t, np.exp(th[2]), th[3])
        return -ll if np.isfinite(ll) else 1e10

    th0 = np.array([start["mu"], 0.0, np.log(start["sigma"]), start["xi"]])
    res = optimize.minimize(nll, th0, method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 40000})
    th = res.x
    H = numerical_hessian(lambda p: -gev_loglik(x, p[0] + p[1] * t, p[2], p[3]),
                          [th[0], th[1], np.exp(th[2]), th[3]])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return {"mu0": th[0], "mu1": th[1], "sigma": np.exp(th[2]), "xi": th[3],
            "se": dict(zip(["mu0", "mu1", "sigma", "xi"], se)),
            "loglik": -res.fun, "converged": res.success}
