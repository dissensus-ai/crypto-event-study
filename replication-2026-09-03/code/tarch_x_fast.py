"""
Fast TARCH-X estimator -- numerically-identical drop-in for the bootstrap.

The canonical estimator (tarch_x_manual.TARCHXEstimator) is correct but slow:
its log-likelihood has a pure-Python per-observation loop and its standard
errors come from a 10x10 numerical Hessian (~400 LL evals). A single fit on the
2.4k-obs panel takes ~50s -- a 2000-draw x 6-asset bootstrap would be ~170 hours.

This module keeps the SAME model (GJR-GARCH, Student-t, demeaned residuals,
variance init = sample variance, exog linear in the variance equation, same
constraints/bounds, same SLSQP optimiser) but:
  * vectorises the Student-t log-likelihood (only the variance recursion is
    genuinely sequential);
  * @njit's the recursion if numba is importable, else uses a tight numpy loop;
  * skips the Hessian entirely (the bootstrap needs only the point delta's, not
    their model SEs).

Validated against the canonical estimator to ~3 decimals on the deltas (see
__main__ self-test and c7's smoke test).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

# ----------------------------------------------------------------------------
# Variance recursion: njit if available (numba lags Python; 3.14 has none yet),
# else a plain-python scalar loop. Both produce identical numbers.
# ----------------------------------------------------------------------------
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # noqa: BLE001
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # no-op decorator fallback
        def wrap(f):
            return f
        if args and callable(args[0]):
            return args[0]
        return wrap


@njit(cache=True, fastmath=False)
def _variance_recursion_core(omega, alpha, gamma, beta, resid, exog_contrib, var0):
    """
    GJR-GARCH variance recursion.

    sigma2_t = omega + alpha*eps2_{t-1} + gamma*eps2_{t-1}*1[eps_{t-1}<0]
               + beta*sigma2_{t-1} + (exog delta . x_t)

    `exog_contrib` is the pre-computed per-t sum_j delta_j x_{j,t} (vectorised
    outside the loop; the only sequential dependence is via sigma2_{t-1}).
    `resid` are the demeaned returns. `var0` initialises sigma2_0.
    """
    n = resid.shape[0]
    variance = np.empty(n)
    variance[0] = var0
    for t in range(1, n):
        e = resid[t - 1]
        eps_sq = e * e
        lev = gamma * eps_sq if e < 0.0 else 0.0
        v = omega + alpha * eps_sq + lev + beta * variance[t - 1] + exog_contrib[t]
        if v < 1e-8:
            v = 1e-8
        variance[t] = v
    return variance


class FastTARCHX:
    """
    Fast GJR-GARCH-X for repeated refits. Mirrors TARCHXEstimator's model.

    Parameters
    ----------
    returns : 1-D np.ndarray (already in %, NaNs dropped by caller)
    exog    : 2-D np.ndarray (n_obs x n_exog), aligned to returns
    """

    def __init__(self, returns, exog):
        self.returns = np.asarray(returns, dtype=float)
        self.exog = np.asarray(exog, dtype=float)
        if self.exog.ndim == 1:
            self.exog = self.exog[:, None]
        self.n_obs = self.returns.shape[0]
        self.n_exog = self.exog.shape[1]
        self.n_params = 5 + self.n_exog
        # cached, recomputed each fit only because returns can change per draw
        self.mean_return = self.returns.mean()
        self.resid = self.returns - self.mean_return
        self.var0 = np.var(self.returns)

    # -- pieces -------------------------------------------------------------
    def _variance(self, params):
        omega, alpha, gamma, beta = params[0], params[1], params[2], params[3]
        deltas = params[5:]
        exog_contrib = self.exog @ deltas if self.n_exog else np.zeros(self.n_obs)
        return _variance_recursion_core(
            omega, alpha, gamma, beta, self.resid, exog_contrib, self.var0
        )

    def _neg_loglik(self, params):
        nu = params[4]
        if nu <= 2.0:
            return 1e8
        variance = self._variance(params)
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
            return 1e8
        std2 = (self.resid * self.resid) / variance
        # Student-t log density (vectorised; same algebra as the canonical loop)
        const = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * (nu - 2.0))
        ll = (const
              - 0.5 * np.log(variance)
              - ((nu + 1.0) / 2.0) * np.log1p(std2 / (nu - 2.0)))
        s = ll.sum()
        if not np.isfinite(s):
            return 1e8
        return -s

    # -- optimiser scaffolding (identical to canonical) ---------------------
    def _bounds(self):
        b = [(1e-8, None), (1e-8, 0.3), (-0.5, 0.5), (1e-8, 0.95), (2.1, 50)]
        b += [(None, None)] * self.n_exog
        return b

    def _constraints(self):
        return [
            {'type': 'ineq', 'fun': lambda x: x[0] - 1e-8},
            {'type': 'ineq', 'fun': lambda x: x[1] - 1e-8},
            {'type': 'ineq', 'fun': lambda x: x[3] - 1e-8},
            {'type': 'ineq', 'fun': lambda x: x[4] - 2.1},
            {'type': 'ineq', 'fun': lambda x: 50 - x[4]},
            {'type': 'ineq', 'fun': lambda x: 0.999 - (x[1] + x[3] + abs(x[2]) / 2)},
        ]

    def _default_start(self):
        sv = np.var(self.returns)
        start = np.array([sv * 0.1, 0.05, 0.05, 0.85, 5.0])
        if self.n_exog:
            start = np.append(start, np.zeros(self.n_exog))
        return start

    def fit(self, start=None, max_iter=2000):
        """Single SLSQP fit. Returns (params, neg_loglik, success)."""
        if start is None:
            start = self._default_start()
        bounds = self._bounds()
        cons = self._constraints()
        res = minimize(self._neg_loglik, start, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': max_iter, 'disp': False})
        success = bool(res.success) and res.fun < 1e6
        return res.x, float(res.fun), success

    def fit_multistart(self, n_starts=5, max_iter=2000, seed=None, jitter=True):
        """
        Multistart fit: default start + (n_starts-1) randomised starts; keep best
        log-lik. Returns (best_params, best_negll, any_success).
        """
        rng = np.random.default_rng(seed)
        starts = [self._default_start()]
        sv = np.var(self.returns)
        for _ in range(max(0, n_starts - 1)):
            s = np.array([
                sv * rng.uniform(0.02, 0.30),     # omega
                rng.uniform(0.01, 0.15),          # alpha
                rng.uniform(-0.10, 0.20),         # gamma
                rng.uniform(0.70, 0.93),          # beta
                rng.uniform(3.0, 12.0),           # nu
            ])
            if self.n_exog:
                s = np.append(s, rng.normal(0.0, 0.5, self.n_exog))
            # enforce stationarity feasibility of the seed
            if s[1] + s[3] + abs(s[2]) / 2 >= 0.999:
                s[3] = 0.90 - s[1] - abs(s[2]) / 2
            starts.append(s)

        best = None          # best over ALL starts (fallback only)
        best_ok = None       # best over CONVERGED starts (preferred)
        for s in starts:
            try:
                p, f, ok = self.fit(start=s, max_iter=max_iter)
            except Exception:  # noqa: BLE001
                continue
            if best is None or f < best[1]:
                best = (p, f, ok)
            if ok and (best_ok is None or f < best_ok[1]):
                best_ok = (p, f, ok)
        # Return the lowest-objective CONVERGED fit. Previously this returned the
        # lowest-objective fit overall while reporting `any_ok`, which could hand
        # back a non-converged optimum flagged as a success.
        if best_ok is not None:
            return best_ok[0], best_ok[1], True
        if best is None:
            return self._default_start(), 1e8, False
        return best[0], best[1], False

    def deltas(self, params):
        """Return the n_exog delta coefficients (variance-equation exog coefs)."""
        return params[5:]


if __name__ == "__main__":
    # Self-test vs the canonical estimator on btc baseline.
    import warnings; warnings.simplefilter("ignore")
    import time
    import pandas as pd
    import c2_relaxed_threshold_sensitivity as c2

    panel = c2.load_returns_panel()
    common = pd.DatetimeIndex(sorted(set.intersection(*[set(s.index) for s in panel.values()])))
    sent = c2.load_sentiment_daily(common)
    events = pd.read_csv(c2.DATA_DIR / "events.csv"); events["date"] = pd.to_datetime(events["date"])
    census = pd.read_csv(c2.OUT_DIR / "c1-dropout-census.csv"); census["date"] = pd.to_datetime(census["date"])
    inf_d, reg_d = c2.get_event_dates_for_spec("S1_baseline", events, census)

    a = "btc"
    r = panel[a].loc[panel[a].index >= pd.Timestamp(c2.START_DATE)]
    dum = c2.build_event_dummies(r.index, inf_d, reg_d, c2.WINDOW_DAYS_BEFORE, c2.WINDOW_DAYS_AFTER)
    s = sent.reindex(r.index).fillna(0)
    exog = pd.concat([dum[["D_infrastructure", "D_regulatory"]],
                      s[["S_gdelt_normalized", "S_reg_decomposed", "S_infra_decomposed"]]], axis=1).fillna(0)

    print(f"numba available: {_HAVE_NUMBA}")
    fast = FastTARCHX(r.values, exog.values)
    t0 = time.time()
    p, f, ok = fast.fit_multistart(n_starts=5, seed=0)
    t1 = time.time()
    names = ["omega", "alpha", "gamma", "beta", "nu", "D_infra", "D_reg", "S_gdelt", "S_reg", "S_infra"]
    print(f"fast fit time {t1 - t0:.2f}s  success={ok}  negLL={f:.3f}")
    for nm, v in zip(names, p):
        print(f"  {nm:10s} {v: .5f}")
    print(f"\n  canonical (from smoke test): dInfra 1.0076  dReg 0.2983  LL -5919.70")
    print(f"  fast:                        dInfra {p[5]:.4f}  dReg {p[6]:.4f}  LL {-f:.2f}")
