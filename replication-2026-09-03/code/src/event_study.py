"""
Event Study Module
==================

Implements proper event study methodology for cryptocurrency liquidity analysis.

Key features:
- Constant mean model for expected returns (preferred for crypto per lit)
- Kolari-Pynnonen adjusted t-test for cross-sectional correlation
- Bootstrap BCA confidence intervals
- Difference-in-Differences estimator for infrastructure vs regulatory

References:
- MacKinlay (1997) Event Studies in Economics and Finance
- Kolari & Pynnonen (2010) Event Study Testing with Cross-sectional Correlation
- Brown & Warner (1985) Using Daily Stock Returns
- Corbet et al. (2019) Cryptocurrency Event Studies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import warnings
import json
from pathlib import Path

from . import config


@dataclass
class EventStudyResult:
    """Container for event study results."""
    event_id: int
    event_date: str
    event_type: str
    event_name: str
    symbol: str

    # Estimation period stats
    estimation_mean: float
    estimation_std: float
    estimation_n: int

    # Event window stats
    car: float           # Cumulative Abnormal Return
    car_t_stat: float    # t-statistic
    car_p_value: float

    # Bootstrap results
    car_bootstrap_ci_low: float
    car_bootstrap_ci_high: float
    car_bootstrap_pvalue: float

    # Additional metrics
    aar: np.ndarray     # Average Abnormal Returns by day
    event_window: Tuple[int, int]


class ConstantMeanModel:
    """
    Constant Mean Return Model for expected returns.

    Preferred for crypto markets per Corbet et al. (2019) due to:
    - No stable "market index" for crypto
    - High correlation between assets reduces market model benefit
    - Simpler assumptions more robust to structural breaks
    """

    def __init__(
        self,
        estimation_window: int = config.ESTIMATION_WINDOW_DAYS,
        gap_window: int = config.GAP_WINDOW_DAYS
    ):
        self.estimation_window = estimation_window
        self.gap_window = gap_window

    def compute_abnormal_returns(
        self,
        returns: pd.Series,
        event_date: str,
        event_window: Tuple[int, int] = (-5, 30)
    ) -> Dict:
        """
        Compute abnormal returns using constant mean model.

        Args:
            returns: Daily return series with datetime index
            event_date: Event date (YYYY-MM-DD)
            event_window: (pre_days, post_days) relative to event

        Returns:
            Dictionary with estimation stats and abnormal returns
        """
        event_dt = pd.to_datetime(event_date)
        returns = returns.copy()
        returns.index = pd.to_datetime(returns.index)

        # Define periods
        estimation_end = event_dt - pd.Timedelta(days=self.gap_window + 1)
        estimation_start = estimation_end - pd.Timedelta(days=self.estimation_window)

        event_start = event_dt + pd.Timedelta(days=event_window[0])
        event_end = event_dt + pd.Timedelta(days=event_window[1])

        # Get estimation period returns
        estimation_returns = returns[
            (returns.index >= estimation_start) &
            (returns.index <= estimation_end)
        ].dropna()

        if len(estimation_returns) < config.ESTIMATION_WINDOW_MIN:
            return {
                'error': f'Insufficient estimation data: {len(estimation_returns)} days',
                'estimation_mean': np.nan,
                'estimation_std': np.nan,
                'estimation_n': len(estimation_returns)
            }

        # Compute expected return (constant mean)
        expected_return = estimation_returns.mean()
        estimation_std = estimation_returns.std()

        # Get event window returns
        event_returns = returns[
            (returns.index >= event_start) &
            (returns.index <= event_end)
        ].dropna()

        if len(event_returns) < 5:
            return {
                'error': f'Insufficient event window data: {len(event_returns)} days',
                'estimation_mean': expected_return,
                'estimation_std': estimation_std,
                'estimation_n': len(estimation_returns)
            }

        # Compute abnormal returns
        abnormal_returns = event_returns - expected_return

        # Compute CAR and t-stat
        car = abnormal_returns.sum()
        sar = car / (estimation_std * np.sqrt(len(abnormal_returns)))

        return {
            'estimation_mean': expected_return,
            'estimation_std': estimation_std,
            'estimation_n': len(estimation_returns),
            'abnormal_returns': abnormal_returns,
            'car': car,
            'sar': sar,  # Standardized abnormal return
            'event_window_n': len(abnormal_returns)
        }


class MarketModel:
    """
    Market-Adjusted Return Model for expected returns.

    Uses BTC as market proxy for altcoins, or equal-weighted crypto index.
    Addresses bull market drift that may cause positive placebo CARs.

    For BTC itself, falls back to constant mean model.
    """

    def __init__(
        self,
        estimation_window: int = config.ESTIMATION_WINDOW_DAYS,
        gap_window: int = config.GAP_WINDOW_DAYS,
        market_proxy: str = 'BTC'
    ):
        self.estimation_window = estimation_window
        self.gap_window = gap_window
        self.market_proxy = market_proxy
        self._market_returns: Optional[pd.Series] = None

    def set_market_returns(self, market_returns: pd.Series):
        """Set the market return series (typically BTC returns)."""
        self._market_returns = market_returns.copy()
        self._market_returns.index = pd.to_datetime(self._market_returns.index)

    def compute_abnormal_returns(
        self,
        returns: pd.Series,
        event_date: str,
        event_window: Tuple[int, int] = (-5, 30),
        symbol: str = None
    ) -> Dict:
        """
        Compute abnormal returns using market model.

        For the market proxy itself (BTC), uses constant mean model.
        For other assets, uses: AR_i,t = R_i,t - beta_i * R_m,t

        Args:
            returns: Daily return series with datetime index
            event_date: Event date (YYYY-MM-DD)
            event_window: (pre_days, post_days) relative to event
            symbol: Asset symbol (if BTC, uses constant mean)

        Returns:
            Dictionary with estimation stats and abnormal returns
        """
        event_dt = pd.to_datetime(event_date)
        returns = returns.copy()
        returns.index = pd.to_datetime(returns.index)

        # For market proxy itself, fall back to constant mean
        if symbol == self.market_proxy or self._market_returns is None:
            cm = ConstantMeanModel(self.estimation_window, self.gap_window)
            result = cm.compute_abnormal_returns(returns, event_date, event_window)
            result['model'] = 'constant_mean'
            return result

        # Define periods
        estimation_end = event_dt - pd.Timedelta(days=self.gap_window + 1)
        estimation_start = estimation_end - pd.Timedelta(days=self.estimation_window)

        event_start = event_dt + pd.Timedelta(days=event_window[0])
        event_end = event_dt + pd.Timedelta(days=event_window[1])

        # Align asset and market returns
        combined = pd.DataFrame({
            'asset': returns,
            'market': self._market_returns
        }).dropna()

        # Get estimation period
        estimation_data = combined[
            (combined.index >= estimation_start) &
            (combined.index <= estimation_end)
        ]

        if len(estimation_data) < config.ESTIMATION_WINDOW_MIN:
            return {
                'error': f'Insufficient estimation data: {len(estimation_data)} days',
                'estimation_mean': np.nan,
                'estimation_std': np.nan,
                'estimation_n': len(estimation_data),
                'model': 'market_model'
            }

        # Estimate market model: R_i = alpha + beta * R_m + epsilon
        X = estimation_data['market'].values
        Y = estimation_data['asset'].values

        # OLS regression
        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            beta_hat = np.linalg.lstsq(X_with_const, Y, rcond=None)[0]
            alpha, beta = beta_hat[0], beta_hat[1]
        except:
            # Fallback to constant mean if regression fails
            cm = ConstantMeanModel(self.estimation_window, self.gap_window)
            return cm.compute_abnormal_returns(returns, event_date, event_window)

        # Estimation period residuals for standard deviation
        predicted = alpha + beta * estimation_data['market']
        residuals = estimation_data['asset'] - predicted
        estimation_std = residuals.std()

        # Get event window data
        event_data = combined[
            (combined.index >= event_start) &
            (combined.index <= event_end)
        ]

        if len(event_data) < 5:
            return {
                'error': f'Insufficient event window data: {len(event_data)} days',
                'estimation_mean': alpha,
                'estimation_std': estimation_std,
                'estimation_n': len(estimation_data),
                'model': 'market_model'
            }

        # Compute abnormal returns
        expected_returns = alpha + beta * event_data['market']
        abnormal_returns = event_data['asset'] - expected_returns

        # CAR and standardized AR
        car = abnormal_returns.sum()
        sar = car / (estimation_std * np.sqrt(len(abnormal_returns)))

        return {
            'estimation_mean': alpha,  # Alpha from market model
            'estimation_std': estimation_std,
            'estimation_n': len(estimation_data),
            'abnormal_returns': abnormal_returns,
            'car': car,
            'sar': sar,
            'event_window_n': len(abnormal_returns),
            'model': 'market_model',
            'beta': beta,
            'alpha': alpha
        }


class EWMarketModel:
    """
    Equal-Weighted Market Model for expected returns.

    Instead of using BTC as the single market proxy (which creates mechanical
    correlation since BTC is in our sample), this uses an equal-weighted
    index of all assets in the sample.

    Benefits:
    - Avoids BTC-dominance bias (~50% of market cap)
    - No mechanical correlation from using an asset as its own benchmark
    - More diversified market proxy

    For each asset, the market return excludes that asset (leave-one-out)
    to avoid mechanical correlation.
    """

    def __init__(
        self,
        estimation_window: int = config.ESTIMATION_WINDOW_DAYS,
        gap_window: int = config.GAP_WINDOW_DAYS
    ):
        self.estimation_window = estimation_window
        self.gap_window = gap_window
        self._returns_dict: Optional[Dict[str, pd.Series]] = None

    def set_returns_dict(self, returns_dict: Dict[str, pd.Series]):
        """Set the full returns dictionary for computing EW market."""
        self._returns_dict = {
            k: v.copy() for k, v in returns_dict.items()
        }
        # Ensure datetime index
        for k in self._returns_dict:
            self._returns_dict[k].index = pd.to_datetime(self._returns_dict[k].index)

    def _compute_ew_market_excluding(self, exclude_symbol: str) -> pd.Series:
        """Compute equal-weighted market return excluding one asset."""
        if self._returns_dict is None:
            raise ValueError("Must call set_returns_dict first")

        other_assets = [
            v for k, v in self._returns_dict.items()
            if k != exclude_symbol
        ]

        if len(other_assets) == 0:
            raise ValueError(f"No other assets to compute market for {exclude_symbol}")

        # Align all series and compute mean
        combined = pd.concat(other_assets, axis=1).dropna()
        return combined.mean(axis=1)

    def compute_abnormal_returns(
        self,
        returns: pd.Series,
        event_date: str,
        event_window: Tuple[int, int] = (-5, 30),
        symbol: str = None
    ) -> Dict:
        """
        Compute abnormal returns using equal-weighted market model.

        Uses leave-one-out EW market: for asset i, market = mean(all assets except i)

        Args:
            returns: Daily return series with datetime index
            event_date: Event date (YYYY-MM-DD)
            event_window: (pre_days, post_days) relative to event
            symbol: Asset symbol (required for leave-one-out)

        Returns:
            Dictionary with estimation stats and abnormal returns
        """
        if self._returns_dict is None:
            # Fall back to constant mean if no returns dict
            cm = ConstantMeanModel(self.estimation_window, self.gap_window)
            result = cm.compute_abnormal_returns(returns, event_date, event_window)
            result['model'] = 'constant_mean_fallback'
            return result

        if symbol is None:
            # Can't do leave-one-out without knowing the symbol
            cm = ConstantMeanModel(self.estimation_window, self.gap_window)
            result = cm.compute_abnormal_returns(returns, event_date, event_window)
            result['model'] = 'constant_mean_no_symbol'
            return result

        event_dt = pd.to_datetime(event_date)
        returns = returns.copy()
        returns.index = pd.to_datetime(returns.index)

        # Compute leave-one-out EW market
        try:
            market_returns = self._compute_ew_market_excluding(symbol)
        except ValueError:
            # Fall back to constant mean
            cm = ConstantMeanModel(self.estimation_window, self.gap_window)
            return cm.compute_abnormal_returns(returns, event_date, event_window)

        # Define periods
        estimation_end = event_dt - pd.Timedelta(days=self.gap_window + 1)
        estimation_start = estimation_end - pd.Timedelta(days=self.estimation_window)

        event_start = event_dt + pd.Timedelta(days=event_window[0])
        event_end = event_dt + pd.Timedelta(days=event_window[1])

        # Align asset and market returns
        combined = pd.DataFrame({
            'asset': returns,
            'market': market_returns
        }).dropna()

        # Get estimation period
        estimation_data = combined[
            (combined.index >= estimation_start) &
            (combined.index <= estimation_end)
        ]

        if len(estimation_data) < config.ESTIMATION_WINDOW_MIN:
            return {
                'error': f'Insufficient estimation data: {len(estimation_data)} days',
                'estimation_mean': np.nan,
                'estimation_std': np.nan,
                'estimation_n': len(estimation_data),
                'model': 'ew_market_model'
            }

        # Estimate market model: R_i = alpha + beta * R_m + epsilon
        X = estimation_data['market'].values
        Y = estimation_data['asset'].values

        # OLS regression
        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            beta_hat = np.linalg.lstsq(X_with_const, Y, rcond=None)[0]
            alpha, beta = beta_hat[0], beta_hat[1]
        except:
            # Fallback to constant mean
            cm = ConstantMeanModel(self.estimation_window, self.gap_window)
            return cm.compute_abnormal_returns(returns, event_date, event_window)

        # Estimation period residuals for standard deviation
        predicted = alpha + beta * estimation_data['market']
        residuals = estimation_data['asset'] - predicted
        estimation_std = residuals.std()

        # Get event window data
        event_data = combined[
            (combined.index >= event_start) &
            (combined.index <= event_end)
        ]

        if len(event_data) < 5:
            return {
                'error': f'Insufficient event window data: {len(event_data)} days',
                'estimation_mean': alpha,
                'estimation_std': estimation_std,
                'estimation_n': len(estimation_data),
                'model': 'ew_market_model'
            }

        # Compute abnormal returns
        expected_returns = alpha + beta * event_data['market']
        abnormal_returns = event_data['asset'] - expected_returns

        # CAR and standardized AR
        car = abnormal_returns.sum()
        sar = car / (estimation_std * np.sqrt(len(abnormal_returns)))

        return {
            'estimation_mean': alpha,
            'estimation_std': estimation_std,
            'estimation_n': len(estimation_data),
            'abnormal_returns': abnormal_returns,
            'car': car,
            'sar': sar,
            'event_window_n': len(abnormal_returns),
            'model': 'ew_market_model',
            'beta': beta,
            'alpha': alpha
        }


class KolariPynnonenTest:
    """
    Kolari-Pynnonen (2010) adjusted t-test for event studies.

    Adjusts for cross-sectional correlation in abnormal returns,
    which is critical when analyzing multiple assets around the same event.
    """

    @staticmethod
    def compute_adjusted_t(
        abnormal_returns: List[pd.Series],
        estimation_stds: List[float],
        estimation_ns: List[int]
    ) -> Dict:
        """
        Compute Kolari-Pynnonen adjusted t-statistic.

        Args:
            abnormal_returns: List of abnormal return series (one per asset)
            estimation_stds: List of estimation period standard deviations
            estimation_ns: List of estimation period sample sizes

        Returns:
            Dictionary with test statistics
        """
        if len(abnormal_returns) < 2:
            # Fall back to single-asset test
            ar = abnormal_returns[0]
            car = ar.sum()
            std = estimation_stds[0]
            t_stat = car / (std * np.sqrt(len(ar)))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(ar) - 1))

            return {
                't_stat': t_stat,
                'p_value': p_value,
                'adjustment_factor': 1.0,
                'cross_correlation': 0.0,
                'n_assets': 1
            }

        # Align abnormal returns to common dates
        aligned = pd.concat(abnormal_returns, axis=1).dropna()

        if len(aligned) < 5:
            return {'error': 'Insufficient aligned data'}

        # Compute cross-sectional average abnormal return
        aar = aligned.mean(axis=1)  # Average across assets

        # Compute variance of AAR
        n_assets = aligned.shape[1]
        n_days = len(aar)

        # Cross-correlation estimate
        cross_corr = aligned.corr().values
        np.fill_diagonal(cross_corr, 0)
        avg_corr = cross_corr.sum() / (n_assets * (n_assets - 1))

        # Adjusted variance
        var_adjustment = 1 + (n_assets - 1) * avg_corr

        # CAR and adjusted t-stat
        car = aar.sum()
        avg_std = np.mean(estimation_stds)

        # Standard t-stat
        t_stat_unadj = car / (avg_std * np.sqrt(n_days) / np.sqrt(n_assets))

        # KP adjusted t-stat
        t_stat_kp = t_stat_unadj / np.sqrt(var_adjustment) if var_adjustment > 0 else t_stat_unadj

        # P-value (two-tailed)
        df = sum(estimation_ns) - n_assets
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat_kp), df))

        return {
            't_stat_unadjusted': t_stat_unadj,
            't_stat_kp': t_stat_kp,
            'p_value': p_value,
            'adjustment_factor': np.sqrt(var_adjustment),
            'cross_correlation': avg_corr,
            'n_assets': n_assets,
            'car': car,
            'aar': aar.values
        }


class BootstrapBCA:
    """
    Bootstrap Bias-Corrected Accelerated (BCA) confidence intervals.

    BCA is preferred for event studies because it:
    - Corrects for bias in the bootstrap distribution
    - Adjusts for skewness (acceleration)
    - Provides more accurate confidence intervals for small samples
    """

    def __init__(
        self,
        n_bootstrap: int = config.BOOTSTRAP_REPLICATIONS,
        confidence: float = config.BOOTSTRAP_CONFIDENCE,
        random_state: int = config.RANDOM_SEED
    ):
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.rng = np.random.default_rng(random_state)

    def compute_bca_ci(
        self,
        data: np.ndarray,
        statistic_func=np.mean
    ) -> Dict:
        """
        Compute BCA confidence interval for a statistic.

        Args:
            data: Array of observations
            statistic_func: Function to compute the statistic (default: mean)

        Returns:
            Dictionary with CI bounds and bootstrap p-value
        """
        n = len(data)
        original_stat = statistic_func(data)

        # Generate bootstrap samples
        bootstrap_stats = np.array([
            statistic_func(self.rng.choice(data, size=n, replace=True))
            for _ in range(self.n_bootstrap)
        ])

        # Bias correction (z0)
        proportion_below = np.mean(bootstrap_stats < original_stat)
        z0 = stats.norm.ppf(proportion_below) if 0 < proportion_below < 1 else 0

        # Acceleration (a) via jackknife
        jackknife_stats = np.array([
            statistic_func(np.delete(data, i))
            for i in range(n)
        ])
        theta_dot = jackknife_stats.mean()
        num = np.sum((theta_dot - jackknife_stats) ** 3)
        denom = 6 * (np.sum((theta_dot - jackknife_stats) ** 2) ** 1.5)
        a = num / denom if denom != 0 else 0

        # Adjusted percentiles
        alpha = (1 - self.confidence) / 2
        z_alpha_low = stats.norm.ppf(alpha)
        z_alpha_high = stats.norm.ppf(1 - alpha)

        # BCA adjustment formula
        def bca_percentile(z_alpha):
            numerator = z0 + z_alpha
            denominator = 1 - a * (z0 + z_alpha)
            if denominator == 0:
                return 0.5
            z_adjusted = z0 + numerator / denominator
            return stats.norm.cdf(z_adjusted)

        p_low = bca_percentile(z_alpha_low)
        p_high = bca_percentile(z_alpha_high)

        # Get confidence interval from bootstrap distribution
        ci_low = np.percentile(bootstrap_stats, p_low * 100)
        ci_high = np.percentile(bootstrap_stats, p_high * 100)

        # Bootstrap p-value (two-tailed test against zero)
        if original_stat >= 0:
            p_value = 2 * np.mean(bootstrap_stats <= 0)
        else:
            p_value = 2 * np.mean(bootstrap_stats >= 0)
        p_value = min(p_value, 1.0)

        return {
            'statistic': original_stat,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'ci_width': ci_high - ci_low,
            'bootstrap_std': bootstrap_stats.std(),
            'bootstrap_pvalue': p_value,
            'bias_correction': z0,
            'acceleration': a
        }


class DifferenceInDifferences:
    """
    Difference-in-Differences estimator for comparing infrastructure vs regulatory events.

    DiD = (Post_infra - Pre_infra) - (Post_reg - Pre_reg)

    Tests whether infrastructure events have different liquidity impact
    than regulatory events.
    """

    @staticmethod
    def compute_did(
        infra_pre: np.ndarray,
        infra_post: np.ndarray,
        reg_pre: np.ndarray,
        reg_post: np.ndarray
    ) -> Dict:
        """
        Compute DiD estimate with standard errors.

        Args:
            infra_pre: Pre-event values for infrastructure events
            infra_post: Post-event values for infrastructure events
            reg_pre: Pre-event values for regulatory events
            reg_post: Post-event values for regulatory events

        Returns:
            Dictionary with DiD estimate and test statistics
        """
        # Group means
        infra_diff = infra_post.mean() - infra_pre.mean()
        reg_diff = reg_post.mean() - reg_pre.mean()
        did = infra_diff - reg_diff

        # Standard errors (assuming independence)
        se_infra_pre = infra_pre.std() / np.sqrt(len(infra_pre))
        se_infra_post = infra_post.std() / np.sqrt(len(infra_post))
        se_reg_pre = reg_pre.std() / np.sqrt(len(reg_pre))
        se_reg_post = reg_post.std() / np.sqrt(len(reg_post))

        se_did = np.sqrt(
            se_infra_pre**2 + se_infra_post**2 +
            se_reg_pre**2 + se_reg_post**2
        )

        # t-statistic and p-value
        t_stat = did / se_did if se_did > 0 else 0
        df = len(infra_pre) + len(infra_post) + len(reg_pre) + len(reg_post) - 4
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        return {
            'did_estimate': did,
            'infra_diff': infra_diff,
            'reg_diff': reg_diff,
            'se_did': se_did,
            't_stat': t_stat,
            'p_value': p_value,
            'df': df,
            'infra_pre_mean': infra_pre.mean(),
            'infra_post_mean': infra_post.mean(),
            'reg_pre_mean': reg_pre.mean(),
            'reg_post_mean': reg_post.mean()
        }

    @staticmethod
    def bootstrap_did(
        infra_pre: np.ndarray,
        infra_post: np.ndarray,
        reg_pre: np.ndarray,
        reg_post: np.ndarray,
        n_bootstrap: int = 5000,
        confidence: float = 0.95,
        random_state: int = 42
    ) -> Dict:
        """Compute bootstrap confidence interval for DiD."""
        rng = np.random.default_rng(random_state)

        # Original DiD
        original_did = (
            (infra_post.mean() - infra_pre.mean()) -
            (reg_post.mean() - reg_pre.mean())
        )

        # Bootstrap
        bootstrap_dids = []
        for _ in range(n_bootstrap):
            b_infra_pre = rng.choice(infra_pre, size=len(infra_pre), replace=True)
            b_infra_post = rng.choice(infra_post, size=len(infra_post), replace=True)
            b_reg_pre = rng.choice(reg_pre, size=len(reg_pre), replace=True)
            b_reg_post = rng.choice(reg_post, size=len(reg_post), replace=True)

            b_did = (
                (b_infra_post.mean() - b_infra_pre.mean()) -
                (b_reg_post.mean() - b_reg_pre.mean())
            )
            bootstrap_dids.append(b_did)

        bootstrap_dids = np.array(bootstrap_dids)

        # Percentile CI
        alpha = (1 - confidence) / 2
        ci_low = np.percentile(bootstrap_dids, alpha * 100)
        ci_high = np.percentile(bootstrap_dids, (1 - alpha) * 100)

        # Bootstrap p-value
        if original_did >= 0:
            p_value = 2 * np.mean(bootstrap_dids <= 0)
        else:
            p_value = 2 * np.mean(bootstrap_dids >= 0)

        return {
            'did_estimate': original_did,
            'bootstrap_ci_low': ci_low,
            'bootstrap_ci_high': ci_high,
            'bootstrap_pvalue': min(p_value, 1.0),
            'bootstrap_std': bootstrap_dids.std()
        }


class EventStudyAnalyzer:
    """
    Main class for running event studies.

    Combines all components: constant mean model, KP test, bootstrap BCA.
    """

    def __init__(self):
        self.model = ConstantMeanModel()
        self.kp_test = KolariPynnonenTest()
        self.bootstrap = BootstrapBCA()

    def analyze_single_event(
        self,
        returns: pd.Series,
        event_date: str,
        event_id: int,
        event_name: str,
        event_type: str,
        symbol: str,
        event_window: Tuple[int, int] = (-5, 30)
    ) -> Optional[EventStudyResult]:
        """
        Analyze a single event for a single asset.

        Args:
            returns: Daily return series
            event_date: Event date
            event_id: Event identifier
            event_name: Event name
            event_type: 'Infrastructure' or 'Regulatory'
            symbol: Asset symbol
            event_window: Event window as (pre, post) days

        Returns:
            EventStudyResult or None if insufficient data
        """
        # Compute abnormal returns
        result = self.model.compute_abnormal_returns(
            returns, event_date, event_window
        )

        if 'error' in result:
            warnings.warn(f"Event {event_id} ({symbol}): {result['error']}")
            return None

        ar = result['abnormal_returns']
        car = result['car']

        # T-test
        t_stat = car / (result['estimation_std'] * np.sqrt(len(ar)))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), result['estimation_n'] - 1))

        # Bootstrap CAR
        bootstrap_result = self.bootstrap.compute_bca_ci(
            ar.values,
            statistic_func=np.sum  # CAR = sum of abnormal returns
        )

        return EventStudyResult(
            event_id=event_id,
            event_date=event_date,
            event_type=event_type,
            event_name=event_name,
            symbol=symbol,
            estimation_mean=result['estimation_mean'],
            estimation_std=result['estimation_std'],
            estimation_n=result['estimation_n'],
            car=car,
            car_t_stat=t_stat,
            car_p_value=p_value,
            car_bootstrap_ci_low=bootstrap_result['ci_low'],
            car_bootstrap_ci_high=bootstrap_result['ci_high'],
            car_bootstrap_pvalue=bootstrap_result['bootstrap_pvalue'],
            aar=ar.values,
            event_window=event_window
        )

    def analyze_event_cross_sectional(
        self,
        returns_dict: Dict[str, pd.Series],
        event_date: str,
        event_id: int,
        event_name: str,
        event_type: str,
        event_window: Tuple[int, int] = (-5, 30)
    ) -> Dict:
        """
        Analyze event across multiple assets with Kolari-Pynnonen adjustment.

        Args:
            returns_dict: Dictionary mapping symbol to return series
            event_date: Event date
            event_id: Event identifier
            event_name: Event name
            event_type: Event type
            event_window: Event window

        Returns:
            Dictionary with cross-sectional test results
        """
        abnormal_returns = []
        estimation_stds = []
        estimation_ns = []
        individual_results = []

        for symbol, returns in returns_dict.items():
            result = self.model.compute_abnormal_returns(
                returns, event_date, event_window
            )

            if 'error' not in result:
                abnormal_returns.append(result['abnormal_returns'])
                estimation_stds.append(result['estimation_std'])
                estimation_ns.append(result['estimation_n'])
                individual_results.append({
                    'symbol': symbol,
                    'car': result['car'],
                    'sar': result['sar']
                })

        if len(abnormal_returns) == 0:
            return {'error': 'No valid data for any asset'}

        # Kolari-Pynnonen test
        kp_result = self.kp_test.compute_adjusted_t(
            abnormal_returns,
            estimation_stds,
            estimation_ns
        )

        # Bootstrap the cross-sectional CAR
        if 'car' in kp_result:
            aligned = pd.concat(abnormal_returns, axis=1).dropna()
            aar = aligned.mean(axis=1)  # Average abnormal return
            bootstrap_result = self.bootstrap.compute_bca_ci(
                aar.values,
                statistic_func=np.sum
            )
        else:
            bootstrap_result = {}

        return {
            'event_id': event_id,
            'event_date': event_date,
            'event_name': event_name,
            'event_type': event_type,
            'n_assets': len(abnormal_returns),
            'kp_test': kp_result,
            'bootstrap': bootstrap_result,
            'individual_results': individual_results
        }


def compute_pre_event_car(
    returns: pd.Series,
    event_date: str,
    pre_window: Tuple[int, int] = (-30, -1),
    estimation_window: int = config.ESTIMATION_WINDOW_DAYS,
    gap_window: int = config.GAP_WINDOW_DAYS
) -> Dict:
    """
    Compute CAR for pre-event period to detect anticipation effects.

    Markets may anticipate certain events (especially regulatory announcements),
    leading to price movements before the official event date. This function
    computes CAR in a pre-event window to quantify anticipation.

    Args:
        returns: Daily return series with datetime index
        event_date: Event date (YYYY-MM-DD)
        pre_window: Window before event to compute CAR (default: [-30, -1])
        estimation_window: Estimation window length for expected returns
        gap_window: Gap between estimation and event window

    Returns:
        Dictionary with pre-event CAR statistics
    """
    event_dt = pd.to_datetime(event_date)
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)

    # Estimation period ends before the pre-event window
    pre_window_start = event_dt + pd.Timedelta(days=pre_window[0])
    estimation_end = pre_window_start - pd.Timedelta(days=gap_window + 1)
    estimation_start = estimation_end - pd.Timedelta(days=estimation_window)

    # Get estimation period returns
    estimation_returns = returns[
        (returns.index >= estimation_start) &
        (returns.index <= estimation_end)
    ].dropna()

    if len(estimation_returns) < config.ESTIMATION_WINDOW_MIN:
        return {
            'error': f'Insufficient estimation data: {len(estimation_returns)} days',
            'pre_event_car': np.nan
        }

    # Expected return (constant mean)
    expected_return = estimation_returns.mean()
    estimation_std = estimation_returns.std()

    # Pre-event window returns
    pre_window_end = event_dt + pd.Timedelta(days=pre_window[1])
    pre_returns = returns[
        (returns.index >= pre_window_start) &
        (returns.index <= pre_window_end)
    ].dropna()

    if len(pre_returns) < 5:
        return {
            'error': f'Insufficient pre-event data: {len(pre_returns)} days',
            'pre_event_car': np.nan
        }

    # Abnormal returns and CAR
    pre_abnormal = pre_returns - expected_return
    pre_car = pre_abnormal.sum()
    pre_sar = pre_car / (estimation_std * np.sqrt(len(pre_abnormal)))

    # T-test
    t_stat = pre_car / (estimation_std * np.sqrt(len(pre_abnormal)))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(estimation_returns) - 1))

    return {
        'pre_event_car': pre_car,
        'pre_event_sar': pre_sar,
        'pre_event_t_stat': t_stat,
        'pre_event_p_value': p_value,
        'pre_window': pre_window,
        'pre_window_n': len(pre_abnormal),
        'estimation_mean': expected_return,
        'estimation_std': estimation_std
    }


def run_market_model_analysis(
    returns_dict: Dict[str, pd.Series],
    events: List[Dict],
    market_proxy: str = 'BTC',
    event_window: Tuple[int, int] = (-5, 30)
) -> pd.DataFrame:
    """
    Run event study using market model instead of constant mean.

    Uses BTC as market proxy. For BTC itself, uses constant mean model.

    Args:
        returns_dict: Dictionary of return series by symbol
        events: List of event dictionaries
        market_proxy: Symbol to use as market (default: BTC)
        event_window: Event window (pre, post)

    Returns:
        DataFrame with market-model adjusted CARs
    """
    # Initialize market model
    mm = MarketModel(market_proxy=market_proxy)

    if market_proxy in returns_dict:
        mm.set_market_returns(returns_dict[market_proxy])
    else:
        # Fall back to first asset
        first_asset = list(returns_dict.keys())[0]
        mm.set_market_returns(returns_dict[first_asset])

    results = []

    for event in events:
        event_id = event['event_id']
        event_date = event['date']
        event_type = event.get('type_detailed', event.get('type', 'Unknown'))
        event_name = event.get('title', event.get('label', ''))

        for symbol, returns in returns_dict.items():
            result = mm.compute_abnormal_returns(
                returns, event_date, event_window, symbol=symbol
            )

            if 'error' not in result:
                from scipy import stats as sp_stats

                car = result['car']
                est_std = result['estimation_std']
                n = result['event_window_n']

                t_stat = car / (est_std * np.sqrt(n)) if est_std > 0 else 0
                p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), result['estimation_n'] - 1))

                results.append({
                    'event_id': event_id,
                    'event_date': event_date,
                    'event_type': event_type,
                    'event_name': event_name,
                    'symbol': symbol,
                    'car': car,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'estimation_n': result['estimation_n'],
                    'model': result.get('model', 'unknown'),
                    'beta': result.get('beta', np.nan),
                    'alpha': result.get('alpha', np.nan)
                })

    return pd.DataFrame(results)


def run_ew_market_analysis(
    returns_dict: Dict[str, pd.Series],
    events: List[Dict],
    event_window: Tuple[int, int] = (-5, 30)
) -> pd.DataFrame:
    """
    Run event study using equal-weighted market model.

    Uses leave-one-out EW market: for each asset, the market is the
    equal-weighted average of all OTHER assets.

    Args:
        returns_dict: Dictionary of return series by symbol
        events: List of event dictionaries
        event_window: Event window (pre, post)

    Returns:
        DataFrame with EW market-model adjusted CARs
    """
    # Initialize EW market model
    ew = EWMarketModel()
    ew.set_returns_dict(returns_dict)

    results = []

    for event in events:
        event_id = event['event_id']
        event_date = event['date']
        event_type = event.get('type_detailed', event.get('type', 'Unknown'))
        event_name = event.get('title', event.get('label', ''))

        for symbol, returns in returns_dict.items():
            result = ew.compute_abnormal_returns(
                returns, event_date, event_window, symbol=symbol
            )

            if 'error' not in result:
                car = result['car']
                est_std = result['estimation_std']
                n = result['event_window_n']

                t_stat = car / (est_std * np.sqrt(n)) if est_std > 0 else 0
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), result['estimation_n'] - 1))

                results.append({
                    'event_id': event_id,
                    'event_date': event_date,
                    'event_type': event_type,
                    'event_name': event_name,
                    'symbol': symbol,
                    'car': car,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'estimation_n': result['estimation_n'],
                    'model': result.get('model', 'unknown'),
                    'beta': result.get('beta', np.nan),
                    'alpha': result.get('alpha', np.nan)
                })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing Event Study Module...")

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    returns = pd.Series(np.random.randn(500) * 0.02, index=dates)

    # Add event effect
    event_date = '2023-08-01'
    event_idx = returns.index.get_loc(pd.to_datetime(event_date))
    returns.iloc[event_idx:event_idx+10] += 0.03  # Positive shock

    analyzer = EventStudyAnalyzer()

    result = analyzer.analyze_single_event(
        returns=returns,
        event_date=event_date,
        event_id=1,
        event_name='Test Event',
        event_type='Infrastructure',
        symbol='TEST'
    )

    if result:
        print(f"\nCAR: {result.car:.4f}")
        print(f"t-stat: {result.car_t_stat:.4f}")
        print(f"p-value: {result.car_p_value:.4f}")
        print(f"Bootstrap CI: [{result.car_bootstrap_ci_low:.4f}, {result.car_bootstrap_ci_high:.4f}]")
        print("\n[SUCCESS] Event study module working!")
    else:
        print("\n[FAIL] No result returned")
