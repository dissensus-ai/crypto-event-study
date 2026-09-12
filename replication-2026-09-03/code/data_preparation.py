"""
Data preparation module for cryptocurrency event study.
Handles loading, cleaning, and merging of price, event, and sentiment data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
from pathlib import Path

try:
    from . import config
except ImportError:
    import config


class DataPreparation:
    """
    Main class for preparing cryptocurrency event study data.
    Handles price data, event dummies, and sentiment merging.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data preparation module.

        Args:
            data_path: Path to the data directory containing CSV files (defaults to config.DATA_DIR)
        """
        self.data_path = Path(data_path) if data_path else Path(config.DATA_DIR)
        self.cryptocurrencies = ['btc', 'eth', 'xrp', 'bnb', 'ltc', 'ada']
        # Ensure analysis window is UTC timezone-aware
        self.start_date = pd.Timestamp('2019-01-01', tz='UTC')
        self.end_date = pd.Timestamp('2025-08-31', tz='UTC')

        # Special event handling configurations
        self.special_events = {
            'sec_twin_suits': {
                'event_ids': [31, 32],
                'composite_name': 'D_SEC_enforcement_2023',
                'window_start': pd.Timestamp('2023-06-02', tz='UTC'),
                'window_end': pd.Timestamp('2023-06-09', tz='UTC')
            },
            'eip_poly_overlap': {
                'event_ids': [17, 18],
                'overlap_dates': [
                    pd.Timestamp('2021-08-07', tz='UTC'),
                    pd.Timestamp('2021-08-08', tz='UTC')
                ],
                'adjustment': -0.5
            },
            'bybit_sec_truncate': {
                'event_ids': [43, 44],
                'bybit_end': pd.Timestamp('2025-02-23', tz='UTC'),
                'sec_start': pd.Timestamp('2025-02-27', tz='UTC')
            }
        }

    def _ensure_utc_timezone(self, ts):
        """
        Ensure the given timestamp-like object is converted to UTC timezone.

        Accepts scalar timestamps, strings, arrays, Series, or DatetimeIndex and
        returns the same structure normalized to UTC.
        """
        return pd.to_datetime(ts, utc=True)

    def load_crypto_prices(self, crypto: str) -> pd.DataFrame:
        """
        Load cryptocurrency price data from CSV file.

        Args:
            crypto: Cryptocurrency symbol (e.g., 'btc', 'eth')

        Returns:
            DataFrame with date index and price columns
        """
        file_path = self.data_path / f"{crypto}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Price data file not found: {file_path}")

        # Load data
        df = pd.read_csv(file_path)

        # Parse and set datetime index in UTC
        df['snapped_at'] = self._ensure_utc_timezone(df['snapped_at'])
        df.set_index('snapped_at', inplace=True)

        # Filter date range
        df = df.loc[self.start_date:self.end_date]

        # Ensure we have the required columns
        if 'price' not in df.columns:
            raise ValueError(f"Price column not found in {crypto} data")

        # Sort by date and remove duplicates
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        return df

    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate log returns from price series.

        Args:
            prices: Series of prices

        Returns:
            Series of log returns (multiplied by 100 for percentage)
        """
        # Calculate log returns: ln(P_t / P_{t-1}) * 100
        log_returns = np.log(prices / prices.shift(1)) * 100

        # Drop first NaN value
        return log_returns.dropna()

    def winsorize_returns(self, returns: pd.Series,
                         window: int = 30,
                         n_std: float = 5.0) -> pd.Series:
        """
        Winsorize returns at specified standard deviations using rolling window.

        Args:
            returns: Series of returns
            window: Rolling window size in days
            n_std: Number of standard deviations for winsorization

        Returns:
            Winsorized returns series
        """
        # Calculate rolling mean and std
        rolling_mean = returns.rolling(window=window, min_periods=1).mean()
        rolling_std = returns.rolling(window=window, min_periods=1).std()

        # Calculate bounds
        upper_bound = rolling_mean + n_std * rolling_std
        lower_bound = rolling_mean - n_std * rolling_std

        # Winsorize
        winsorized = returns.clip(lower=lower_bound, upper=upper_bound)

        return winsorized

    def load_events(self) -> pd.DataFrame:
        """
        Load event data from CSV file.

        Returns:
            DataFrame with event information
        """
        file_path = self.data_path / "events.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Events file not found: {file_path}")

        df = pd.read_csv(file_path)
        # Ensure event dates are UTC timezone-aware
        df['date'] = self._ensure_utc_timezone(df['date'])

        return df

    def create_event_window(self, event_date: pd.Timestamp,
                          days_before: int = 3,
                          days_after: int = 3) -> List[pd.Timestamp]:
        """
        Create event window dates.

        Args:
            event_date: Date of the event
            days_before: Days before event
            days_after: Days after event

        Returns:
            List of dates in event window
        """
        # Normalize to UTC before calculations
        event_date = self._ensure_utc_timezone(event_date)
        start = event_date - timedelta(days=days_before)
        end = event_date + timedelta(days=days_after)

        return pd.date_range(start=start, end=end, freq='D').tolist()

    def create_event_dummies(self, date_index: pd.DatetimeIndex,
                           events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create event dummy variables with special overlap handling.

        Args:
            date_index: DatetimeIndex for the data
            events_df: DataFrame with event information

        Returns:
            DataFrame with event dummy variables
        """
        # Initialize dummy DataFrame
        dummies = pd.DataFrame(index=date_index)

        # Process each event
        for idx, event in events_df.iterrows():
            event_id = event['event_id']
            # Ensure timezone-aware UTC event date
            event_date = self._ensure_utc_timezone(event['date'])

            # Check if this is part of special handling
            if event_id in self.special_events['sec_twin_suits']['event_ids']:
                # SEC Twin Suits - create composite dummy
                if event_id == 31:  # Only create once for first event
                    dummy_name = self.special_events['sec_twin_suits']['composite_name']
                    window_start = self.special_events['sec_twin_suits']['window_start']
                    window_end = self.special_events['sec_twin_suits']['window_end']

                    dummies[dummy_name] = 0
                    mask = (dummies.index >= window_start) & (dummies.index <= window_end)
                    dummies.loc[mask, dummy_name] = 1
                continue

            elif event_id == 43:  # Bybit hack - truncated window
                dummy_name = f"D_event_{event_id}"
                dummies[dummy_name] = 0

                window = self.create_event_window(event_date)
                truncate_date = self.special_events['bybit_sec_truncate']['bybit_end']

                for date in window:
                    date_utc = self._ensure_utc_timezone(date)
                    if date_utc <= truncate_date and date_utc in dummies.index:
                        dummies.loc[date_utc, dummy_name] = 1
                continue

            elif event_id == 44:  # SEC dismissal - truncated window
                dummy_name = f"D_event_{event_id}"
                dummies[dummy_name] = 0

                window = self.create_event_window(event_date)
                start_date = self.special_events['bybit_sec_truncate']['sec_start']

                for date in window:
                    date_utc = self._ensure_utc_timezone(date)
                    if date_utc >= start_date and date_utc in dummies.index:
                        dummies.loc[date_utc, dummy_name] = 1
                continue

            # Regular event handling
            dummy_name = f"D_event_{event_id}"
            dummies[dummy_name] = 0

            window = self.create_event_window(event_date)
            for date in window:
                date_utc = self._ensure_utc_timezone(date)
                if date_utc in dummies.index:
                    dummies.loc[date_utc, dummy_name] = 1

        # Apply EIP-1559 & Poly hack overlap adjustment
        # IMPORTANT DOCUMENTATION: Overlap Adjustment Interpretation
        # =========================================================
        # When two events overlap (e.g., EIP-1559 and Polygon hack on Aug 7-8, 2021),
        # we assign each event a weight of 0.5 instead of 1.0 on overlapping days.
        #
        # Coefficient Interpretation:
        # - Model coefficients represent the effect when dummy = 1.0
        # - On overlap days with dummy = 0.5, actual effect = coefficient * 0.5
        # - Total volatility impact on overlap = 0.5*coef_event17 + 0.5*coef_event18
        #
        # Rationale: Prevents double-counting volatility on days with multiple events
        # Trade-off: May underestimate if events have independent additive effects
        # Alternative: Could use max(event1, event2) or keep both at 1.0
        if 'D_event_17' in dummies.columns and 'D_event_18' in dummies.columns:
            overlap_dates = self.special_events['eip_poly_overlap']['overlap_dates']
            adjustment = self.special_events['eip_poly_overlap']['adjustment']  # -0.5

            for date in overlap_dates:
                if date in dummies.index:
                    # Apply -0.5 adjustment: 1 + (-0.5) = 0.5 for overlapping days
                    if dummies.loc[date, 'D_event_17'] == 1 and dummies.loc[date, 'D_event_18'] == 1:
                        dummies.loc[date, 'D_event_17'] = 1 + adjustment  # Becomes 0.5
                        dummies.loc[date, 'D_event_18'] = 1 + adjustment  # Becomes 0.5

        # Create aggregate event type dummies
        events_df_indexed = events_df.set_index('event_id')

        # Infrastructure events dummy
        infra_events = events_df[events_df['type'] == 'Infrastructure']['event_id'].tolist()
        infra_cols = [f"D_event_{eid}" for eid in infra_events if f"D_event_{eid}" in dummies.columns]
        if 'D_SEC_enforcement_2023' not in dummies.columns:
            dummies['D_infrastructure'] = dummies[infra_cols].max(axis=1)
        else:
            # Exclude SEC composite from infrastructure
            dummies['D_infrastructure'] = dummies[infra_cols].max(axis=1)

        # Regulatory events dummy
        reg_events = events_df[events_df['type'] == 'Regulatory']['event_id'].tolist()
        reg_cols = [f"D_event_{eid}" for eid in reg_events if f"D_event_{eid}" in dummies.columns]
        if 'D_SEC_enforcement_2023' in dummies.columns:
            reg_cols.append('D_SEC_enforcement_2023')
        dummies['D_regulatory'] = dummies[reg_cols].max(axis=1) if reg_cols else 0

        return dummies

    def load_gdelt_sentiment(self) -> pd.DataFrame:
        """
        Load and process GDELT sentiment data with proper 3-stage methodology:

        Stage 1: Load raw GDELT data (already in CSV)
        Stage 2: Z-score normalization with 52-week rolling window
        Stage 3: Theme decomposition using regulatory/infrastructure proportions

        Returns:
            DataFrame with computed normalized and decomposed sentiment columns
        """
        file_path = self.data_path / "gdelt.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"GDELT sentiment file not found: {file_path}")

        # Stage 1: Load raw data
        df = pd.read_csv(file_path)
        df['week_start'] = pd.to_datetime(df['week_start'], utc=True)
        df.set_index('week_start', inplace=True)
        df = df.sort_index()

        print("Processing GDELT sentiment data...")
        print(f"  Data range: {df.index.min()} to {df.index.max()}")
        print(f"  Total weeks: {len(df)}")

        # Stage 2: Z-score normalization with 52-week rolling window
        # Use 26-week initialization period as per methodology
        window_size = 52
        min_periods = 26  # Start calculating after 26 weeks of data

        # Calculate rolling statistics
        rolling_mean = df['S_gdelt_raw'].rolling(
            window=window_size,
            min_periods=min_periods,
            center=False
        ).mean()

        rolling_std = df['S_gdelt_raw'].rolling(
            window=window_size,
            min_periods=min_periods,
            center=False
        ).std()

        # Compute normalized sentiment (z-score)
        # S_gdelt_normalized = (S_gdelt_raw - rolling_mean) / rolling_std
        df['S_gdelt_normalized'] = (df['S_gdelt_raw'] - rolling_mean) / rolling_std

        # Handle edge cases where std is 0 or very small
        df.loc[rolling_std < 0.001, 'S_gdelt_normalized'] = 0

        # Stage 3: Theme decomposition using proportions
        # S_reg_decomposed = S_gdelt_normalized × reg_proportion
        # S_infra_decomposed = S_gdelt_normalized × infra_proportion
        df['S_reg_decomposed'] = df['S_gdelt_normalized'] * df['reg_proportion']
        df['S_infra_decomposed'] = df['S_gdelt_normalized'] * df['infra_proportion']

        # Handle missing values before June 2019 (set to 0)
        # This accounts for the initialization period where we don't have enough data
        cutoff_date = pd.Timestamp('2019-06-01', tz='UTC')
        sentiment_cols = ['S_gdelt_normalized', 'S_reg_decomposed', 'S_infra_decomposed']

        for col in sentiment_cols:
            # Set NaN values before cutoff to 0
            df.loc[df.index < cutoff_date, col] = df.loc[df.index < cutoff_date, col].fillna(0)
            # Also handle any remaining NaN values after cutoff (shouldn't be many)
            df[col] = df[col].fillna(0)

        # Print summary statistics
        print(f"\nGDELT sentiment processing complete:")
        print(f"  Normalized sentiment range: [{df['S_gdelt_normalized'].min():.3f}, {df['S_gdelt_normalized'].max():.3f}]")
        print(f"  Non-zero normalized values: {(df['S_gdelt_normalized'] != 0).sum()} weeks")
        print(f"  Regulatory decomposed range: [{df['S_reg_decomposed'].min():.3f}, {df['S_reg_decomposed'].max():.3f}]")
        print(f"  Infrastructure decomposed range: [{df['S_infra_decomposed'].min():.3f}, {df['S_infra_decomposed'].max():.3f}]")

        return df

    def merge_sentiment_data(self, daily_data: pd.DataFrame,
                           sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge weekly sentiment data with daily data using forward-fill.

        Args:
            daily_data: DataFrame with daily frequency
            sentiment_df: DataFrame with weekly sentiment

        Returns:
            Merged DataFrame with forward-filled sentiment
        """
        # Select relevant sentiment columns
        sentiment_cols = ['S_gdelt_normalized', 'S_reg_decomposed', 'S_infra_decomposed']

        # Filter columns that exist
        available_cols = [col for col in sentiment_cols if col in sentiment_df.columns]
        sentiment_subset = sentiment_df[available_cols].copy()

        # Create daily index with explicit UTC timezone
        daily_index = pd.date_range(start=sentiment_subset.index.min(),
                                   end=daily_data.index.max(),
                                   freq='D', tz='UTC')

        # Use reindex with forward fill
        sentiment_daily = sentiment_subset.reindex(daily_index).ffill()

        # Handle missing sentiment before June 2019 (set to 0)
        cutoff_date = pd.Timestamp('2019-06-01', tz='UTC')
        sentiment_daily.loc[sentiment_daily.index < cutoff_date] = 0

        # Merge with daily data
        merged = daily_data.merge(sentiment_daily,
                                 left_index=True,
                                 right_index=True,
                                 how='left')

        # Fill any remaining NaN values with 0
        for col in available_cols:
            if col in merged.columns:
                merged[col].fillna(0, inplace=True)

        return merged

    def prepare_crypto_data(self, crypto: str,
                          include_events: bool = True,
                          include_sentiment: bool = True) -> pd.DataFrame:
        """
        Complete data preparation pipeline for a single cryptocurrency.

        Args:
            crypto: Cryptocurrency symbol
            include_events: Whether to include event dummies
            include_sentiment: Whether to include sentiment data

        Returns:
            Prepared DataFrame with all features
        """
        print(f"Preparing data for {crypto.upper()}...")

        # Load price data
        price_data = self.load_crypto_prices(crypto)

        # Calculate returns
        price_data['returns'] = self.calculate_log_returns(price_data['price'])

        # Winsorize returns
        price_data['returns_winsorized'] = self.winsorize_returns(price_data['returns'])

        # Prepare base dataframe
        result = price_data[['price', 'returns', 'returns_winsorized']].copy()

        # Add event dummies if requested
        if include_events:
            events_df = self.load_events()
            event_dummies = self.create_event_dummies(result.index, events_df)
            result = result.merge(event_dummies, left_index=True, right_index=True, how='left')

            # Fill NaN values in dummy variables with 0
            dummy_cols = [col for col in result.columns if col.startswith('D_')]
            result[dummy_cols] = result[dummy_cols].fillna(0)

        # Add sentiment if requested
        if include_sentiment:
            sentiment_df = self.load_gdelt_sentiment()
            result = self.merge_sentiment_data(result, sentiment_df)

        print(f"Data preparation complete for {crypto.upper()}")
        print(f"  Shape: {result.shape}")
        print(f"  Date range: {result.index.min()} to {result.index.max()}")

        return result

    def prepare_all_cryptos(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for all cryptocurrencies.

        Args:
            **kwargs: Arguments to pass to prepare_crypto_data

        Returns:
            Dictionary mapping crypto symbols to prepared DataFrames
        """
        results = {}

        for crypto in self.cryptocurrencies:
            try:
                results[crypto] = self.prepare_crypto_data(crypto, **kwargs)
            except Exception as e:
                warnings.warn(f"Failed to prepare data for {crypto}: {str(e)}")
                continue

        return results

    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate prepared data for quality issues.

        Args:
            df: Prepared DataFrame

        Returns:
            Dictionary with validation results
        """
        validation = {}

        # Check for missing values
        validation['missing_values'] = df.isnull().sum().to_dict()

        # Check for infinite values in returns
        if 'returns_winsorized' in df.columns:
            validation['infinite_returns'] = np.isinf(df['returns_winsorized']).sum()

        # Check date continuity
        expected_days = (df.index.max() - df.index.min()).days + 1
        actual_days = len(df)
        validation['missing_days'] = expected_days - actual_days

        # Check for extreme returns (after winsorization)
        if 'returns_winsorized' in df.columns:
            returns = df['returns_winsorized'].dropna()
            validation['max_return'] = returns.max()
            validation['min_return'] = returns.min()
            validation['return_std'] = returns.std()

        # Check event dummy consistency
        event_cols = [col for col in df.columns if col.startswith('D_event_')]
        if event_cols:
            validation['total_event_days'] = sum(df[event_cols].sum())
            validation['events_with_data'] = sum(df[event_cols].sum() > 0)

        return validation


# Utility functions for standalone use
def load_and_prepare_single_crypto(crypto: str, data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load and prepare data for a single cryptocurrency.

    Args:
        crypto: Cryptocurrency symbol
        data_path: Path to data directory (defaults to config.DATA_DIR)

    Returns:
        Prepared DataFrame
    """
    prep = DataPreparation(data_path)
    return prep.prepare_crypto_data(crypto)


def load_and_prepare_all_cryptos(data_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load and prepare data for all cryptocurrencies.

    Args:
        data_path: Path to data directory (defaults to config.DATA_DIR)

    Returns:
        Dictionary of prepared DataFrames
    """
    prep = DataPreparation(data_path)
    return prep.prepare_all_cryptos()
