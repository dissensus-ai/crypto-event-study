"""
Configuration settings for the Sentiment Without Structure event study.

Centralizes all parameters, paths, and settings for the analysis.
"""

import os
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==============================================================================
# PATHS
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
TABLES_DIR = OUTPUTS_DIR / 'tables'
PUBLICATION_DIR = OUTPUTS_DIR / 'publication'

# Create directories
for d in [DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR, PUBLICATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# ASSET UNIVERSE
# ==============================================================================

# Tier 1: Large-cap leaders (always included)
TIER1_ASSETS = ['BTC', 'ETH']

# Tier 2: Alternative L1s
TIER2_ASSETS = ['SOL', 'ADA', 'AVAX', 'DOT']

# Tier 3: DeFi protocol tokens
TIER3_ASSETS = ['UNI', 'AAVE']

# Tier 4: Stablecoins (control group - should show minimal response)
STABLECOINS = ['USDT', 'USDC']

# Full asset universe
ALL_ASSETS = TIER1_ASSETS + TIER2_ASSETS + TIER3_ASSETS + STABLECOINS

# Asset metadata
ASSET_METADATA = {
    'BTC': {'name': 'Bitcoin', 'tier': 1, 'type': 'L1', 'futures_since': '2019-09-08'},
    'ETH': {'name': 'Ethereum', 'tier': 1, 'type': 'L1', 'futures_since': '2019-09-08'},
    'SOL': {'name': 'Solana', 'tier': 2, 'type': 'L1', 'futures_since': '2021-08-18'},
    'ADA': {'name': 'Cardano', 'tier': 2, 'type': 'L1', 'futures_since': '2020-03-23'},
    'AVAX': {'name': 'Avalanche', 'tier': 2, 'type': 'L1', 'futures_since': '2021-09-22'},
    'DOT': {'name': 'Polkadot', 'tier': 2, 'type': 'L1', 'futures_since': '2020-09-16'},
    'UNI': {'name': 'Uniswap', 'tier': 3, 'type': 'DeFi', 'futures_since': '2020-09-25'},
    'AAVE': {'name': 'Aave', 'tier': 3, 'type': 'DeFi', 'futures_since': '2020-10-19'},
    'USDT': {'name': 'Tether', 'tier': 4, 'type': 'Stablecoin', 'futures_since': None},
    'USDC': {'name': 'USD Coin', 'tier': 4, 'type': 'Stablecoin', 'futures_since': None},
}

# Binance symbol mapping
BINANCE_SPOT_SYMBOLS = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT', 'ADA': 'ADAUSDT',
    'AVAX': 'AVAXUSDT', 'DOT': 'DOTUSDT', 'UNI': 'UNIUSDT', 'AAVE': 'AAVEUSDT',
    'USDT': 'USDTUSD', 'USDC': 'USDCUSDT',
}

BINANCE_FUTURES_SYMBOLS = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT', 'ADA': 'ADAUSDT',
    'AVAX': 'AVAXUSDT', 'DOT': 'DOTUSDT', 'UNI': 'UNIUSDT', 'AAVE': 'AAVEUSDT',
}


# ==============================================================================
# EVENT STUDY PARAMETERS
# ==============================================================================

# Estimation window (for computing expected returns)
ESTIMATION_WINDOW_DAYS = 250  # calendar-day span; inclusive engine slice yields 251 daily returns
ESTIMATION_WINDOW_MIN = 120   # Minimum for newer assets

# Gap between estimation and event (excludes anticipation effects)
GAP_WINDOW_DAYS = 30

# Event window
EVENT_WINDOW_PRE = 5     # Days before event date in event window
EVENT_WINDOW_POST = 30   # Days after event date in event window

# Alternative windows for robustness
EVENT_WINDOWS = {
    'short': (-1, 5),     # Short-term immediate reaction
    'main': (-5, 30),     # Main analysis window
    'long': (-5, 60),     # Extended persistence
}

# Overlap handling
OVERLAP_THRESHOLD_DAYS = 30  # Events within this range flagged


# ==============================================================================
# STATISTICAL PARAMETERS
# ==============================================================================

# Significance levels
ALPHA_LEVELS = [0.01, 0.05, 0.10]

# Bootstrap
BOOTSTRAP_REPLICATIONS = 5000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_METHOD = 'percentile'  # Percentile CIs (see src/robustness.py); matches reported results

# Random seed for reproducibility
RANDOM_SEED = 42

# Kolari-Pynnonen adjustment
KP_ADJUSTMENT = True


# ==============================================================================
# LIQUIDITY METRICS
# ==============================================================================

# Rolling window for liquidity metric computation
LIQUIDITY_WINDOW = 21  # ~1 month

# Metrics to compute
LIQUIDITY_METRICS = [
    'amihud',       # Amihud (2002) illiquidity ratio
    'roll_spread',  # Roll (1984) spread estimator
    'cs_spread',    # Corwin-Schultz (2012) spread
    'kyle_lambda',  # Kyle's lambda (price impact)
    'funding_rate', # Perpetual futures funding rate
]


# ==============================================================================
# DATA COLLECTION
# ==============================================================================

# Date range for analysis
START_DATE = '2019-01-01'  # Post-ICO era
END_DATE = datetime.now().strftime('%Y-%m-%d')

# API rate limits
BINANCE_RATE_LIMIT = 0.1  # seconds between requests
COINGECKO_RATE_LIMIT = 1.2

# API keys (from environment)
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', '')


# ==============================================================================
# OUTPUT FORMATS
# ==============================================================================

# Table formats
TABLE_FLOAT_FORMAT = '.4f'
TABLE_PVALUE_FORMAT = '.3f'

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'pdf'  # For LaTeX compatibility
