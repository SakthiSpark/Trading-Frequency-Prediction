"""
CONFIG.PY - Configuration & Constants
All hardcoded values and settings in one place
"""

# ================================================================
# MAIN CONFIGURATION
# ================================================================
CONFIG = {
    'TICKER'       : 'EURUSD=X',
    'START_DATE'   : '2016-01-01',
    'END_DATE'     : '2025-12-31',
    'FORWARD_BARS' : 3,
    'THRESHOLD'    : 0.005,
    'TEST_SIZE'    : 0.2,
    'TIMESTEPS'    : 20
}

# Available forex pairs
PAIRS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'NZDUSD=X']

# Test tickers for backtesting
TEST_TICKERS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', '^NSEI']

# ================================================================
# VISUALIZATION COLORS (GitHub Dark Theme)
# ================================================================
COLORS = {
    'bg'      : '#0d1117',
    'panel'   : '#161b22',
    'green'   : '#39d353',
    'red'     : '#f85149',
    'yellow'  : '#d29922',
    'blue'    : '#58a6ff',
    'purple'  : '#bc8cff',
    'text'    : '#e6edf3',
    'subtext' : '#8b949e',
}

# ================================================================
# MODEL HYPERPARAMETERS
# ================================================================
XGBOOST_PARAMS = {
    'n_estimators'    : 1500,
    'max_depth'       : 3,
    'learning_rate'   : 0.005,
    'subsample'       : 0.6,
    'colsample_bytree': 0.6,
    'min_child_weight': 8,
    'gamma'           : 0.2,
    'reg_alpha'       : 0.5,
    'reg_lambda'      : 2.0,
    
    'eval_metric'     : 'logloss'
}

# ================================================================
# TECHNICAL INDICATOR PERIODS
# ================================================================
PERIODS = {
    'MA_SHORT': 5,
    'MA_MEDIUM': 20,
    'MA_LONG': 50,
    'RSI': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'BOLLINGER': 20,
    'ATR': 14,
    'scale_pos_weight':3, 
    'VOLUME_SHORT': 5,
    'VOLUME_LONG': 20,
}

# ================================================================
# BACKTESTING PARAMETERS
# ================================================================
BACKTEST = {
    'MIN_CONFIDENCE': 0.6,
    'TRADING_COST': 0.0001,  # Spread/commission
    'POSITION_SIZE': 1.0,
}

# ================================================================
# ECONOMIC CALENDAR EVENTS
# ================================================================
# These are the major events that impact currency pairs
# You can add more events here
ECONOMIC_EVENTS = {
    'NFP': {
        'name': 'Non-Farm Payrolls',
        'day_of_week': 4,  # Friday
        'day_of_month_max': 7,  # First week
        'impact': 'CRITICAL',
    },
    'FED': {
        'name': 'Federal Reserve Decision',
        'months': [1, 3, 5, 7, 9, 11],  # 8 times per year
        'impact': 'CRITICAL',
    },
    'ECB': {
        'name': 'European Central Bank Decision',
        'impact': 'CRITICAL',
    },
}

# ================================================================
# FEATURE SELECTION
# ================================================================
# Which features to exclude (if needed)
FEATURES_TO_EXCLUDE = [
    'Close',    # Target variable
    'High',     # OHLC - not needed after feature engineering
    'Low',
    'Open',
    'Volume',   # Not useful for forex
]

# ================================================================
# LOGGING & OUTPUT
# ================================================================
VERBOSE = True
SAVE_DASHBOARD = True
DASHBOARD_FILENAME = 'prediction_dashboard.png'
DASHBOARD_DPI = 150

# ================================================================
# PATHS
# ================================================================
OUTPUT_DIR = './'
MODELS_DIR = './models/'
LOGS_DIR = './logs/'
