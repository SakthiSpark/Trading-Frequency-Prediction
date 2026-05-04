"""
UTILS.PY - Utility Functions
Common helper functions used throughout the project
"""

import warnings
warnings.filterwarnings('ignore')

from config import COLORS, CONFIG
import matplotlib
matplotlib.use('TkAgg')

# ================================================================
# PRINTING UTILITIES
# ================================================================

def print_section(title, emoji='📌'):
    """Print a formatted section header"""
    print(f"\n{emoji} {title}")
    print("=" * 60)

def print_config():
    """Print configuration summary"""
    print_section("CONFIGURATION SUMMARY", "⚙️")
    print(f"   Ticker       : {CONFIG['TICKER']}")
    print(f"   Period       : {CONFIG['START_DATE']} → {CONFIG['END_DATE']}")
    print(f"   Prediction   : {CONFIG['FORWARD_BARS']} bars ahead")
    print(f"   Test Size    : {CONFIG['TEST_SIZE']*100:.0f}%")
    print(f"   Threshold    : {CONFIG['THRESHOLD']}")

def print_success(message, count=None):
    """Print success message"""
    if count is not None:
        print(f"   ✅ {message}: {count}")
    else:
        print(f"   ✅ {message}")

def print_info(message):
    """Print info message"""
    print(f"   ℹ️  {message}")

def print_warning(message):
    """Print warning message"""
    print(f"   ⚠️  {message}")

def print_error(message):
    """Print error message"""
    print(f"   ❌ {message}")

# ================================================================
# RESULT FORMATTING
# ================================================================

def print_model_results(accuracy, win_rate, profit):
    """Print model performance metrics"""
    print_section("MODEL PERFORMANCE", "🎯")
    print(f"   Accuracy    : {accuracy:.2f}%")
    print(f"   Win Rate    : {win_rate:.2f}%")
    print(f"   Total Profit: {profit:.4f}")

def print_prediction_results(ticker, direction, confidence, signal_strength):
    """Print prediction for a ticker"""
    emoji = "📈" if "UP" in direction else "📉"
    color = COLORS['green'] if "UP" in direction else COLORS['red']
    
    print_section(f"{ticker} PREDICTION", emoji)
    print(f"   Direction   : {direction}")
    print(f"   Confidence  : {confidence:.1f}%")
    print(f"   Signal      : {signal_strength}")

def print_top_results(results_dict, top_n=3):
    """Print top N trading signals"""
    print_section(f"🏆 TOP {top_n} SIGNALS", "🏆")
    
    # Sort by confidence
    top_results = sorted(
        results_dict.items(),
        key=lambda x: x[1]['confidence'],
        reverse=True
    )[:top_n]
    
    for rank, (pair, data) in enumerate(top_results, 1):
        print(f"\n   {rank}. {pair}")
        print(f"      Prediction  : {data['prediction']}")
        print(f"      Confidence  : {data['confidence']:.1f}%")
        print(f"      Accuracy    : {data['accuracy']:.2f}%")
        print(f"      Signal      : {data['signal_strength']}")

# ================================================================
# FEATURE UTILITIES
# ================================================================

def get_feature_groups():
    """Return grouped feature categories"""
    return {
        'price': ['Return', 'MA5', 'MA20', 'MA50', 'MA_Cross'],
        'volatility': ['Volatility', 'Vol_5', 'Vol_20', 'Vol_Regime', 'Volatility_Ratio'],
        'momentum': ['Momentum', 'RSI', 'RSI_Slope', 'MACD_Hist'],
        'volume': ['Vol_Ratio', 'Vol_Trend', 'PV_Trend'],
        'calendar': ['Is_NFP_Week', 'Is_Fed_Week', 'Is_ECB_Week', 'DayOfWeek'],
        'candle': ['Body_Size', 'Upper_Wick', 'Lower_Wick', 'Is_Bullish'],
    }

def get_numeric_features(df):
    """Get all numeric columns from dataframe"""
    return df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# ================================================================
# COLOR UTILITIES
# ================================================================

def get_signal_color(value, is_positive=True):
    """Get color based on signal value"""
    if is_positive:
        return COLORS['green'] if value > 0.5 else COLORS['red']
    else:
        return COLORS['green'] if value > 0 else COLORS['red']

def get_direction_emoji(prediction):
    """Get emoji for direction"""
    return "📈" if prediction == 1 else "📉"

# ================================================================
# VALIDATION
# ================================================================

def validate_config():
    """Validate configuration parameters"""
    assert CONFIG['TEST_SIZE'] > 0 and CONFIG['TEST_SIZE'] < 1, "TEST_SIZE must be between 0 and 1"
    assert CONFIG['FORWARD_BARS'] > 0, "FORWARD_BARS must be positive"
    assert CONFIG['THRESHOLD'] > 0, "THRESHOLD must be positive"
    print_success("Configuration validated")

def validate_dataframe(df, required_columns):
    """Validate dataframe has required columns"""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    if df.isnull().sum().sum() > 0:
        print_warning(f"Found {df.isnull().sum().sum()} missing values - will be dropped")
    
    print_success(f"DataFrame validated ({len(df)} rows, {len(df.columns)} columns)")

# ================================================================
# TIPS & HELP
# ================================================================

def print_tips():
    """Print helpful tips"""
    print_section("💡 TIPS", "💡")
    print("""
   Try other tickers by changing CONFIG['TICKER']:
   
   Forex Pairs  : 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'NZDUSD=X'
   Tech Stocks  : 'AAPL', 'TSLA', 'GOOGL', 'MSFT', 'NVDA'
   India Stocks : '^NSEI', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS'
   Indices      : '^GSPC' (S&P 500), '^DJI' (Dow Jones), '^BSESN' (BSE)
    """)

def print_libraries_info():
    """Print library versions"""
    import numpy as np
    import pandas as pd
    
    print_section("LIBRARIES LOADED", "✅")
    print(f"   NumPy       : {np.__version__}")
    print(f"   Pandas      : {pd.__version__}")
    print(f"   Matplotlib  : {matplotlib.__version__}")
    
    try:
        import yfinance as yf
        print(f"   yfinance    : {yf.__version__}")
    except:
        pass
    
    try:
        from sklearn import __version__ as sklearn_version
        print(f"   scikit-learn: {sklearn_version}")
    except:
        pass
    
    try:
        import xgboost as xgb
        print(f"   XGBoost     : {xgb.__version__}")
    except:
        pass
