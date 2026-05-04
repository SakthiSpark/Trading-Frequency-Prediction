"""
FEATURE_ENGINEERING.PY - Technical Indicators & Features
Builds comprehensive feature set for predictions
"""

import pandas as pd
import numpy as np
from config import PERIODS
from utils import print_section, print_success

# ================================================================
# RSI CALCULATION
# ================================================================

def compute_rsi(series, period=14):
    """
    Calculate Relative Strength Index (RSI)
    
    RSI measures momentum - ranges from 0 to 100
    - RSI < 30: Oversold (potential BUY)
    - RSI > 70: Overbought (potential SELL)
    
    Args:
        series (pd.Series): Price series
        period (int): Lookback period (default 14)
    
    Returns:
        pd.Series: RSI values
    """
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

# ================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# ================================================================

def add_technical_features(df):
    """
    Add all technical features to dataframe
    
    This is the main feature engineering function that creates:
    - Trend indicators (MA, slopes)
    - Momentum indicators (RSI, MACD)
    - Volatility indicators
    - Volume indicators
    - Candle patterns
    - Mean reversion indicators
    
    Args:
        df (pd.DataFrame): OHLCV dataframe
    
    Returns:
        pd.DataFrame: Dataframe with all features
    """
    print_section("🔧 BUILDING TECHNICAL FEATURES", "🔧")
    
    df = df.copy()
    
    # ── TREND INDICATORS ────────────────────────────────────────
    print("   Adding trend indicators...")
    df['Return']     = df['Close'].pct_change()
    df['MA5']        = df['Close'].rolling(PERIODS['MA_SHORT']).mean()
    df['MA20']       = df['Close'].rolling(PERIODS['MA_MEDIUM']).mean()
    df['MA50']       = df['Close'].rolling(PERIODS['MA_LONG']).mean()
    df['MA_Cross']   = df['MA5'] - df['MA20']
    
    # MA slopes (trend acceleration)
    df['MA5_slope']  = df['MA5'].diff(3) / 3
    df['MA20_slope'] = df['MA20'].diff(5) / 5
    
    # ── VOLATILITY INDICATORS ──────────────────────────────────
    print("   Adding volatility indicators...")
    df['Volatility'] = df['Return'].rolling(10).std()
    df['Vol_5']      = df['Return'].rolling(5).std()
    df['Vol_20']     = df['Return'].rolling(20).std()
    df['Vol_Regime'] = df['Vol_5'] / (df['Vol_20'] + 1e-10)
    df['Volatility_Ratio'] = df['Volatility'] / df['Volatility'].rolling(20).mean()
    
    # Volatility squeeze (low vol → big move coming)
    df['Vol_Squeeze'] = (df['Volatility'] < df['Volatility'].rolling(30).quantile(0.25)).astype(int)
    
    # ── MOMENTUM INDICATORS ────────────────────────────────────
    print("   Adding momentum indicators...")
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df['Return_MA5'] = df['Return'].rolling(5).mean()
    
    # RSI and zones
    df['RSI'] = compute_rsi(df['Close'], PERIODS['RSI'])
    df['RSI_OS'] = (df['RSI'] < 35).astype(int)      # Oversold
    df['RSI_OB'] = (df['RSI'] > 65).astype(int)      # Overbought
    df['RSI_Slope'] = df['RSI'].diff(3)
    
    # MACD (momentum trend indicator)
    ema12 = df['Close'].ewm(span=PERIODS['MACD_FAST']).mean()
    ema26 = df['Close'].ewm(span=PERIODS['MACD_SLOW']).mean()
    macd = ema12 - ema26
    df['MACD_Hist'] = macd - macd.ewm(span=PERIODS['MACD_SIGNAL']).mean()
    
    # ── VOLATILITY BANDS ────────────────────────────────────────
    print("   Adding volatility bands...")
    roll_mean = df['Close'].rolling(PERIODS['BOLLINGER']).mean()
    roll_std = df['Close'].rolling(PERIODS['BOLLINGER']).std()
    df['BB_Pos'] = (df['Close'] - roll_mean) / (2 * roll_std + 1e-10)
    df['BB_High'] = roll_mean + (2 * roll_std)
    df['BB_Low'] = roll_mean - (2 * roll_std)
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / roll_mean
    
    # ── VOLUME FEATURES ────────────────────────────────────────
    print("   Adding volume features...")
    df['Vol_Ratio'] = df['Volume'] / (df['Volume'].rolling(PERIODS['VOLUME_LONG']).mean() + 1e-10)
    df['Vol_Trend'] = df['Volume'].rolling(PERIODS['VOLUME_SHORT']).mean() / \
                      (df['Volume'].rolling(PERIODS['VOLUME_LONG']).mean() + 1e-10)
    df['PV_Trend'] = df['Return'] * df['Vol_Ratio']
    
    # ── CANDLE STRUCTURE ────────────────────────────────────────
    print("   Adding candle patterns...")
    hl_range = df['High'] - df['Low'] + 1e-10
    df['Body_Size'] = abs(df['Close'] - df['Open']) / hl_range
    df['Upper_Wick'] = (df['High'] - np.maximum(df['Close'], df['Open'])) / hl_range
    df['Lower_Wick'] = (np.minimum(df['Close'], df['Open']) - df['Low']) / hl_range
    df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)
    df['Range'] = (df['High'] - df['Low']) / df['Close']
    df['ATR'] = df['High'] - df['Low']  # Average True Range
    
    # ── MEAN REVERSION ──────────────────────────────────────────
    print("   Adding mean reversion indicators...")
    df['Z_Score'] = (df['Close'] - df['Close'].rolling(20).mean()) / \
                    (df['Close'].rolling(20).std() + 1e-10)
    df['Price_Position'] = (df['Close'] - df['Close'].rolling(20).min()) / \
                          (df['Close'].rolling(20).max() - df['Close'].rolling(20).min() + 1e-10)
    df['MA50_dist'] = (df['Close'] - df['MA50']) / (df['MA50'] + 1e-10)
    
    # ── UP/DOWN STREAKS ────────────────────────────────────────
    print("   Adding streak indicators...")
    df['Up_Flag'] = df['Return'].gt(0).astype(int)
    df['Up_Streak'] = df['Up_Flag'].groupby(
        (df['Up_Flag'] != df['Up_Flag'].shift()).cumsum()
    ).cumcount() + 1
    df['Up_Streak'] = df['Up_Streak'] * df['Up_Flag']
    
    # ── LAGGED RETURNS ─────────────────────────────────────────
    print("   Adding lagged returns...")
    for lag in [1, 2, 3, 5]:
        df[f'Return_lag{lag}'] = df['Return'].shift(lag)
    
    # ── ADVANCED MOMENTUM FEATURES ─────────────────────────────
    print("   Adding advanced momentum features...")
    
    # High-frequency momentum
    df['Return_2'] = df['Close'].pct_change(2)
    df['Return_3'] = df['Close'].pct_change(3)
    df['Return_5'] = df['Close'].pct_change(5)
    
    # Accumulated momentum
    df['Mom_Acc'] = df['Return'].rolling(5).sum()
    df['Mom_Acc_20'] = df['Return'].rolling(20).sum()
    
    # ── PRICE LEVELS ───────────────────────────────────────────
    print("   Adding price level features...")
    df['Support'] = df['Close'].rolling(20).min()
    df['Resistance'] = df['Close'].rolling(20).max()
    df['Distance_Support'] = (df['Close'] - df['Support']) / df['Support']
    df['Distance_Resistance'] = (df['Resistance'] - df['Close']) / df['Close']
    
    print_success(f"Created {len(df.columns) - 5} features")  # -5 for OHLCV
    
    return df

# ================================================================
# FEATURE SELECTION & ANALYSIS
# ================================================================

def get_feature_list(df):
    """Get list of all features (excluding OHLCV)"""
    exclude = ['Close', 'High', 'Low', 'Open', 'Volume']
    features = [col for col in df.columns if col not in exclude]
    return features

def drop_nan_rows(df):
    """Drop rows with NaN values (created by rolling indicators)"""
    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)
    print(f"   Dropped {dropped} rows with NaN values")
    return df

def select_important_features(df, importance_df, top_n=50):
    """
    Select top N features by importance
    
    Args:
        df (pd.DataFrame): Feature dataframe
        importance_df (pd.Series): Feature importance from model
        top_n (int): Number of features to keep
    
    Returns:
        pd.DataFrame: Dataframe with top features
    """
    features = importance_df.head(top_n).index.tolist()
    return df[features]

# ── ADX & MOMENTUM CONFIRM ─────────────────────────────────
    high_low   = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close  = (df['Low']  - df['Close'].shift()).abs()
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ADX']  = tr.rolling(14).mean()

    df['Mom_Confirm'] = (
        (df['Return'] > 0) & (df['MACD_Hist'] > 0)
    ).astype(int)

    print_success(f"Created {len(df.columns) - 5} features")  # ← this stays last
    return df