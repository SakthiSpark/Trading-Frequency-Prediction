"""
DATA_HANDLER.PY - Data Download & Processing
Handles all data operations from yfinance
"""

import pandas as pd
import numpy as np
import yfinance as yf
from config import CONFIG
from utils import print_section, print_success, print_error

# ================================================================
# DATA DOWNLOAD
# ================================================================

def download_data(ticker, start_date, end_date):
    """
    Download OHLCV data from yfinance
    
    Args:
        ticker (str): Stock/forex ticker (e.g., 'EURUSD=X', 'AAPL')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pd.DataFrame: OHLCV data
    """
    print_section(f"📥 DOWNLOADING DATA", "📥")
    print(f"   Ticker: {ticker}")
    print(f"   Period: {start_date} → {end_date}")
    
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        
        print_success(f"Downloaded {len(df)} bars")
        return df
        
    except Exception as e:
        print_error(f"Failed to download data: {e}")
        raise

# ================================================================
# DATA CLEANING & PREPROCESSING
# ================================================================

def clean_data(df):
    """
    Clean and prepare downloaded data
    
    Args:
        df (pd.DataFrame): Raw OHLCV dataframe
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print_section("🧹 CLEANING DATA", "🧹")
    
    df = df.copy()
    initial_rows = len(df)
    
    # Step 1: Flatten MultiIndex columns (yfinance sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        print("   Flattening MultiIndex columns...")
        df.columns = [col[0].strip() for col in df.columns]
    
    # Step 2: Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Step 3: Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Step 4: Keep only OHLCV columns
    required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    available_cols = [col for col in required_cols if col in df.columns]
    
    if not available_cols:
        raise ValueError(f"No OHLC columns found. Available: {df.columns.tolist()}")
    
    df = df[available_cols].copy()
    print_success(f"Selected {len(available_cols)} OHLCV columns")
    
    # Step 5: Sort by index
    df.sort_index(inplace=True)
    
    # Step 6: Remove NaN values
    df = df[df['Close'].notna()].copy()
    removed = initial_rows - len(df)
    
    if removed > 0:
        print(f"   Removed {removed} rows with missing values")
    
    print_success(f"Final dataset: {len(df)} rows")
    return df

def get_ohlcv_data(ticker=None, start_date=None, end_date=None):
    """
    Convenience function to download and clean data in one step
    
    Args:
        ticker (str): Ticker symbol. Uses CONFIG if None
        start_date (str): Start date. Uses CONFIG if None
        end_date (str): End date. Uses CONFIG if None
    
    Returns:
        pd.DataFrame: Cleaned OHLCV data
    """
    if ticker is None:
        ticker = CONFIG['TICKER']
    if start_date is None:
        start_date = CONFIG['START_DATE']
    if end_date is None:
        end_date = CONFIG['END_DATE']
    
    df = download_data(ticker, start_date, end_date)
    df = clean_data(df)
    
    return df

# ================================================================
# ECONOMIC CALENDAR FLAGS
# ================================================================

def add_economic_calendar(df):
    """
    Add economic calendar event flags to dataframe
    
    These are major events that impact currency pairs:
    - NFP (Non-Farm Payrolls): First Friday of each month - affects USD
    - Fed Meetings: 8 times per year - affects USD
    - ECB Meetings: Regular meetings - affects EUR
    - Quarter Rebalancing: End/start of quarters - affects all pairs
    
    Args:
        df (pd.DataFrame): OHLCV dataframe with datetime index
    
    Returns:
        pd.DataFrame: Dataframe with calendar flags
    """
    print_section("📅 ADDING ECONOMIC CALENDAR", "📅")
    
    df = df.copy()
    
    # Basic calendar features
    df['Is_MonthEnd']   = (df.index.is_month_end).astype(int)
    df['Is_MonthStart'] = (df.index.is_month_start).astype(int)
    df['Is_QuarterEnd'] = (df.index.is_quarter_end).astype(int)
    df['DayOfWeek']     = df.index.dayofweek
    df['WeekOfMonth']   = (df.index.day - 1) // 7 + 1
    
    # NFP: First Friday of each month (highest impact for USD)
    df['Is_NFP_Week'] = (
        (df.index.dayofweek == 4) & (df.index.day <= 7)
    ).astype(int)
    
    # Fed Meetings: 8 times per year, roughly every 6 weeks
    # Usually Wednesday in 3rd week of Jan, Mar, May, Jul, Sep, Nov
    df['Is_Fed_Week'] = (
        (df.index.month.isin([1, 3, 5, 7, 9, 11])) &
        (df.index.dayofweek == 2) & 
        (df['WeekOfMonth'] == 3)
    ).astype(int)
    
    # ECB Meetings: Usually Thursday in 1st or 2nd week of month
    df['Is_ECB_Week'] = (
        (df.index.dayofweek == 3) & 
        (df['WeekOfMonth'].isin([1, 2]))
    ).astype(int)
    
    # Quarter Rebalancing: Large funds rebalance at quarter boundaries
    df['Is_Quarter_Rebalance'] = (
        (df.index.is_quarter_end) | (df.index.is_quarter_start)
    ).astype(int)
    
    # Print summary
    print_success(f"NFP weeks flagged: {df['Is_NFP_Week'].sum()}")
    print_success(f"Fed weeks flagged: {df['Is_Fed_Week'].sum()}")
    print_success(f"ECB weeks flagged: {df['Is_ECB_Week'].sum()}")
    print_success(f"Quarter rebalance flagged: {df['Is_Quarter_Rebalance'].sum()}")
    
    return df

def add_date_features(df):
    """
    Add additional date-based features
    
    Args:
        df (pd.DataFrame): Dataframe with datetime index
    
    Returns:
        pd.DataFrame: Dataframe with date features
    """
    df = df.copy()
    
    df['Hour']       = df.index.hour
    df['Day']        = df.index.day
    df['Month']      = df.index.month
    df['Quarter']    = df.index.quarter
    df['Year']       = df.index.year
    df['DayOfYear']  = df.index.dayofyear
    
    # Weekend flag
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    return df
