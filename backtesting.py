"""
BACKTESTING.PY - Backtest Strategy
Test trading strategy on historical data
"""

import pandas as pd
import numpy as np
from config import CONFIG, BACKTEST
from data_handler import get_ohlcv_data, add_economic_calendar
from feature_engineering import add_technical_features, drop_nan_rows
from model_training import train_full_pipeline
from prediction import make_predictions
from utils import print_section, print_success, print_info

# ================================================================
# BACKTEST EXECUTION
# ================================================================

def run_backtest(df, predictions, probabilities, actuals=None, forward_bars=None):
    """
    Run backtest on predictions

    Args:
        df (pd.DataFrame): OHLCV dataframe
        predictions (np.array): Model predictions
        probabilities (np.array): Prediction probabilities
        actuals (np.array): Actual labels (optional)
        forward_bars (int): Forward period. Uses CONFIG if None

    Returns:
        dict: Backtest results
    """
    if forward_bars is None:
        forward_bars = CONFIG['FORWARD_BARS']

    print_section("📊 RUNNING BACKTEST", "📊")

    trades = []

    for i in range(len(predictions) - forward_bars):
        # Skip low confidence predictions
        prob = probabilities[i]
        confidence = abs(prob - 0.5) * 2

        if confidence < 0.70:
            continue
        try:
            ma5  = df['MA5'].iloc[i]       # ← 4 spaces indent
            ma20 = df['MA20'].iloc[i]
            trend_strength = abs(ma5 - ma20) / ma20
            if trend_strength < 0.001:
                continue
        except:
            continue
          
        # Get entry and exit prices
        try:
            entry_price = df['Close'].iloc[i]
            exit_price = df['Close'].iloc[i + forward_bars]
        except IndexError:
            continue

        # Calculate profit/loss
        if predictions[i] == 1:  # UP prediction
            profit = exit_price - entry_price
            trade_type = "LONG"
        else:  # DOWN prediction
            profit = entry_price - exit_price
            trade_type = "SHORT"

        # Subtract trading costs
        profit -= BACKTEST['TRADING_COST']

        trades.append({
            'index': i,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': profit,
            'type': trade_type,
            'confidence': confidence * 100,
            'prediction': predictions[i]
        })

    if not trades:
        print("   No trades generated (low confidence)")
        return None

    trades_df = pd.DataFrame(trades)

    # Calculate metrics
    total_profit = trades_df['profit'].sum()
    winning_trades = (trades_df['profit'] > 0).sum()
    total_trades = len(trades_df)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['profit'] < 0]['profit'].mean() if (total_trades - winning_trades) > 0 else 0

    # Risk-reward ratio
    if abs(avg_loss) > 0:
        rr_ratio = abs(avg_win / avg_loss)
    else:
        rr_ratio = 0

    print_success(f"Generated {total_trades} trades")
    print_success(f"Win Rate: {win_rate:.1f}%")
    print_success(f"Total Profit: {total_profit:.6f}")

    return {
        'trades': trades_df,
        'total_profit': total_profit,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades))) if (total_trades - winning_trades) > 0 else np.inf,
        'rr_ratio': rr_ratio
    }


# ================================================================
# RISK MANAGED BACKTEST
# ================================================================

def backtest_with_risk_management(predictions, actuals, prices, atr_values, probabilities):
    """Backtest with stop loss, position sizing and risk management"""

    if actuals is None:
        actuals = prices.copy()

    trades = []
    account_balance = 10000
    risk_per_trade = 0.02

    for i in range(len(predictions)):
        if i < 2:
            continue

        confidence = max(probabilities[i], 1 - probabilities[i])
        if confidence < 0.70:
            continue

        pred = predictions[i]
        actual = actuals[i]
        current_price = prices[i]
        atr = atr_values[i]

        if pred == 0:  # DOWN
            entry = current_price
            stop_loss = entry + (atr * 2)
            profit_target = entry - (atr * 3)
            risk = stop_loss - entry
            reward = entry - profit_target
        else:  # UP
            entry = current_price
            stop_loss = entry - (atr * 2)
            profit_target = entry + (atr * 3)
            risk = entry - stop_loss
            reward = profit_target - entry

        if reward <= 0 or risk <= 0:
            continue
        if (reward / risk) < 1.5:
            continue

        position_size = (10000 * risk_per_trade) / risk

        if pred == 0:
            won = (actual == 0) 
        else:
            won = (actual == 1)

        if won:
            profit = reward * position_size
            account_balance += profit
            trades.append({'profit': profit, 'win': 1})
        else:
            loss = risk * position_size
            account_balance -= loss
            trades.append({'profit': -loss, 'win': 0})

    return trades, account_balance


# ================================================================
# MULTI-TICKER BACKTEST
# ================================================================

def run_model_for_ticker(ticker, verbose=True):
    """
    Train model and run backtest for a specific ticker

    Args:
        ticker (str): Ticker symbol
        verbose (bool): Print details

    Returns:
        dict: Results summary
    """
    try:
        if verbose:
            print_section(f"TESTING {ticker}", "📈")

        # Download and prepare data
        df = get_ohlcv_data(ticker)
        df = add_economic_calendar(df)
        df = add_technical_features(df)

        # Train model
        pipeline = train_full_pipeline(df)

        # Get predictions
        X_test = pipeline['X_test']
        y_test = pipeline['y_test']
        split_point = pipeline['split_point']

        predictions = pipeline['metrics']['predictions']
        probabilities = pipeline['metrics']['probabilities']
        accuracy = pipeline['metrics']['accuracy']

        # Run backtest
        
        test_df = df.iloc[split_point:].copy()
        backtest_result = run_backtest(test_df, predictions, probabilities)

        if backtest_result is None:
            return None
        

        return {
            'ticker': ticker,
            'accuracy': accuracy * 100,
            'win_rate': backtest_result['win_rate'],
            'profit': backtest_result['total_profit'],
            'trades': backtest_result['total_trades'],
            'backtest': backtest_result
        }

    except Exception as e:
        if verbose:
            print(f"   Error: {e}")
        return None
    


def backtest_multiple_tickers(tickers, verbose=False):
    """
    Backtest multiple tickers

    Args:
        tickers (list): List of ticker symbols
        verbose (bool): Print details

    Returns:
        pd.DataFrame: Results comparison
    """
    print_section(f"BACKTESTING {len(tickers)} TICKERS", "🧪")

    results = []

    for ticker in tickers:
        result = run_model_for_ticker(ticker, verbose=verbose)
        if result:
            results.append(result)

    if not results:
        print("   No results generated")
        return None

    results_df = pd.DataFrame([{
        'Ticker': r['ticker'],
        'Accuracy %': f"{r['accuracy']:.2f}%",
        'Win Rate %': f"{r['win_rate']:.1f}%",
        'Total Profit': f"{r['profit']:.6f}",
        'Trades': r['trades']
    } for r in results])

    print("\n" + "="*60)
    print(results_df.to_string(index=False))
    print("="*60)

    return results_df, results


# ================================================================
# PERFORMANCE ANALYSIS
# ================================================================

def calculate_backtest_metrics(trades_df):
    """Calculate comprehensive backtest metrics"""
    if trades_df.empty:
        return None

    profits = trades_df['profit'].values
    winning_trades = (profits > 0).sum()
    losing_trades = (profits < 0).sum()
    total_trades = len(trades_df)

    metrics = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        'total_profit': profits.sum(),
        'avg_profit': profits.mean(),
        'std_profit': profits.std(),
        'max_profit': profits.max(),
        'max_loss': profits.min(),
        'profit_factor': _calculate_profit_factor(profits),
        'sharpe_ratio': _calculate_sharpe(profits),
        'max_drawdown': _calculate_max_drawdown(profits.cumsum()),
    }

    return metrics


def _calculate_profit_factor(profits):
    """Calculate profit factor (gross profit / gross loss)"""
    gross_profit = profits[profits > 0].sum()
    gross_loss = abs(profits[profits < 0].sum())

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0

    return gross_profit / gross_loss


def _calculate_sharpe(profits, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    if len(profits) < 2:
        return 0

    returns = pd.Series(profits).pct_change().dropna()

    if len(returns) == 0 or returns.std() == 0:
        return 0

    return (returns.mean() - risk_free_rate / 252) / returns.std() * np.sqrt(252)


def _calculate_max_drawdown(cumulative_profit):
    """Calculate maximum drawdown"""
    running_max = pd.Series(cumulative_profit).expanding().max()
    drawdown = (cumulative_profit - running_max) / running_max
    return drawdown.min() * 100


def print_backtest_summary(backtest_result):
    """Print formatted backtest summary"""
    if backtest_result is None:
        return

    print_section("BACKTEST SUMMARY", "📊")
    print(f"   Total Trades     : {backtest_result['total_trades']}")
    print(f"   Win Rate         : {backtest_result['win_rate']:.1f}%")
    print(f"   Total Profit     : {backtest_result['total_profit']:.6f}")
    print(f"   Avg Win          : {backtest_result['avg_win']:.6f}")
    print(f"   Avg Loss         : {backtest_result['avg_loss']:.6f}")
    print(f"   R:R Ratio        : {backtest_result['rr_ratio']:.2f}")


# ================================================================
# TRADE ANALYSIS
# ================================================================

def analyze_trades(trades_df):
    """Detailed analysis of trades"""
    if trades_df.empty:
        return None

    long_trades = trades_df[trades_df['type'] == 'LONG']
    short_trades = trades_df[trades_df['type'] == 'SHORT']

    return {
        'long_trades': len(long_trades),
        'long_win_rate': (long_trades['profit'] > 0).mean() * 100 if len(long_trades) > 0 else 0,
        'long_avg_profit': long_trades['profit'].mean() if len(long_trades) > 0 else 0,
        'short_trades': len(short_trades),
        'short_win_rate': (short_trades['profit'] > 0).mean() * 100 if len(short_trades) > 0 else 0,
        'short_avg_profit': short_trades['profit'].mean() if len(short_trades) > 0 else 0,
    }