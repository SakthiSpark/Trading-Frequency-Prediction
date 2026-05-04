"""
MAIN.PY - Main Orchestration File
Complete workflow for forex/stock prediction

Run this file to execute the full pipeline:
    python main.py
"""

import warnings
import numpy as np 
warnings.filterwarnings('ignore')

# ================================================================
# IMPORTS
# ================================================================
from config import CONFIG, TEST_TICKERS
from utils import (
    print_libraries_info, print_config, print_success,
    print_section, print_top_results, print_tips, validate_config
)
from data_handler import get_ohlcv_data, add_economic_calendar
from feature_engineering import add_technical_features
from model_training import train_full_pipeline
from prediction import make_predictions, predict_next_bar, generate_signal
from backtesting import run_backtest, run_model_for_ticker
from visualization import create_dashboard, save_dashboard, show_dashboard

# ================================================================
# MAIN PIPELINE
# ================================================================

def main():
    """
    Main execution function - runs complete pipeline
    """
    
    # ── SETUP ────────────────────────────────────────────────────────
    print_libraries_info()
    validate_config()
    print_config()
    
    # ── DATA DOWNLOAD & PREPARATION ─────────────────────────────────
    print_section("STEP 1: DATA DOWNLOAD & PREPARATION", "1️⃣")
    
    df = get_ohlcv_data()
    df = add_economic_calendar(df)
    df = add_technical_features(df)
    
    # ── MODEL TRAINING ───────────────────────────────────────────────
    print_section("STEP 2: MODEL TRAINING", "2️⃣")
    
    pipeline = train_full_pipeline(df)
    
    # ── PREDICTIONS ──────────────────────────────────────────────────
    print_section("STEP 3: MAKING PREDICTIONS", "3️⃣")
    
    X_test = pipeline['X_test']
    y_test = pipeline['y_test']
    model = pipeline['model']
    scaler = pipeline['scaler']
    pred_dict = make_predictions(model, scaler, X_test)
    predictions = pred_dict['predictions']
    probabilities = pred_dict['up_probability']

    # ── CONFIDENCE FILTER ────────────────────────────────────────
    min_confidence = 0.65
    confidence_mask = np.maximum(probabilities, 1 - probabilities) > min_confidence
    xgb_preds_filtered = predictions.copy()
    xgb_preds_filtered[~confidence_mask] = 0.5

    print(f"\n🎯 CONFIDENCE ANALYSIS")
    print(f"{'='*55}")
    print(f"Total predictions      : {len(predictions)}")
    print(f"High confidence (>65%) : {confidence_mask.sum()}")
    print(f"Low confidence (<65%)  : {(~confidence_mask).sum()}")
    print(f"Trade rate             : {confidence_mask.sum()/len(predictions)*100:.1f}%")
    
    # Make predictions on test set
    pred_dict = make_predictions(model, scaler, X_test)
    predictions = pred_dict['predictions']
    probabilities = pred_dict['up_probability']
    confidence = pred_dict['confidence']
    
    # Get latest prediction for next bar
    latest_features = X_test.iloc[-1]
    latest_pred = predict_next_bar(model, scaler, latest_features)
    
    print_section("LATEST SIGNAL", "🎯")
    print(f"   Direction   : {latest_pred['direction']}")
    print(f"   Confidence  : {latest_pred['confidence']:.1f}%")
    print(f"   Probability : {latest_pred['up_probability']:.4f}")
    
    signal = generate_signal(latest_pred)
    print(f"   Signal      : {signal['signal']}")
    print(f"   Strength    : {signal['strength']}")
    
    # ── BACKTESTING ──────────────────────────────────────────────────
    print_section("STEP 4: BACKTESTING", "4️⃣")
    
    split_point = pipeline['split_point']
    test_df = df.iloc[split_point:].copy()
    backtest_result = run_backtest(test_df, predictions, probabilities, actuals=y_test.values)
    
    if backtest_result:
        print(f"\n   Total Trades   : {backtest_result['total_trades']}")
        print(f"   Win Rate       : {backtest_result['win_rate']:.1f}%")
        print(f"   Total Profit   : {backtest_result['total_profit']:.6f}")
        print(f"   Avg Win        : {backtest_result['avg_win']:.6f}")
        print(f"   Avg Loss       : {backtest_result['avg_loss']:.6f}")

        # ── RISK MANAGED BACKTEST ────────────────────────────────────
    from backtesting import backtest_with_risk_management

    trades, final_balance = backtest_with_risk_management(
        predictions,
        y_test.values,
        test_df['Close'].values,
        test_df['ATR'].values,
        probabilities
    )

    if trades:
        wins = sum(1 for t in trades if t['win'] == 1)
        total_profit = sum(t['profit'] for t in trades)
        print(f"\n   📊 RISK MANAGED BACKTEST")
        print(f"   {'─'*37}")
        print(f"   Total Trades  : {len(trades)}")
        print(f"   Win Rate      : {wins/len(trades)*100:.1f}%")
        print(f"   Total Profit  : ${total_profit:.2f}")
        print(f"   Final Balance : ${final_balance:.2f}")
        print(f"   Return        : {(final_balance-10000)/10000*100:.1f}%")
    else:
        print("   No trades passed risk filters")
    
    # ── VISUALIZATION ────────────────────────────────────────────────
    print_section("STEP 5: VISUALIZATION", "5️⃣")
    
    fig = create_dashboard(df, pipeline, backtest_result)
    saved = save_dashboard(fig)
    
    # ── MULTI-TICKER ANALYSIS ────────────────────────────────────────
    print_section("STEP 6: MULTI-TICKER ANALYSIS", "6️⃣")
    
    print(f"\n   Testing {len(TEST_TICKERS)} tickers...")
    results_dict = {}
    
    for ticker in TEST_TICKERS:
        try:
            result = run_model_for_ticker(ticker, verbose=False)
            if result:
                results_dict[ticker] = {
                    'prediction': f"{'📈 UP' if result['accuracy'] > 50 else '📉 DOWN'}",
                    'confidence': result['accuracy'],
                    'accuracy': result['accuracy'],
                    'signal_strength': '🟢 STRONG' if result['accuracy'] > 60 else '🟡 MODERATE'
                }
        except Exception as e:
            print(f"   Error with {ticker}: {e}")
    
    if results_dict:
        print_top_results(results_dict, top_n=3)
    
    # ── SUMMARY ──────────────────────────────────────────────────────
    print_section("EXECUTION COMPLETE", "✅")
    
    print(f"""
   📊 SUMMARY
   ─────────────────────────────────────
   ✅ Data downloaded    : {len(df)} bars
   ✅ Model trained      : {pipeline['metrics']['accuracy']*100:.2f}% accuracy
   ✅ Predictions made   : {len(predictions)} samples
   ✅ Backtest complete  : {backtest_result['total_trades'] if backtest_result else 0} trades
   ✅ Dashboard created  : {'Yes' if saved else 'No'}
   ✅ Tickers analyzed   : {len(results_dict)}/{len(TEST_TICKERS)}
   
   🎯 NEXT BAR PREDICTION
   ─────────────────────────────────────
   Direction  : {latest_pred['direction']}
   Confidence : {latest_pred['confidence']:.1f}%
   Signal     : {signal['signal']} - {signal['strength']}
   
   💡 TIP
   ─────────────────────────────────────
   Try changing CONFIG['TICKER'] in config.py to test different assets!
    """)
    
    print_tips()
    
    # Display dashboard
    try:
        show_dashboard(fig)
    except:
        pass

# ================================================================
# QUICK SCRIPTS (UNCOMMENT TO USE)
# ================================================================

def quick_single_ticker():
    """Quick test on single ticker"""
    from data_handler import get_ohlcv_data, add_economic_calendar
    from feature_engineering import add_technical_features
    from model_training import train_full_pipeline
    
    ticker = 'AAPL'
    print(f"\n📍 Testing single ticker: {ticker}")
    
    df = get_ohlcv_data(ticker)
    df = add_economic_calendar(df)
    df = add_technical_features(df)
    
    pipeline = train_full_pipeline(df)
    print(f"✅ Accuracy: {pipeline['metrics']['accuracy']*100:.2f}%")

def quick_backtest_all():
    """Quick backtest multiple tickers"""
    from backtesting import backtest_multiple_tickers
    
    tickers = ['EURUSD=X', 'GBPUSD=X', 'AAPL']
    print(f"\n📍 Backtesting {len(tickers)} tickers...")
    
    results_df, results = backtest_multiple_tickers(tickers, verbose=False)
    print(results_df)

def quick_predict():
    """Quick prediction on current data"""
    ticker = CONFIG['TICKER']
    
    df = get_ohlcv_data(ticker)
    df = add_economic_calendar(df)
    df = add_technical_features(df)
    
    pipeline = train_full_pipeline(df)
    
    latest = pipeline['X_test'].iloc[-1]
    pred = predict_next_bar(pipeline['model'], pipeline['scaler'], latest)
    
    print(f"\n🔮 Next bar prediction for {ticker}:")
    print(f"   Direction: {pred['direction']}")
    print(f"   Confidence: {pred['confidence']:.1f}%")

# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    """
    Run the complete pipeline
    
    Usage:
        python main.py                    # Run full pipeline
        python main.py --single-ticker    # Test single ticker
        python main.py --backtest         # Quick backtest
        python main.py --predict          # Quick prediction
    """
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--single-ticker':
            quick_single_ticker()
        elif sys.argv[1] == '--backtest':
            quick_backtest_all()
        elif sys.argv[1] == '--predict':
            quick_predict()
        else:
            print("Unknown argument. Running full pipeline...")
            main()
    else:
        main()
