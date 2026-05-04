# 🚀 QUICK START GUIDE - Forex Prediction Model (Refactored)

## What You Got

Your 600+ line monolithic file has been refactored into **10 organized modules** with clear responsibilities. Everything works the same, but now it's maintainable, testable, and extensible! ✨

---

## 📁 Project Structure

```
forex_prediction_model/
├── main.py                  👈 RUN THIS FILE
├── config.py               # All settings in one place
├── data_handler.py         # Download & clean data
├── feature_engineering.py  # Technical indicators
├── model_training.py       # ML training
├── prediction.py           # Generate signals
├── backtesting.py          # Test strategy
├── visualization.py        # Create dashboard
├── utils.py                # Helper functions
├── requirements.txt        # Dependencies
└── README files
```

---

## ⚡ Quick Start (3 steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Pipeline
```bash
python main.py
```

### Step 3: Wait for Results! ☕
- Dashboard saved as `prediction_dashboard.png`
- Feature importance exported as `feature_importance.csv`
- Trades logged as `backtest_trades.csv`

---

## 🎯 What Each File Does

| File | Purpose | Key Functions |
|------|---------|---|
| **main.py** | Orchestra - runs everything | `main()` |
| **config.py** | Settings (change TICKER here) | CONFIG, COLORS, PERIODS |
| **data_handler.py** | Download & clean data | `get_ohlcv_data()`, `add_economic_calendar()` |
| **feature_engineering.py** | Build indicators | `add_technical_features()`, `compute_rsi()` |
| **model_training.py** | Train XGBoost | `train_full_pipeline()`, `train_xgboost()` |
| **prediction.py** | Generate signals | `make_predictions()`, `predict_next_bar()` |
| **backtesting.py** | Test strategy | `run_backtest()`, `run_model_for_ticker()` |
| **visualization.py** | Create charts | `create_dashboard()`, `save_dashboard()` |
| **utils.py** | Helpers | `print_*()`, `validate_*()` |

---

## 🔧 How to Customize

### Change the Ticker
Open `config.py` and modify:
```python
CONFIG = {
    'TICKER': 'EURUSD=X',  # 👈 CHANGE THIS
    'START_DATE': '2016-01-01',
    'END_DATE': '2025-12-31',
    # ... other settings
}
```

**Available Tickers:**
- Forex: `'EURUSD=X'`, `'GBPUSD=X'`, `'USDJPY=X'`
- Stocks: `'AAPL'`, `'TSLA'`, `'GOOGL'`, `'MSFT'`
- India: `'^NSEI'`, `'RELIANCE.NS'`, `'TCS.NS'`
- Indices: `'^GSPC'` (S&P500), `'^DJI'` (Dow Jones)

### Adjust Model Parameters
In `config.py`:
```python
XGBOOST_PARAMS = {
    'n_estimators': 200,  # More trees = better but slower
    'max_depth': 6,       # Tree depth
    'learning_rate': 0.1, # Lower = slower but better
}
```

### Modify Feature Periods
In `config.py`:
```python
PERIODS = {
    'MA_SHORT': 5,    # 5-day moving average
    'MA_MEDIUM': 20,  # 20-day moving average
    'RSI': 14,        # 14-period RSI
}
```

---

## 📚 Common Workflows

### Workflow 1: Test a Different Ticker
```python
# In config.py
CONFIG['TICKER'] = 'GBPUSD=X'

# Run
python main.py
```

### Workflow 2: Quick Single Ticker Test
```bash
python main.py --single-ticker
```

### Workflow 3: Backtest All Tickers
```bash
python main.py --backtest
```

### Workflow 4: Quick Prediction
```bash
python main.py --predict
```

### Workflow 5: Use in Your Code
```python
from data_handler import get_ohlcv_data, add_economic_calendar
from feature_engineering import add_technical_features
from model_training import train_full_pipeline
from prediction import predict_next_bar

# Get data
df = get_ohlcv_data('AAPL')
df = add_economic_calendar(df)
df = add_technical_features(df)

# Train
pipeline = train_full_pipeline(df)

# Predict
latest = pipeline['X_test'].iloc[-1]
pred = predict_next_bar(pipeline['model'], pipeline['scaler'], latest)

print(f"Direction: {pred['direction']}")
print(f"Confidence: {pred['confidence']:.1f}%")
```

---

## 🔍 Understanding the Output

### Console Output
```
✅ All libraries imported successfully
📌 Ticker   : EURUSD=X
📥 Downloaded 2452 bars
🧹 Final dataset: 2452 rows
📅 NFP weeks flagged  : 98
🔧 Created 68 features
🤖 Model trained
📈 Accuracy: 58.25%
🔮 Generated 392 predictions
📊 Running backtest...
🏆 TOP 3 FOREX PAIRS
   1. EURUSD=X - Confidence: 72.3%
```

### Dashboard (`prediction_dashboard.png`)
- **Top Left**: Price chart (last 300 bars)
- **Top Right**: Model accuracy
- **Middle**: Predictions distribution, win rate, P&L
- **Bottom**: Feature importance (top 15)

### CSV Exports
- **feature_importance.csv**: Which features matter most
- **backtest_trades.csv**: Each trade with entry/exit/profit

---

## 🐛 Troubleshooting

### Error: "No data found for ticker"
- Check ticker spelling (use correct yfinance format)
- Example: `'EURUSD=X'` not `'EUR/USD'`

### Error: "Module not found"
```bash
pip install -r requirements.txt
```

### Model accuracy too low?
- Add more features in `feature_engineering.py`
- Adjust `XGBOOST_PARAMS` in `config.py`
- Use longer historical data (extend `END_DATE`)
- Try different `PERIODS` settings

### Dashboard not showing?
- Check if `prediction_dashboard.png` was created
- Try: `plt.show()` requires display (may not work in remote environments)

---

## 📖 Module Guide

### Import from Modules

```python
# Data operations
from data_handler import download_data, clean_data, add_economic_calendar

# Feature building
from feature_engineering import add_technical_features, compute_rsi

# Model training
from model_training import train_xgboost, evaluate_model, get_feature_importance

# Predictions
from prediction import make_predictions, predict_next_bar, generate_signal

# Backtesting
from backtesting import run_backtest, run_model_for_ticker

# Visualization
from visualization import create_dashboard, save_dashboard

# Utilities
from utils import print_success, validate_config, print_tips
```

---

## 🎓 What Changed (Architecture)

### Before (600+ lines, one file)
❌ Hard to navigate
❌ Difficult to test
❌ Hard to reuse components
❌ Mixing concerns
❌ Hard to extend

### After (Modular design)
✅ Clear structure
✅ Easy to unit test
✅ Reusable components
✅ Separated concerns
✅ Easy to add new models

---

## 💡 Advanced Usage

### Add New Technical Indicator
In `feature_engineering.py`:
```python
# Add your indicator function
def compute_my_indicator(series, period=20):
    # Your calculation
    return indicator_values

# Add to add_technical_features()
df['MyIndicator'] = compute_my_indicator(df['Close'])
```

### Add New Model
In `model_training.py`:
```python
def train_lstm(X_train, y_train):
    # Your LSTM code
    return model
```

### Export Results
In `visualization.py`:
```python
def export_to_excel(pipeline):
    # Your export code
    pass
```

---

## 📊 Feature List (68 Total)

### Trend Indicators (6)
MA5, MA20, MA50, MA_Cross, MA5_slope, MA20_slope

### Volatility (7)
Volatility, Vol_5, Vol_20, Vol_Regime, Volatility_Ratio, Vol_Squeeze, BB_Width

### Momentum (8)
Momentum, RSI, RSI_OS, RSI_OB, RSI_Slope, MACD_Hist, Return_MA5, Mom_Acc

### Volume (4)
Vol_Ratio, Vol_Trend, PV_Trend, Return

### Candles (6)
Body_Size, Upper_Wick, Lower_Wick, Is_Bullish, Range, ATR

### Mean Reversion (4)
Z_Score, Price_Position, MA50_dist, BB_Pos

### Calendar (5)
Is_MonthEnd, Is_MonthStart, Is_QuarterEnd, Is_NFP_Week, Is_Fed_Week, Is_ECB_Week

### Plus 22 more advanced features!

---

## 🚀 Next Steps

1. **Run it**: `python main.py`
2. **Change ticker**: Edit `config.py`
3. **Customize**: Modify parameters in `config.py`
4. **Extend**: Add new features in `feature_engineering.py`
5. **Deploy**: Use predictions in your trading bot!

---

## 📞 Support

- Check console output for errors
- Review docstrings in each module
- Check `REFACTORING_GUIDE.md` for detailed explanation
- Modify `config.py` for all user-facing settings

---

**Happy Trading! 📈**
