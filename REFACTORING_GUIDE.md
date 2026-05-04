# 🚀 Forex Prediction Model - Refactoring Guide

## 📁 Proposed Project Structure

```
forex_prediction_model/
│
├── config.py                 # Configuration & constants
├── data_handler.py          # Data download & loading
├── feature_engineering.py   # Technical indicators & features
├── model_training.py        # Model training (XGBoost)
├── prediction.py            # Make predictions
├── backtesting.py           # Backtest logic
├── visualization.py         # Dashboard & plotting
├── utils.py                 # Helper functions
├── main.py                  # Main orchestration file
└── requirements.txt         # Dependencies
```

---

## 📋 File Breakdown

### 1. **config.py** - Configuration & Constants
**What it contains:**
- All hardcoded values (ticker, dates, thresholds)
- Configuration dictionaries
- Color schemes for visualization
- Hyperparameters

**Why separate?**
- Easy to change parameters without touching code logic
- Single source of truth for all settings

---

### 2. **data_handler.py** - Data Download & Processing
**What it contains:**
- `download_data()` - Fetch data from yfinance
- `clean_data()` - Clean and prepare dataframe
- `add_economic_calendar()` - Add calendar events

**Why separate?**
- Data handling is a distinct responsibility
- Can be reused for different models
- Easy to test independently

---

### 3. **feature_engineering.py** - Technical Indicators
**What it contains:**
- `compute_rsi()` - RSI calculation
- `add_technical_features()` - All OHLC features
- `add_calendar_features()` - Economic calendar flags
- All indicator calculations (MACD, Bollinger Bands, etc.)

**Why separate?**
- Feature building is complex and modular
- Can be extended with new features easily
- Can be reused across different models

---

### 4. **model_training.py** - Model Training
**What it contains:**
- `create_labels()` - Create target variable
- `scale_features()` - Normalize data
- `train_xgboost()` - Train XGBoost model
- `evaluate_model()` - Get accuracy & metrics

**Why separate?**
- ML training logic isolated
- Easy to add new models (LSTM, etc.)
- Clean train/test split handling

---

### 5. **prediction.py** - Make Predictions
**What it contains:**
- `make_prediction()` - Generate next bar prediction
- `get_confidence()` - Calculate confidence score
- `interpret_signal()` - Convert to trading signal

**Why separate?**
- Prediction logic separate from training
- Can be called independently for live trading
- Easy to add ensemble methods

---

### 6. **backtesting.py** - Backtest Strategy
**What it contains:**
- `run_backtest()` - Execute backtest
- `calculate_metrics()` - PnL, win rate, etc.
- `run_model_for_ticker()` - Multi-ticker testing

**Why separate?**
- Backtesting is distinct from training
- Can test different strategies
- Easy to modify PnL calculation

---

### 7. **visualization.py** - Dashboard & Plotting
**What it contains:**
- `create_dashboard()` - Main visualization
- `styled_ax()` - Consistent styling
- `plot_price_history()` - Price chart
- `plot_accuracy()` - Accuracy bar chart
- `plot_features()` - Feature importance

**Why separate?**
- All visualization code in one place
- Easy to modify charts without touching ML logic
- Can export different formats easily

---

### 8. **utils.py** - Helper Functions
**What it contains:**
- `print_section()` - Formatted printing
- `print_results()` - Results formatting
- `get_color()` - Color selection
- Error handling utilities

**Why separate?**
- Common utilities used everywhere
- Keep code DRY (Don't Repeat Yourself)
- Easy to maintain helper functions

---

### 9. **main.py** - Main Orchestration
**What it contains:**
- Imports all modules
- Orchestrates the pipeline
- Calls functions in order
- Runs the complete workflow

**Why separate?**
- High-level flow is clear
- Easy to understand what happens when
- Can run different workflows (train, predict, backtest)

---

## 🔄 Data Flow

```
main.py
  ├── config.py (get parameters)
  ├── data_handler.py (download & clean data)
  ├── feature_engineering.py (add features)
  ├── model_training.py (train model)
  ├── prediction.py (generate signals)
  ├── backtesting.py (test strategy)
  ├── visualization.py (create dashboard)
  └── utils.py (helpers throughout)
```

---

## 💡 Key Benefits of This Structure

✅ **Modularity** - Each file has one responsibility
✅ **Reusability** - Import functions in other projects
✅ **Testability** - Easy to write unit tests
✅ **Maintainability** - Find and fix bugs quickly
✅ **Scalability** - Easy to add new models, tickers, features
✅ **Clarity** - Clear separation of concerns
✅ **Flexibility** - Can use individual components

---

## 🚀 How to Use

### Run the complete pipeline:
```bash
python main.py
```

### Use individual components:
```python
from data_handler import download_data, clean_data
from feature_engineering import add_technical_features
from model_training import train_xgboost

# Download data
df = download_data('EURUSD=X', '2016-01-01', '2025-12-31')
df = clean_data(df)

# Add features
df = add_technical_features(df)

# Train model
model, scaler = train_xgboost(df)
```

---

## 📦 Dependencies
```
numpy
pandas
matplotlib
yfinance
scikit-learn
xgboost
```

---

## 🔧 Next Steps

1. Create these files in a new folder
2. Run: `pip install -r requirements.txt`
3. Run: `python main.py`
4. The model will work exactly the same, but now it's organized!

