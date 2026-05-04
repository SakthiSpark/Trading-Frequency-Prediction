# 📈 Forex Price Prediction using Machine Learning

## 🚀 Project Overview

This project focuses on building a machine learning-based system to predict short-term Forex price movements (e.g., EUR/USD) using historical market data. The goal is not just prediction accuracy, but evaluating whether the model can generate profitable trading signals.

---

## 🎯 Objectives

* Predict price direction (UP/DOWN) for the next *N bars*
* Build a robust feature engineering pipeline using technical indicators
* Avoid data leakage using time-series-aware validation
* Evaluate real-world performance using backtesting

---

## 🧠 Approach

### 🔹 Data Collection

* Historical price data (Open, High, Low, Close)
* Time period: 2016 – 2025
* Assets tested: EURUSD, GBPUSD, AAPL, NSEI

---

### 🔹 Feature Engineering

Created 40+ features including:

**📊 Volatility**

* Rolling standard deviation
* Volatility ratio
* Volatility squeeze

**📈 Momentum**

* Returns, momentum
* RSI (Relative Strength Index)
* MACD (Moving Average Convergence Divergence)

**📉 Trend**

* Moving averages (MA5, MA20, MA50)
* MA slope and distance

**🕯️ Candle Structure**

* Body size
* Upper wick / Lower wick
* Bullish/Bearish encoding

---

### 🔹 Model

* Algorithm: XGBoost Classifier
* Focus on regularization to reduce overfitting
* Manual hyperparameter tuning + Bayesian optimization concepts

---

### 🔹 Validation Strategy

* Time-based train-test split (no shuffling)
* Prevented data leakage by ensuring future data is not used in training

---

### 🔹 Backtesting Engine

Simulated trading strategy:

* Buy/Sell based on model predictions
* Exit after fixed number of bars
* Included:

  * Transaction cost
  * Confidence-based filtering

---

## 📊 Results

| Metric   | Value                      |
| -------- | -------------------------- |
| Accuracy | ~77%                       |
| Win Rate | ~70%                       |
| Profit   | Positive (varies by asset) |

⚠️ Key Insight:

> Higher accuracy does not always lead to higher profitability.

---

## 🧠 Key Learnings

* Data leakage can lead to misleading results in time-series models
* Feature engineering is more important than model complexity
* Backtesting is essential to evaluate real-world performance
* Financial ML requires optimizing for profit, not just accuracy

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib

---

## 📂 Project Structure

```
├── data/                 # Raw & processed data
├── features/             # Feature engineering
├── model/                # Model training
├── backtesting/          # Trading simulation
├── utils/                # Helper functions
└── main.py               # Main pipeline
```

---

## 📌 Future Improvements

* Walk-forward validation
* Advanced backtesting (stop loss, position sizing)
* SHAP-based model explainability
* Integration with real-time data

---

## 🔗 Author

Developed as part of a machine learning project focused on financial forecasting and trading strategy design.
