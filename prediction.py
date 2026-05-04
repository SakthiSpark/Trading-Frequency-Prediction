"""
PREDICTION.PY - Make Predictions
Generate trading signals and predictions
"""

import pandas as pd
import numpy as np
from config import CONFIG
from utils import print_section, print_success, get_direction_emoji

# ================================================================
# PREDICTION FUNCTIONS
# ================================================================

def make_predictions(model, scaler, X_test):
    """
    Make predictions on test data
    
    Args:
        model (XGBClassifier): Trained model
        scaler (StandardScaler): Fitted scaler
        X_test (pd.DataFrame): Test features
    
    Returns:
        dict: predictions, probabilities, confidence scores
    """
    print_section("🔮 MAKING PREDICTIONS", "🔮")
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Extract UP probability (class 1)
    up_prob = probabilities[:, 1]
    
    # Calculate confidence (distance from 50%)
    confidence = np.abs(up_prob - 0.5) * 2 * 100
    
    print_success(f"Generated {len(predictions)} predictions")
    print_success(f"Mean confidence: {confidence.mean():.1f}%")
    print_success(f"Max confidence: {confidence.max():.1f}%")
    
    return {
        'predictions': predictions,
        'up_probability': up_prob,
        'confidence': confidence
    }

def predict_next_bar(model, scaler, latest_features):
    """
    Predict direction for next bar using latest features
    
    Args:
        model (XGBClassifier): Trained model
        scaler (StandardScaler): Fitted scaler
        latest_features (pd.DataFrame): Latest feature row (1 row)
    
    Returns:
        dict: Direction, confidence, probability
    """
    # Ensure it's a 2D array
    if latest_features.ndim == 1:
        latest_features = latest_features.values.reshape(1, -1)
    elif isinstance(latest_features, pd.DataFrame):
        latest_features = latest_features.values
    
    # Make prediction
    prediction = model.predict(latest_features)[0]
    probability = model.predict_proba(latest_features)[0]
    
    up_prob = probability[1]
    confidence = abs(up_prob - 0.5) * 2 * 100
    
    direction = "📈 UP" if prediction == 1 else "📉 DOWN"
    
    return {
        'prediction': prediction,
        'direction': direction,
        'up_probability': up_prob,
        'confidence': confidence
    }

# ================================================================
# SIGNAL GENERATION
# ================================================================

def generate_signal(prediction_dict, min_confidence=60.0):
    """
    Generate trading signal from prediction
    
    Args:
        prediction_dict (dict): Output from predict_next_bar
        min_confidence (float): Minimum confidence to generate signal
    
    Returns:
        dict: Signal with strength rating
    """
    confidence = prediction_dict['confidence']
    prediction = prediction_dict['prediction']
    
    if confidence < min_confidence:
        signal_strength = "🟡 WEAK"
        signal_action = "HOLD"
    elif confidence < 75:
        signal_strength = "🟢 MODERATE"
        signal_action = "LONG" if prediction == 1 else "SHORT"
    else:
        signal_strength = "🟢 STRONG"
        signal_action = "LONG" if prediction == 1 else "SHORT"
    
    return {
        'signal': signal_action,
        'strength': signal_strength,
        'confidence': confidence
    }

def interpret_market(features_dict):
    """
    Interpret market conditions from features
    
    Args:
        features_dict (dict): Dictionary of latest features
    
    Returns:
        dict: Market interpretation
    """
    interpretation = {}
    
    # Trend
    if features_dict.get('MA_Cross', 0) > 0:
        interpretation['trend'] = "📈 UPTREND"
    elif features_dict.get('MA_Cross', 0) < 0:
        interpretation['trend'] = "📉 DOWNTREND"
    else:
        interpretation['trend'] = "➡️  SIDEWAYS"
    
    # Volatility
    if features_dict.get('Vol_Squeeze', 0) == 1:
        interpretation['volatility'] = "😴 LOW (Explosion coming)"
    elif features_dict.get('Vol_Regime', 1) > 1.5:
        interpretation['volatility'] = "🌪️  HIGH"
    else:
        interpretation['volatility'] = "⚖️  NORMAL"
    
    # Momentum
    rsi = features_dict.get('RSI', 50)
    if rsi > 70:
        interpretation['momentum'] = "🔥 OVERBOUGHT"
    elif rsi < 30:
        interpretation['momentum'] = "🧊 OVERSOLD"
    else:
        interpretation['momentum'] = "⚡ NEUTRAL"
    
    # Price level
    z_score = features_dict.get('Z_Score', 0)
    if z_score > 2:
        interpretation['level'] = "📈 EXTENDED UP"
    elif z_score < -2:
        interpretation['level'] = "📉 EXTENDED DOWN"
    else:
        interpretation['level'] = "⚓ BALANCED"
    
    return interpretation

# ================================================================
# PREDICTION ANALYSIS
# ================================================================

def analyze_prediction_quality(predictions, probabilities, y_true):
    """
    Analyze quality of predictions
    
    Args:
        predictions (np.array): Model predictions
        probabilities (np.array): Prediction probabilities
        y_true (np.array): True labels
    
    Returns:
        dict: Quality metrics
    """
    # Overall accuracy
    accuracy = (predictions == y_true).mean()
    
    # Accuracy by confidence level
    confidence = np.abs(probabilities - 0.5) * 2
    high_conf_idx = confidence > 0.75
    
    if high_conf_idx.sum() > 0:
        high_conf_acc = (predictions[high_conf_idx] == y_true.iloc[high_conf_idx]).mean()
    else:
        high_conf_acc = 0
    
    # Win rate on high confidence
    if high_conf_idx.sum() > 0:
        high_conf_count = high_conf_idx.sum()
        high_conf_correct = (predictions[high_conf_idx] == y_true.iloc[high_conf_idx]).sum()
    else:
        high_conf_count = 0
        high_conf_correct = 0
    
    return {
        'overall_accuracy': accuracy * 100,
        'high_confidence_accuracy': high_conf_acc * 100,
        'high_confidence_count': high_conf_count,
        'high_confidence_correct': high_conf_correct,
        'mean_confidence': confidence.mean() * 100
    }

def get_prediction_stats(prediction_dict):
    """
    Get statistics from prediction dictionary
    
    Args:
        prediction_dict (dict): From make_predictions()
    
    Returns:
        pd.DataFrame: Statistics summary
    """
    preds = prediction_dict['predictions']
    conf = prediction_dict['confidence']
    
    stats = {
        'Total Predictions': len(preds),
        'UP Signals': (preds == 1).sum(),
        'DOWN Signals': (preds == 0).sum(),
        'Mean Confidence': f"{conf.mean():.1f}%",
        'Max Confidence': f"{conf.max():.1f}%",
        'Min Confidence': f"{conf.min():.1f}%",
        'High Confidence': f"{(conf > 75).sum()}",
    }
    
    return pd.DataFrame(stats, index=[0]).T

# ================================================================
# BATCH PREDICTION
# ================================================================

def predict_multiple_tickers(tickers, get_data_func, model, scaler, feature_names):
    """
    Make predictions for multiple tickers
    
    Args:
        tickers (list): Ticker symbols
        get_data_func: Function to get data for ticker
        model: Trained model
        scaler: Fitted scaler
        feature_names (list): Feature column names
    
    Returns:
        dict: Predictions for each ticker
    """
    results = {}
    
    for ticker in tickers:
        try:
            # Get data
            df = get_data_func(ticker)
            
            if df is None or len(df) == 0:
                continue
            
            # Get latest features
            latest = df.iloc[-1][feature_names]
            
            # Make prediction
            pred_dict = predict_next_bar(model, scaler, latest)
            pred_dict['ticker'] = ticker
            
            results[ticker] = pred_dict
            
        except Exception as e:
            print(f"   Error predicting {ticker}: {e}")
            continue
    
    return results

# ================================================================
# SIGNAL RANKING
# ================================================================

def rank_signals(predictions_dict):
    """
    Rank signals by confidence
    
    Args:
        predictions_dict (dict): Predictions for multiple tickers
    
    Returns:
        pd.DataFrame: Ranked signals
    """
    data = []
    
    for ticker, pred in predictions_dict.items():
        data.append({
            'Ticker': ticker,
            'Direction': pred['direction'],
            'Confidence': pred['confidence'],
            'Probability': pred['up_probability'],
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Confidence', ascending=False)
    
    return df
