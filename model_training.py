"""
MODEL_TRAINING.PY - Model Training & Evaluation
Handles all ML model operations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from config import CONFIG, XGBOOST_PARAMS
from feature_engineering import get_feature_list, drop_nan_rows
from utils import print_section, print_success, print_model_results

# ================================================================
# TARGET VARIABLE CREATION
# ================================================================

def create_labels(df, forward_bars=None, threshold=None):
    """
    Create binary labels (0=DOWN, 1=UP) based on future price movement
    
    Args:
        df (pd.DataFrame): OHLCV dataframe
        forward_bars (int): Bars ahead to predict. Uses CONFIG if None
        threshold (float): Minimum % move to count as UP. Uses CONFIG if None
    
    Returns:
        pd.Series: Binary labels (0 or 1)
    """
    if forward_bars is None:
        forward_bars = CONFIG['FORWARD_BARS']
    if threshold is None:
        threshold = CONFIG['THRESHOLD']
    
    future_returns = df['Close'].pct_change(forward_bars).shift(-forward_bars)
    labels = (future_returns > threshold).astype(int)
    
    return labels

# ================================================================
# DATA PREPARATION
# ================================================================

def prepare_training_data(df):
    """
    Prepare data for model training
    
    Args:
        df (pd.DataFrame): Feature-engineered dataframe
    
    Returns:
        tuple: (X, y, feature_names)
    """
    print_section("📊 PREPARING TRAINING DATA", "📊")
    
    df = df.copy()
    
    # Create labels
    print("   Creating target variable...")
    y = create_labels(df)
    
    # Remove OHLCV columns
    exclude = ['Close', 'High', 'Low', 'Open', 'Volume']
    X = df.drop(columns=exclude)
    
    # Drop NaN rows
    valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[valid_idx].copy()
    y = y[valid_idx].copy()
    
    print_success(f"Training set size: {len(X)} samples")
    print_success(f"Features: {len(X.columns)}")
    print_success(f"Class distribution: {(y==0).sum()} DOWN, {(y==1).sum()} UP")
    
    return X, y, X.columns.tolist()

def split_data(X, y, test_size=None, random_state=42):
    """
    Split data into train/test sets
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        test_size (float): Test set fraction. Uses CONFIG if None
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if test_size is None:
        test_size = CONFIG['TEST_SIZE']
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42  # ← add shuffle=False
)

    print_success(f"Train set: {len(X_train)} samples")
    print_success(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

# ================================================================
# FEATURE SCALING
# ================================================================

def scale_features(X_train, X_test):
    """
    Normalize features using StandardScaler
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    print("   Scaling features...")
    
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames with column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print_success("Features scaled")
    
    return X_train_scaled, X_test_scaled, scaler

# ================================================================
# MODEL TRAINING
# ================================================================

def train_xgboost(X_train, y_train, params=None):
    """
    Train XGBoost classifier with class weighting
    
    Args:
        X_train (pd.DataFrame): Training features (scaled)
        y_train (pd.Series): Training labels
        params (dict): XGBoost parameters. Uses XGBOOST_PARAMS if None
    
    Returns:
        XGBClassifier: Trained model
    """
    print_section("🤖 TRAINING XGBOOST MODEL", "🤖")
    
    if params is None:
        params = XGBOOST_PARAMS.copy()
    
    print("   Initializing model...")
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    sample_weights = class_weights[y_train]
    
    # Create and train model
    model = XGBClassifier(
        **params

    )
    
    
    print("   Training...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        verbose=0
    )
    
    print_success("Model trained")
    
    return model

# ================================================================
# MODEL EVALUATION
# ================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set
    
    Args:
        model (XGBClassifier): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
    
    Returns:
        dict: Metrics (accuracy, predictions, probabilities)
    """
    print_section("📈 EVALUATING MODEL", "📈")
    
    # Make predictions
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities > 0.50).astype(int)  # ← 60% threshold
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    print_success(f"Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print("\n   Classification Report:")
    print(classification_report(y_test, predictions, target_names=['DOWN', 'UP']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("   Confusion Matrix:")
    print(f"      True DOWN:  {cm[0,0]} | False UP:  {cm[0,1]}")
    print(f"      False DOWN: {cm[1,0]} | True UP:   {cm[1,1]}")
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'probabilities': probabilities,
        'confusion_matrix': cm
    }

def get_feature_importance(model, feature_names, top_n=15):
    """
    Get feature importance from trained model
    
    Args:
        model (XGBClassifier): Trained model
        feature_names (list): Names of features
        top_n (int): Number of top features to return
    
    Returns:
        pd.Series: Feature importance (sorted descending)
    """
    print_section("🔍 FEATURE IMPORTANCE", "🔍")
    
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    
    print(f"   Top {top_n} Features:")
    for i, (feat, imp) in enumerate(importance.head(top_n).items(), 1):
        print(f"   {i:2d}. {feat:25s} : {imp:.4f}")
    
    return importance

# ================================================================
# MODEL PIPELINE
# ================================================================

def train_full_pipeline(df):
    """
    Complete training pipeline in one function
    
    Args:
        df (pd.DataFrame): Feature-engineered dataframe
    
    Returns:
        dict: model, scaler, feature_names, evaluation_metrics, importance
    """
    print_section("🔄 FULL TRAINING PIPELINE", "🔄")
    
    # Prepare data
    X, y, feature_names = prepare_training_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train model
    model = train_xgboost(X_train_scaled, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    # Feature importance
    importance = get_feature_importance(model, feature_names)
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'metrics': metrics,
        'importance': importance,
        'split_point': len(X_train)
    }

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def get_model_params():
    """Return current XGBoost parameters"""
    return XGBOOST_PARAMS.copy()

def update_model_params(**kwargs):
    """Update XGBoost parameters"""
    XGBOOST_PARAMS.update(kwargs)
    print_success("Model parameters updated")
