"""
Model comparison script for baseline evaluation.
Compares XGBoost, LightGBM, and CatBoost on the same dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    median_absolute_error
)
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from features import add_time_series_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load data and prepare train/val split."""
    logger.info(f"Loading data from {data_path}")

    df = pd.read_parquet(data_path, engine='pyarrow')
    df = add_time_series_features(df, verbose=False)

    # Temporal split
    df['year_month'] = df['pickup_window'].dt.to_period('M')
    unique_months = sorted(df['year_month'].unique())

    train_months = unique_months[:10]
    val_months = unique_months[10:]

    train_df = df[df['year_month'].isin(train_months)]
    val_df = df[df['year_month'].isin(val_months)]

    feature_cols = [
        'PULocationID', 'hour', 'day_of_week', 'month',
        'lag_1', 'lag_4', 'lag_96', 'rolling_mean_4'
    ]

    X_train = train_df[feature_cols].copy()
    y_train = train_df['actual_demand'].copy()
    X_val = val_df[feature_cols].copy()
    y_val = val_df['actual_demand'].copy()

    return X_train, X_val, y_train, y_val


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive metrics."""
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'r2_score': r2_score(y_true, y_pred),
        'median_ae': median_absolute_error(y_true, y_pred),
        'bias': np.mean(y_pred - y_true),
        'max_error': np.max(np.abs(y_pred - y_true)),
        'within_5_trips': np.mean(np.abs(y_pred - y_true) <= 5) * 100,
        'within_10_pct': np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8)) <= 0.1) * 100
    }
    return metrics


def train_xgboost(X_train, y_train, X_val, y_val) -> Tuple[XGBRegressor, Dict, float]:
    """Train XGBoost model."""
    logger.info("\n" + "=" * 70)
    logger.info("Training XGBoost...")
    logger.info("=" * 70)

    start_time = time.time()

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    train_time = time.time() - start_time
    y_pred = model.predict(X_val)
    metrics = evaluate_model(y_val, y_pred)

    return model, metrics, train_time


def train_lightgbm(X_train, y_train, X_val, y_val) -> Tuple[LGBMRegressor, Dict, float]:
    """Train LightGBM model."""
    logger.info("\n" + "=" * 70)
    logger.info("Training LightGBM...")
    logger.info("=" * 70)

    start_time = time.time()

    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective='regression',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    train_time = time.time() - start_time
    y_pred = model.predict(X_val)
    metrics = evaluate_model(y_val, y_pred)

    return model, metrics, train_time


def train_catboost(X_train, y_train, X_val, y_val) -> Tuple[CatBoostRegressor, Dict, float]:
    """Train CatBoost model."""
    logger.info("\n" + "=" * 70)
    logger.info("Training CatBoost...")
    logger.info("=" * 70)

    start_time = time.time()

    model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=5,
        loss_function='RMSE',
        random_state=42,
        verbose=False
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    train_time = time.time() - start_time
    y_pred = model.predict(X_val)
    metrics = evaluate_model(y_val, y_pred)

    return model, metrics, train_time


def print_comparison(results: Dict):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("MODEL COMPARISON RESULTS")
    print("=" * 100)

    # Header
    print(f"\n{'Metric':<25} {'XGBoost':<20} {'LightGBM':<20} {'CatBoost':<20}")
    print("-" * 100)

    # Metrics to compare
    metrics_to_show = [
        ('RMSE', 'rmse', '{:.4f} trips'),
        ('MAE', 'mae', '{:.4f} trips'),
        ('MedAE', 'median_ae', '{:.4f} trips'),
        ('MAPE', 'mape', '{:.2f}%'),
        ('R² Score', 'r2_score', '{:.4f}'),
        ('Bias', 'bias', '{:+.4f} trips'),
        ('Max Error', 'max_error', '{:.2f} trips'),
        ('Within ±5 trips', 'within_5_trips', '{:.1f}%'),
        ('Within ±10%', 'within_10_pct', '{:.1f}%'),
        ('Training Time', 'train_time', '{:.2f}s')
    ]

    for display_name, metric_key, fmt in metrics_to_show:
        xgb_val = results['xgboost']['metrics'].get(metric_key, results['xgboost'].get(metric_key, 0))
        lgb_val = results['lightgbm']['metrics'].get(metric_key, results['lightgbm'].get(metric_key, 0))
        cat_val = results['catboost']['metrics'].get(metric_key, results['catboost'].get(metric_key, 0))

        xgb_str = fmt.format(xgb_val)
        lgb_str = fmt.format(lgb_val)
        cat_str = fmt.format(cat_val)

        # Highlight best (lower is better for most metrics, except R² and accuracy)
        if metric_key in ['r2_score', 'within_5_trips', 'within_10_pct']:
            best_val = max(xgb_val, lgb_val, cat_val)
        else:
            best_val = min(xgb_val, lgb_val, cat_val)

        if xgb_val == best_val:
            xgb_str += " ✅"
        if lgb_val == best_val:
            lgb_str += " ✅"
        if cat_val == best_val:
            cat_str += " ✅"

        print(f"{display_name:<25} {xgb_str:<20} {lgb_str:<20} {cat_str:<20}")

    print("=" * 100)

    # Determine winner
    scores = {
        'xgboost': 0,
        'lightgbm': 0,
        'catboost': 0
    }

    for _, metric_key, _ in metrics_to_show[:-1]:  # Exclude training time
        xgb_val = results['xgboost']['metrics'].get(metric_key, 0)
        lgb_val = results['lightgbm']['metrics'].get(metric_key, 0)
        cat_val = results['catboost']['metrics'].get(metric_key, 0)

        if metric_key in ['r2_score', 'within_5_trips', 'within_10_pct']:
            best_val = max(xgb_val, lgb_val, cat_val)
        else:
            best_val = min(xgb_val, lgb_val, cat_val)

        if xgb_val == best_val:
            scores['xgboost'] += 1
        if lgb_val == best_val:
            scores['lightgbm'] += 1
        if cat_val == best_val:
            scores['catboost'] += 1

    winner = max(scores, key=scores.get)

    print(f"\n🏆 Overall Winner: {winner.upper()}")
    print(f"   Wins: XGBoost={scores['xgboost']}, LightGBM={scores['lightgbm']}, CatBoost={scores['catboost']}")
    print("=" * 100)


def main():
    """Main comparison pipeline."""
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "processed_training_data_2024.parquet"

    print("\n" + "=" * 70)
    print("NYC TAXI DEMAND PREDICTION - MODEL COMPARISON")
    print("=" * 70)

    # Load data
    X_train, X_val, y_train, y_val = load_and_prepare_data(data_path)

    logger.info(f"Training set: {len(X_train):,} samples")
    logger.info(f"Validation set: {len(X_val):,} samples")

    # Setup MLflow
    mlflow.set_experiment("NYC_Taxi_Model_Comparison")

    results = {}

    # Train XGBoost
    with mlflow.start_run(run_name="xgboost_baseline"):
        xgb_model, xgb_metrics, xgb_time = train_xgboost(X_train, y_train, X_val, y_val)
        mlflow.log_params({
            'model': 'XGBoost',
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1
        })
        mlflow.log_metrics(xgb_metrics)
        mlflow.log_metric('train_time', xgb_time)
        results['xgboost'] = {'metrics': xgb_metrics, 'train_time': xgb_time}

    # Train LightGBM
    with mlflow.start_run(run_name="lightgbm_baseline"):
        lgb_model, lgb_metrics, lgb_time = train_lightgbm(X_train, y_train, X_val, y_val)
        mlflow.log_params({
            'model': 'LightGBM',
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1
        })
        mlflow.log_metrics(lgb_metrics)
        mlflow.log_metric('train_time', lgb_time)
        results['lightgbm'] = {'metrics': lgb_metrics, 'train_time': lgb_time}

    # Train CatBoost
    with mlflow.start_run(run_name="catboost_baseline"):
        cat_model, cat_metrics, cat_time = train_catboost(X_train, y_train, X_val, y_val)
        mlflow.log_params({
            'model': 'CatBoost',
            'iterations': 100,
            'depth': 5,
            'learning_rate': 0.1
        })
        mlflow.log_metrics(cat_metrics)
        mlflow.log_metric('train_time', cat_time)
        results['catboost'] = {'metrics': cat_metrics, 'train_time': cat_time}

    # Print comparison
    print_comparison(results)

    print("\n💡 Recommendation:")
    print("   Check MLflow UI for detailed comparison: mlflow ui")
    print("   Access at: http://localhost:5000")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
