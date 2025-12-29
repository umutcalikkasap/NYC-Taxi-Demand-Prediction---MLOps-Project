"""
Trainer module for XGBoost demand prediction model.
Uses temporal split for realistic time-series validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import logging
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    median_absolute_error
)
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from features import add_time_series_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(data_path: Path) -> pd.DataFrame:
    """
    Load processed training data from parquet file.
    
    Args:
        data_path: Path to processed training data parquet file
    
    Returns:
        DataFrame with processed training data
    
    Raises:
        FileNotFoundError: If data file doesn't exist
        Exception: For other loading errors
    """
    try:
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        df = pd.read_parquet(data_path, engine='pyarrow')
        
        # Verify required columns exist
        required_cols = ['pickup_window', 'PULocationID', 'actual_demand', 'hour', 'day_of_week', 'month']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Print verification info
        print("=" * 70)
        print(f"✓ Loaded training data successfully")
        print(f"✓ Total number of rows: {len(df):,}")
        print(f"✓ Date range: {df['pickup_window'].min()} to {df['pickup_window'].max()}")
        print(f"✓ Unique locations: {df['PULocationID'].nunique()}")
        print("=" * 70)
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise




def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and target variable.
    
    Args:
        df: Input DataFrame with all columns including time-series features
    
    Returns:
        Tuple of (X, y) - features and target
    """
    try:
        # Ensure data is properly sorted (should already be sorted from add_lag_features)
        df = df.sort_values(['PULocationID', 'pickup_window']).reset_index(drop=True)
        
        # Feature columns: all engineered features (25 total)
        feature_cols = [
            # Base temporal and location features
            'PULocationID', 'hour', 'day_of_week', 'month',
            # Advanced temporal features
            'is_weekend', 'is_rush_hour', 'day_of_month', 'week_of_year',
            'time_of_day', 'season', 'is_manhattan',
            # Lag features (historical demand)
            'lag_1', 'lag_4', 'lag_24', 'lag_96', 'lag_672',
            # Rolling statistics
            'rolling_mean_4', 'rolling_std_4', 'rolling_max_24', 'rolling_min_24'
        ]
        
        # Ensure PULocationID is treated as integer (can be used as categorical by XGBoost)
        X = df[feature_cols].copy()
        X['PULocationID'] = X['PULocationID'].astype(int)
        
        # Target variable
        y = df['actual_demand'].copy()
        
        logger.info(f"Features prepared: {feature_cols}")
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target variable shape: {y.shape}")
        
        return X, y
    
    except Exception as e:
        logger.error(f"Error preparing features and target: {str(e)}")
        raise


def temporal_train_val_split(
    df: pd.DataFrame,
    train_months: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform temporal split: first N months for training, remaining for validation.
    This simulates real-world scenario where we predict future demand.
    Note: Lag features must be created BEFORE splitting to avoid data leakage.
    
    Args:
        df: DataFrame with pickup_window column and lag features
        train_months: Number of months to use for training (default: 10)
    
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    try:
        # Ensure pickup_window is datetime
        df = df.copy()
        df['pickup_window'] = pd.to_datetime(df['pickup_window'])
        
        # Get unique months sorted
        df['year_month'] = df['pickup_window'].dt.to_period('M')
        unique_months = sorted(df['year_month'].unique())
        
        print("\nTemporal Split Configuration:")
        print(f"  Total months available: {len(unique_months)}")
        print(f"  Training months: {train_months} (first {train_months} months)")
        print(f"  Validation months: {len(unique_months) - train_months} (last {len(unique_months) - train_months} months)")
        
        # Split based on first N months vs remaining months
        train_months_set = unique_months[:train_months]
        val_months_set = unique_months[train_months:]
        
        print(f"\n  Training period: {train_months_set[0]} to {train_months_set[-1]}")
        print(f"  Validation period: {val_months_set[0]} to {val_months_set[-1]}")
        
        # Create train/val masks
        train_mask = df['year_month'].isin(train_months_set)
        val_mask = df['year_month'].isin(val_months_set)
        
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        
        print(f"\n  Training set size: {len(train_df):,} rows")
        print(f"  Validation set size: {len(val_df):,} rows")
        print("=" * 70)
        
        # Prepare features and targets
        X_train, y_train = prepare_features_and_target(train_df)
        X_val, y_val = prepare_features_and_target(val_df)
        
        return X_train, X_val, y_train, y_val
    
    except Exception as e:
        logger.error(f"Error in temporal split: {str(e)}")
        raise


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> XGBRegressor:
    """
    Train XGBoost model with specified parameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
    
    Returns:
        Trained XGBoost model
    """
    try:
        print("\nModel Training Configuration:")
        print("  Algorithm: XGBoost Regressor")
        print("  Parameters:")
        print("    - n_estimators: 100")
        print("    - learning_rate: 0.1")
        print("    - max_depth: 5")
        print("    - objective: reg:squarederror")
        
        # Initialize model with specified parameters
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        print("\nTraining model...")
        
        # Train model
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        print("✓ Model training completed successfully")
        
        return model
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise


def evaluate_model(
    model: XGBRegressor,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model on validation set with comprehensive metrics.

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target

    Returns:
        Dictionary with evaluation metrics
    """
    try:
        print("\nEvaluating model on validation set...")

        # Make predictions
        y_pred = model.predict(X_val)

        # Calculate comprehensive metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)

        # MAPE - Mean Absolute Percentage Error
        # Add small epsilon to avoid division by zero
        mape = mean_absolute_percentage_error(y_val, y_pred) * 100

        # R² Score - Coefficient of determination
        r2 = r2_score(y_val, y_pred)

        # Median Absolute Error - Robust to outliers
        medae = median_absolute_error(y_val, y_pred)

        # Custom metrics
        # Bias - Mean error (positive = overestimation, negative = underestimation)
        bias = np.mean(y_pred - y_val)

        # Max error
        max_error = np.max(np.abs(y_pred - y_val))

        # Percentage of predictions within 5 trips
        within_5 = np.mean(np.abs(y_pred - y_val) <= 5) * 100

        # Percentage of predictions within 10% of actual
        within_10_pct = np.mean(np.abs((y_pred - y_val) / (y_val + 1e-8)) <= 0.1) * 100

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2_score': r2,
            'median_ae': medae,
            'bias': bias,
            'max_error': max_error,
            'within_5_trips': within_5,
            'within_10_pct': within_10_pct
        }

        # Print metrics
        print("\n" + "=" * 70)
        print("COMPREHENSIVE VALIDATION METRICS:")
        print("=" * 70)
        print("\n📊 Primary Metrics:")
        print(f"  RMSE (Root Mean Squared Error):     {rmse:.4f} trips")
        print(f"  MAE  (Mean Absolute Error):         {mae:.4f} trips")
        print(f"  MedAE (Median Absolute Error):      {medae:.4f} trips")
        print(f"  MAPE (Mean Absolute % Error):       {mape:.2f}%")
        print(f"  R² Score (Coefficient of Determ.):  {r2:.4f}")

        print("\n📈 Additional Metrics:")
        print(f"  Bias (Mean Error):                  {bias:+.4f} trips")
        if abs(bias) < 1:
            print(f"    → {'Well-balanced predictions' if abs(bias) < 0.5 else 'Slight bias'}")
        else:
            print(f"    → {'Overestimating' if bias > 0 else 'Underestimating'}")

        print(f"  Maximum Error:                      {max_error:.2f} trips")

        print("\n🎯 Accuracy Thresholds:")
        print(f"  Predictions within ±5 trips:        {within_5:.1f}%")
        print(f"  Predictions within ±10% of actual:  {within_10_pct:.1f}%")

        print("\n💡 Interpretation:")
        if mae < 5:
            print("  ✅ Excellent performance (MAE < 5)")
        elif mae < 10:
            print("  ✅ Good performance (MAE < 10)")
        else:
            print("  ⚠️  Consider model improvements (MAE ≥ 10)")

        if r2 > 0.8:
            print(f"  ✅ Strong predictive power (R² = {r2:.3f})")
        elif r2 > 0.6:
            print(f"  ✅ Moderate predictive power (R² = {r2:.3f})")
        else:
            print(f"  ⚠️  Weak predictive power (R² = {r2:.3f})")

        print("=" * 70)

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise


def save_model(model: XGBRegressor, model_path: Path) -> None:
    """
    Save trained model to disk in JSON format.

    Args:
        model: Trained XGBoost model
        model_path: Path to save the model
    """
    try:
        # Ensure models directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save using booster (compatible with XGBoost 2.0+)
        model.get_booster().save_model(model_path)

        print(f"\n✓ Model saved to: {model_path}")
        print(f"  File size: {model_path.stat().st_size / 1024:.2f} KB")

    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def main() -> None:
    """
    Main training pipeline.
    """
    try:
        # Define paths
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "data" / "processed_training_data_2024.parquet"
        model_path = project_root / "models" / "xgb_model.json"
        
        print("\n" + "=" * 70)
        print("NYC TAXI DEMAND PREDICTION - TRAINING PIPELINE")
        print("=" * 70)
        
        # 1. Load data
        df = load_training_data(data_path)
        
        # 2. Feature Engineering: Add time-series features (lag + rolling)
        # IMPORTANT: This must be done BEFORE temporal split to maintain temporal order
        # and avoid data leakage. Data is sorted within this function.
        df = add_time_series_features(df, verbose=True)
        
        # 3. Temporal split (first 10 months train, last 2 months validation)
        X_train, X_val, y_train, y_val = temporal_train_val_split(df, train_months=10)
        
        # 4. Setup MLflow
        print("\nSetting up MLflow tracking...")
        mlflow.set_experiment("NYC_Taxi_Demand_v1")
        mlflow.xgboost.autolog()
        print("✓ MLflow autologging enabled")
        print("✓ Experiment: NYC_Taxi_Demand_v1")
        
        # 5. Train model with MLflow tracking
        with mlflow.start_run():
            # Train
            model = train_xgboost_model(X_train, y_train, X_val, y_val)
            
            # Evaluate
            metrics = evaluate_model(model, X_val, y_val)
            
            # Log additional metrics to MLflow
            mlflow.log_metrics(metrics)
            
            # 6. Save model
            save_model(model, model_path)
        
        print("\n" + "=" * 70)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Check MLflow UI: mlflow ui")
        print(f"  2. Model ready for inference: {model_path}")
        print("=" * 70 + "\n")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

