"""
Automated retraining pipeline with drift and performance triggers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from features import add_time_series_features
from monitoring.drift_detector import DriftDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'PULocationID', 'hour', 'day_of_week', 'month',
    'is_weekend', 'is_rush_hour', 'day_of_month', 'week_of_year',
    'time_of_day', 'season', 'is_manhattan',
    'lag_1', 'lag_4', 'lag_24', 'lag_96', 'lag_672',
    'rolling_mean_4', 'rolling_std_4', 'rolling_max_24', 'rolling_min_24'
]


class ContinualLearningPipeline:
    """
    Manages continual learning lifecycle:
    - Performance monitoring
    - Drift detection
    - Automated retraining
    - Model versioning
    - A/B testing
    """

    def __init__(
        self,
        project_root: Path,
        baseline_mae_threshold: float = 2.5,
        drift_threshold: float = 0.05
    ):
        """
        Initialize continual learning pipeline.

        Args:
            project_root: Project root directory
            baseline_mae_threshold: MAE threshold for triggering retrain (20% increase)
            drift_threshold: P-value threshold for drift detection
        """
        self.project_root = project_root
        self.baseline_mae_threshold = baseline_mae_threshold
        self.drift_threshold = drift_threshold
        self.models_dir = project_root / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Model registry
        self.registry_file = self.models_dir / "model_registry.json"
        self.registry = self._load_registry()

        logger.info("Continual Learning Pipeline initialized")


    def _load_registry(self) -> Dict:
        """Load model registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {
            "models": [],
            "current_production": None,
            "baseline_mae": 2.32  # From initial training
        }


    def _save_registry(self):
        """Save model registry to file."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
        logger.info(f"Registry saved: {len(self.registry['models'])} models")


    def load_production_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load production data for a time window.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_rows: Maximum rows to load

        Returns:
            DataFrame with production data
        """
        data_path = self.project_root / "data" / "processed_inference_data_2025.parquet"

        logger.info(f"Loading production data from {data_path}")
        df = pd.read_parquet(data_path, engine='pyarrow')

        df['pickup_window'] = pd.to_datetime(df['pickup_window'])

        # Filter by date range
        if start_date:
            df = df[df['pickup_window'] >= start_date]
        if end_date:
            df = df[df['pickup_window'] <= end_date]

        # Sample if needed
        if max_rows and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42)

        logger.info(f"Loaded {len(df):,} production records")
        if len(df) > 0:
            logger.info(f"Date range: {df['pickup_window'].min()} to {df['pickup_window'].max()}")

        return df


    def check_performance(
        self,
        production_data: pd.DataFrame
    ) -> Dict:
        """
        Check current model performance on recent data.

        Args:
            production_data: Recent production data with actuals

        Returns:
            Performance metrics dictionary
        """
        logger.info("Checking model performance...")

        # Load current model
        current_model_path = self.models_dir / "xgb_model.json"
        if not current_model_path.exists():
            raise FileNotFoundError("No production model found")

        booster = XGBRegressor()
        booster.get_booster().load_model(current_model_path)

        # Prepare data
        df = add_time_series_features(production_data.copy(), verbose=False)
        X = df[FEATURE_COLUMNS]
        y = df['actual_demand']

        # Make predictions
        y_pred = booster.predict(X)

        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'baseline_mae': self.registry['baseline_mae'],
            'mae_increase_pct': (mae - self.registry['baseline_mae']) / self.registry['baseline_mae'] * 100,
            'samples': len(y)
        }

        logger.info(f"Performance: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        logger.info(f"MAE change: {metrics['mae_increase_pct']:+.1f}%")

        return metrics


    def check_drift(
        self,
        production_data: pd.DataFrame,
        reference_data: pd.DataFrame
    ) -> Dict:
        """
        Check for data drift.

        Args:
            production_data: Recent production data
            reference_data: Training/reference data

        Returns:
            Drift detection results
        """
        logger.info("Checking for data drift...")

        # Prepare both datasets
        prod_df = add_time_series_features(production_data.copy(), verbose=False)
        ref_df = add_time_series_features(reference_data.copy(), verbose=False)

        # Initialize drift detector
        detector = DriftDetector(
            reference_data=ref_df,
            feature_columns=FEATURE_COLUMNS,
            categorical_features=['PULocationID'],
            drift_threshold=self.drift_threshold
        )

        # Detect drift
        drift_results = detector.detect_feature_drift(prod_df)

        # Count drifted features
        drifted_features = [
            feat for feat, result in drift_results.items()
            if result['drift_detected']
        ]

        summary = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': len(drifted_features) > 0,
            'total_features': len(FEATURE_COLUMNS),
            'drifted_features_count': len(drifted_features),
            'drifted_features': drifted_features,
            'drift_details': drift_results
        }

        if summary['drift_detected']:
            logger.warning(f"DRIFT DETECTED in {len(drifted_features)} features: {drifted_features}")
        else:
            logger.info("No significant drift detected")

        return summary


    def should_retrain(
        self,
        performance_metrics: Dict,
        drift_results: Dict
    ) -> Tuple[bool, str]:
        """
        Decide if retraining is needed.

        Args:
            performance_metrics: Performance check results
            drift_results: Drift detection results

        Returns:
            (should_retrain, reason)
        """
        reasons = []

        # Check 1: Performance degradation
        if performance_metrics['mae'] > self.baseline_mae_threshold:
            reasons.append(
                f"Performance degradation: MAE {performance_metrics['mae']:.4f} > "
                f"threshold {self.baseline_mae_threshold:.4f}"
            )

        # Check 2: Significant MAE increase
        if performance_metrics['mae_increase_pct'] > 20:
            reasons.append(
                f"MAE increased by {performance_metrics['mae_increase_pct']:.1f}% "
                f"(threshold: 20%)"
            )

        # Check 3: Data drift
        if drift_results['drift_detected']:
            reasons.append(
                f"Data drift in {drift_results['drifted_features_count']} features: "
                f"{drift_results['drifted_features']}"
            )

        should_retrain = len(reasons) > 0

        if should_retrain:
            reason_text = " | ".join(reasons)
            logger.warning(f"RETRAINING TRIGGERED: {reason_text}")
            return True, reason_text
        else:
            logger.info("No retraining needed")
            return False, "All checks passed"


    def retrain_model(
        self,
        training_data: pd.DataFrame,
        version: Optional[str] = None
    ) -> Tuple[XGBRegressor, Dict, str]:
        """
        Retrain model on new data.

        Args:
            training_data: New training data
            version: Model version (auto-generated if None)

        Returns:
            (trained_model, metrics, model_version)
        """
        if version is None:
            version = f"v{len(self.registry['models']) + 1}"

        logger.info(f"=" * 70)
        logger.info(f"RETRAINING MODEL {version}")
        logger.info(f"=" * 70)

        # Feature engineering
        df = add_time_series_features(training_data, verbose=True)

        # Temporal split (80/20)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df['actual_demand']
        X_val = val_df[FEATURE_COLUMNS]
        y_val = val_df['actual_demand']

        logger.info(f"Training set: {len(X_train):,} samples")
        logger.info(f"Validation set: {len(X_val):,} samples")

        # Train model
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

        logger.info("Training XGBoost model...")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Evaluate
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }

        logger.info(f"\n{'='*70}")
        logger.info(f"MODEL {version} VALIDATION METRICS:")
        logger.info(f"{'='*70}")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R²:   {r2:.4f}")
        logger.info(f"{'='*70}")

        return model, metrics, version


    def register_model(
        self,
        model: XGBRegressor,
        version: str,
        metrics: Dict,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Register new model in model registry.

        Args:
            model: Trained model
            version: Model version
            metrics: Validation metrics
            metadata: Additional metadata

        Returns:
            Path to saved model
        """
        # Save model
        model_filename = f"xgb_model_{version}.json"
        model_path = self.models_dir / model_filename

        model.get_booster().save_model(model_path)
        logger.info(f"Model saved: {model_path}")

        # Update registry
        model_info = {
            'version': version,
            'filename': model_filename,
            'created_at': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata or {}
        }

        self.registry['models'].append(model_info)
        self._save_registry()

        return str(model_path)


    def compare_models(
        self,
        model_v1: XGBRegressor,
        model_v2: XGBRegressor,
        test_data: pd.DataFrame,
        v1_name: str = "v1",
        v2_name: str = "v2"
    ) -> Dict:
        """
        A/B test: Compare two models.

        Args:
            model_v1: First model
            model_v2: Second model
            test_data: Test data
            v1_name: Name of first model
            v2_name: Name of second model

        Returns:
            Comparison results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"A/B TESTING: {v1_name} vs {v2_name}")
        logger.info(f"{'='*70}")

        df = add_time_series_features(test_data.copy(), verbose=False)
        X = df[FEATURE_COLUMNS]
        y = df['actual_demand']

        # Predictions
        y_pred_v1 = model_v1.predict(X)
        y_pred_v2 = model_v2.predict(X)

        # Metrics
        mae_v1 = mean_absolute_error(y, y_pred_v1)
        mae_v2 = mean_absolute_error(y, y_pred_v2)
        rmse_v1 = np.sqrt(mean_squared_error(y, y_pred_v1))
        rmse_v2 = np.sqrt(mean_squared_error(y, y_pred_v2))
        r2_v1 = r2_score(y, y_pred_v1)
        r2_v2 = r2_score(y, y_pred_v2)

        # Determine winner
        winner = v2_name if mae_v2 < mae_v1 else v1_name
        improvement_pct = abs((mae_v2 - mae_v1) / mae_v1 * 100)

        results = {
            v1_name: {
                'mae': float(mae_v1),
                'rmse': float(rmse_v1),
                'r2': float(r2_v1)
            },
            v2_name: {
                'mae': float(mae_v2),
                'rmse': float(rmse_v2),
                'r2': float(r2_v2)
            },
            'winner': winner,
            'mae_improvement_pct': float(improvement_pct),
            'test_samples': len(y)
        }

        # Print comparison
        logger.info(f"\n{v1_name:20} {v2_name:20} Winner")
        logger.info("-" * 70)
        logger.info(f"MAE:  {mae_v1:10.4f}      {mae_v2:10.4f}      {'✅' if winner == v2_name else '  '}")
        logger.info(f"RMSE: {rmse_v1:10.4f}      {rmse_v2:10.4f}      {'✅' if winner == v2_name else '  '}")
        logger.info(f"R²:   {r2_v1:10.4f}      {r2_v2:10.4f}      {'✅' if winner == v2_name else '  '}")
        logger.info("=" * 70)
        logger.info(f"🏆 WINNER: {winner} ({improvement_pct:.1f}% {'better' if winner == v2_name else 'worse'})")
        logger.info("=" * 70)

        return results


    def deploy_model(self, model_path: str, version: str):
        """
        Deploy model to production.

        Args:
            model_path: Path to model file
            version: Model version
        """
        production_path = self.models_dir / "xgb_model.json"

        # Backup current production model
        if production_path.exists():
            backup_path = self.models_dir / f"xgb_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            production_path.rename(backup_path)
            logger.info(f"Backed up current model to {backup_path}")

        # Copy new model to production
        import shutil
        shutil.copy(model_path, production_path)

        # Update registry
        self.registry['current_production'] = version
        self._save_registry()

        logger.info(f"✅ Model {version} deployed to production!")


def main():
    """Demo continual learning pipeline."""
    project_root = Path(__file__).parent.parent.parent

    # Initialize pipeline
    pipeline = ContinualLearningPipeline(project_root)

    # Simulate weekly check
    logger.info("\n🔄 WEEKLY CONTINUAL LEARNING CHECK")

    # Load recent production data (e.g., last 7 days)
    prod_data = pipeline.load_production_data(
        start_date="2025-01-08",
        end_date="2025-01-14",
        max_rows=50000
    )

    # Load reference data
    ref_data = pd.read_parquet(
        project_root / "data" / "processed_training_data_2024.parquet"
    ).sample(n=50000, random_state=42)

    # Check performance
    perf_metrics = pipeline.check_performance(prod_data)

    # Check drift
    drift_results = pipeline.check_drift(prod_data, ref_data)

    # Decide if retraining needed
    should_retrain, reason = pipeline.should_retrain(perf_metrics, drift_results)

    if should_retrain:
        logger.info(f"\n⚠️ RETRAINING TRIGGERED: {reason}")

        # Retrain on last 30 days
        retrain_data = pipeline.load_production_data(
            start_date="2025-01-01",
            end_date="2025-01-30",
            max_rows=None
        )

        new_model, metrics, version = pipeline.retrain_model(retrain_data)

        # Register model
        pipeline.register_model(new_model, version, metrics)

        # A/B test (compare with current)
        # ... (would load current model and compare)

        logger.info(f"✅ Model {version} trained and registered!")
    else:
        logger.info("✅ All checks passed - no retraining needed")


if __name__ == "__main__":
    main()
