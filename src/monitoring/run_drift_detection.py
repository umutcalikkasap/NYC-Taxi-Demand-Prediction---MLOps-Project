"""
Script to run drift detection on production data.
Compares recent production data against training reference data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.drift_detector import DriftDetector, load_reference_data
from features import add_time_series_features
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    'PULocationID', 'hour', 'day_of_week', 'month',
    'is_weekend', 'is_rush_hour', 'day_of_month', 'week_of_year',
    'time_of_day', 'season', 'is_manhattan',
    'lag_1', 'lag_4', 'lag_24', 'lag_96', 'lag_672',
    'rolling_mean_4', 'rolling_std_4', 'rolling_max_24', 'rolling_min_24'
]


def load_production_data(
    data_path: Path,
    days_back: int = 7,
    sample_size: int = 10000
) -> pd.DataFrame:
    """
    Load recent production/inference data.

    Args:
        data_path: Path to production data
        days_back: Number of days to look back
        sample_size: Number of samples to use

    Returns:
        DataFrame with production data
    """
    logger.info(f"Loading production data from {data_path}")

    df = pd.read_parquet(data_path, engine='pyarrow')

    # Filter recent data if timestamp available (but keep more for lag calculation)
    if 'pickup_window' in df.columns:
        df['pickup_window'] = pd.to_datetime(df['pickup_window'])
        # Keep extra days for lag feature calculation
        cutoff_date = df['pickup_window'].max() - timedelta(days=days_back + 30)
        df = df[df['pickup_window'] >= cutoff_date]
        logger.info(f"Loaded last {days_back + 30} days for lag calculation: {len(df):,} records")

    # Apply feature engineering FIRST (needs historical data for lags)
    logger.info("Applying feature engineering to production data...")
    df = add_time_series_features(df, verbose=False)

    # Drop NaN rows (from lag calculation)
    df = df.dropna()
    logger.info(f"After feature engineering: {len(df):,} records")

    # Now filter to actual time window
    if 'pickup_window' in df.columns:
        cutoff_date = df['pickup_window'].max() - timedelta(days=days_back)
        df = df[df['pickup_window'] >= cutoff_date]
        logger.info(f"Filtered to last {days_back} days: {len(df):,} records")

    # Sample if too large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size:,} records")

    return df[FEATURE_COLUMNS].copy()


def main():
    """Main drift detection pipeline."""
    parser = argparse.ArgumentParser(description='Run drift detection on production data')
    parser.add_argument(
        '--reference-data',
        type=str,
        default='data/processed_training_data_2024.parquet',
        help='Path to reference (training) data'
    )
    parser.add_argument(
        '--production-data',
        type=str,
        default='data/processed_inference_data_2025.parquet',
        help='Path to production/inference data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='monitoring/drift_reports',
        help='Directory to save drift reports'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=7,
        help='Number of days of production data to analyze'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.05,
        help='P-value threshold for KS test'
    )

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    reference_path = project_root / args.reference_data
    production_path = project_root / args.production_data
    output_dir = project_root / args.output_dir

    logger.info("=" * 70)
    logger.info("DRIFT DETECTION PIPELINE")
    logger.info("=" * 70)

    # Load reference data
    logger.info("\n1. Loading reference (training) data...")
    reference_df_raw = pd.read_parquet(reference_path, engine='pyarrow')
    logger.info(f"   Loaded {len(reference_df_raw):,} raw records")

    # Apply feature engineering FIRST (needs history for lags)
    logger.info("   Applying feature engineering to reference data...")
    reference_df = add_time_series_features(reference_df_raw, verbose=False)

    # Drop NaN rows
    reference_df = reference_df.dropna()
    logger.info(f"   After feature engineering: {len(reference_df):,} records")

    # Sample if too large
    if len(reference_df) > 100000:
        reference_df = reference_df.sample(n=100000, random_state=42)
        logger.info(f"   Sampled 100,000 rows for reference distribution")

    reference_df = reference_df[FEATURE_COLUMNS].copy()
    logger.info(f"   Reference data ready: {len(reference_df):,} records")

    # Load production data
    logger.info("\n2. Loading production data...")
    production_df = load_production_data(
        production_path,
        days_back=args.days_back
    )
    logger.info(f"   Production data: {len(production_df):,} records")

    # Initialize drift detector
    logger.info("\n3. Initializing drift detector...")
    detector = DriftDetector(
        reference_data=reference_df,
        feature_columns=FEATURE_COLUMNS,
        categorical_features=['PULocationID'],
        drift_threshold=args.threshold
    )

    # Generate drift report
    logger.info("\n4. Running drift detection...")
    output_path = output_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # For demonstration, we'll also generate mock predictions and actuals
    # In production, these would come from logged predictions
    # Use a baseline value for mock data
    baseline_value = production_df[FEATURE_COLUMNS].mean().mean()
    mock_predictions = np.ones(len(production_df)) * baseline_value + np.random.normal(0, 2, len(production_df))
    mock_actuals = np.ones(len(production_df)) * baseline_value + np.random.normal(0, 1, len(production_df))

    report = detector.generate_drift_report(
        production_data=production_df,
        predictions=mock_predictions,
        actuals=mock_actuals,
        output_path=output_path
    )

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("DRIFT DETECTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Drift Detected: {report['summary']['drift_detected']}")
    logger.info(f"Features Analyzed: {report['summary']['total_features']}")
    logger.info(f"Features with Drift: {report['summary']['drifted_features']}")

    if report['summary']['drifted_features'] > 0:
        logger.warning(f"\nDrifted Features: {', '.join(report['summary']['drifted_feature_names'])}")

        logger.info("\nDrift Details:")
        for feature in report['summary']['drifted_feature_names']:
            drift_info = report['feature_drift'][feature]
            logger.warning(
                f"  {feature}:"
                f"\n    Test: {drift_info['test']}"
                f"\n    Drift Score: {drift_info['drift_score']:.4f}"
                f"\n    Reference Mean: {drift_info['reference_mean']:.2f}"
                f"\n    Production Mean: {drift_info['production_mean']:.2f}"
                f"\n    Mean Shift: {drift_info['mean_shift']:.2f}"
            )
    else:
        logger.info("\n✓ No significant drift detected in any features")

    # Print prediction performance
    if 'performance' in report['prediction_drift']:
        perf = report['prediction_drift']['performance']
        logger.info("\nPrediction Performance (on sampled data):")
        logger.info(f"  MAE:  {perf['mae']:.4f}")
        logger.info(f"  RMSE: {perf['rmse']:.4f}")
        logger.info(f"  MAPE: {perf['mape']:.2f}%")
        logger.info(f"  Bias: {perf['bias']:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info(f"Report saved to: {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
