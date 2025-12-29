"""
Model drift detection using statistical tests.
Monitors distribution changes in features and predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects data drift and concept drift in production.

    Uses Kolmogorov-Smirnov test for continuous features
    and Population Stability Index (PSI) for categorical features.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: List[str],
        categorical_features: List[str] = None,
        drift_threshold: float = 0.05
    ):
        """
        Initialize drift detector with reference (training) data.

        Args:
            reference_data: Training data as reference distribution
            feature_columns: List of feature column names
            categorical_features: List of categorical feature names
            drift_threshold: P-value threshold for KS test (default: 0.05)
        """
        self.reference_data = reference_data[feature_columns].copy()
        self.feature_columns = feature_columns
        self.categorical_features = categorical_features or ['PULocationID']
        self.drift_threshold = drift_threshold

        # Store reference statistics
        self.reference_stats = self._compute_statistics(self.reference_data)

        logger.info(f"DriftDetector initialized with {len(feature_columns)} features")


    def _compute_statistics(self, data: pd.DataFrame) -> Dict:
        """Compute statistics for reference data."""
        stats_dict = {}

        for col in self.feature_columns:
            stats_dict[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'median': data[col].median(),
                'q25': data[col].quantile(0.25),
                'q75': data[col].quantile(0.75)
            }

        return stats_dict


    def detect_feature_drift(
        self,
        production_data: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Detect drift in feature distributions.

        Args:
            production_data: Recent production data

        Returns:
            Dictionary with drift detection results per feature
        """
        drift_results = {}

        for feature in self.feature_columns:
            if feature in self.categorical_features:
                # Use PSI for categorical features
                drift_score = self._calculate_psi(
                    self.reference_data[feature],
                    production_data[feature]
                )
                drift_detected = drift_score > 0.1  # PSI threshold
                test_name = "PSI"
                p_value = None

            else:
                # Use KS test for continuous features
                ks_statistic, p_value = stats.ks_2samp(
                    self.reference_data[feature],
                    production_data[feature]
                )
                drift_score = ks_statistic
                drift_detected = p_value < self.drift_threshold
                test_name = "KS"

            # Compute distribution statistics
            prod_stats = {
                'mean': production_data[feature].mean(),
                'std': production_data[feature].std(),
                'median': production_data[feature].median()
            }

            ref_stats = self.reference_stats[feature]

            drift_results[feature] = {
                'drift_detected': bool(drift_detected),
                'test': test_name,
                'drift_score': float(drift_score),
                'p_value': float(p_value) if p_value else None,
                'reference_mean': float(ref_stats['mean']),
                'production_mean': float(prod_stats['mean']),
                'mean_shift': float(prod_stats['mean'] - ref_stats['mean']),
                'reference_std': float(ref_stats['std']),
                'production_std': float(prod_stats['std'])
            }

        return drift_results


    def _calculate_psi(
        self,
        reference: pd.Series,
        production: pd.Series,
        buckets: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI < 0.1: No significant change
        PSI 0.1-0.2: Moderate change
        PSI > 0.2: Significant change

        Args:
            reference: Reference distribution
            production: Production distribution
            buckets: Number of buckets for binning

        Returns:
            PSI score
        """
        # Create bins from reference data
        if reference.dtype in ['int64', 'int32'] and reference.nunique() < buckets:
            # For categorical-like integer features
            bins = sorted(reference.unique())
        else:
            # For continuous features
            _, bins = pd.qcut(reference, q=buckets, retbins=True, duplicates='drop')

        # Calculate distributions
        ref_dist = pd.cut(reference, bins=bins, include_lowest=True).value_counts(normalize=True)
        prod_dist = pd.cut(production, bins=bins, include_lowest=True).value_counts(normalize=True)

        # Align distributions
        ref_dist, prod_dist = ref_dist.align(prod_dist, fill_value=0.0001)

        # Calculate PSI
        psi = np.sum((prod_dist - ref_dist) * np.log(prod_dist / ref_dist))

        return psi


    def detect_prediction_drift(
        self,
        predictions: np.ndarray,
        actuals: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Detect drift in prediction distribution and performance.

        Args:
            predictions: Recent predictions
            actuals: Actual values (if available)

        Returns:
            Dictionary with prediction drift metrics
        """
        results = {
            'prediction_stats': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'median': float(np.median(predictions))
            }
        }

        # If actuals available, compute performance metrics
        if actuals is not None:
            errors = predictions - actuals

            results['performance'] = {
                'mae': float(np.mean(np.abs(errors))),
                'rmse': float(np.sqrt(np.mean(errors ** 2))),
                'mape': float(np.mean(np.abs(errors / (actuals + 1e-8))) * 100),
                'bias': float(np.mean(errors))
            }

        return results


    def generate_drift_report(
        self,
        production_data: pd.DataFrame,
        predictions: np.ndarray,
        actuals: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None
    ) -> Dict:
        """
        Generate comprehensive drift detection report.

        Args:
            production_data: Recent production feature data
            predictions: Recent predictions
            actuals: Actual values (if available)
            output_path: Path to save report JSON

        Returns:
            Complete drift report dictionary
        """
        logger.info("Generating drift detection report...")

        # Feature drift
        feature_drift = self.detect_feature_drift(production_data)

        # Prediction drift
        prediction_drift = self.detect_prediction_drift(predictions, actuals)

        # Summary
        drifted_features = [
            feat for feat, results in feature_drift.items()
            if results['drift_detected']
        ]

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_features': len(self.feature_columns),
                'drifted_features': len(drifted_features),
                'drift_detected': len(drifted_features) > 0,
                'drifted_feature_names': drifted_features
            },
            'feature_drift': feature_drift,
            'prediction_drift': prediction_drift,
            'production_data_size': len(production_data)
        }

        # Save report if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Drift report saved to: {output_path}")

        # Log summary
        if report['summary']['drift_detected']:
            logger.warning(
                f"DRIFT DETECTED in {len(drifted_features)} features: "
                f"{', '.join(drifted_features)}"
            )
        else:
            logger.info("No significant drift detected")

        return report


def load_reference_data(data_path: Path, feature_columns: List[str]) -> pd.DataFrame:
    """
    Load reference (training) data for drift detection.

    Args:
        data_path: Path to training data parquet
        feature_columns: Feature columns to load

    Returns:
        DataFrame with reference data
    """
    logger.info(f"Loading reference data from {data_path}")

    # Load training data
    df = pd.read_parquet(data_path, engine='pyarrow')

    # Sample if too large (for efficiency)
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=42)
        logger.info(f"Sampled 100,000 rows for reference distribution")

    return df[feature_columns].copy()
