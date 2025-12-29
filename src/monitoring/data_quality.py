"""
Data quality validation for production inference data.
Ensures input data meets quality standards before prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    Validates data quality for taxi demand prediction.

    Checks for:
    - Missing values
    - Out-of-range values
    - Invalid data types
    - Statistical anomalies
    """

    def __init__(self):
        """Initialize data quality checker with validation rules."""
        # Define expected ranges for features
        self.feature_ranges = {
            'PULocationID': (1, 263),
            'hour': (0, 23),
            'day_of_week': (0, 6),
            'month': (1, 12),
            'lag_1': (0, float('inf')),
            'lag_4': (0, float('inf')),
            'lag_96': (0, float('inf')),
            'rolling_mean_4': (0, float('inf'))
        }

        # Expected data types
        self.expected_dtypes = {
            'PULocationID': 'int',
            'hour': 'int',
            'day_of_week': 'int',
            'month': 'int',
            'lag_1': 'float',
            'lag_4': 'float',
            'lag_96': 'float',
            'rolling_mean_4': 'float'
        }

        # Statistical thresholds (based on training data statistics)
        # These should be updated with actual training data statistics
        self.statistical_thresholds = {
            'lag_1': {'mean': 15.0, 'std': 10.0, 'max_z_score': 5.0},
            'lag_4': {'mean': 15.5, 'std': 10.5, 'max_z_score': 5.0},
            'lag_96': {'mean': 16.0, 'std': 11.0, 'max_z_score': 5.0},
            'rolling_mean_4': {'mean': 15.5, 'std': 10.0, 'max_z_score': 5.0}
        }


    def validate_single_record(
        self,
        record: Dict
    ) -> Tuple[bool, List[str]]:
        """
        Validate a single prediction record.

        Args:
            record: Dictionary with feature values

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check for missing features
        required_features = set(self.feature_ranges.keys())
        provided_features = set(record.keys())
        missing = required_features - provided_features

        if missing:
            errors.append(f"Missing features: {missing}")
            return False, errors

        # Check for extra features (warning, not error)
        extra = provided_features - required_features
        if extra:
            logger.warning(f"Extra features provided (will be ignored): {extra}")

        # Validate each feature
        for feature, value in record.items():
            if feature not in self.feature_ranges:
                continue

            # Check for None/NaN
            if value is None or (isinstance(value, float) and np.isnan(value)):
                errors.append(f"{feature}: Value is None or NaN")
                continue

            # Check range
            min_val, max_val = self.feature_ranges[feature]
            if not (min_val <= value <= max_val):
                errors.append(
                    f"{feature}: Value {value} out of range [{min_val}, {max_val}]"
                )

            # Check statistical anomalies for continuous features
            if feature in self.statistical_thresholds:
                stats = self.statistical_thresholds[feature]
                z_score = abs((value - stats['mean']) / stats['std'])

                if z_score > stats['max_z_score']:
                    errors.append(
                        f"{feature}: Statistical anomaly detected (z-score: {z_score:.2f})"
                    )

        is_valid = len(errors) == 0
        return is_valid, errors


    def validate_batch(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate a batch of records.

        Args:
            df: DataFrame with prediction features

        Returns:
            Tuple of (valid_df, validation_report)
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'valid_records': 0,
            'invalid_records': 0,
            'errors': [],
            'warnings': []
        }

        valid_indices = []
        invalid_indices = []

        # Validate each row
        for idx, row in df.iterrows():
            record = row.to_dict()
            is_valid, errors = self.validate_single_record(record)

            if is_valid:
                valid_indices.append(idx)
            else:
                invalid_indices.append(idx)
                report['errors'].append({
                    'index': idx,
                    'errors': errors
                })

        # Create valid DataFrame
        valid_df = df.loc[valid_indices].copy()

        report['valid_records'] = len(valid_indices)
        report['invalid_records'] = len(invalid_indices)
        report['validation_rate'] = len(valid_indices) / len(df) * 100

        # Log summary
        if report['invalid_records'] > 0:
            logger.warning(
                f"Data quality check: {report['invalid_records']}/{report['total_records']} "
                f"records failed validation ({report['validation_rate']:.1f}% pass rate)"
            )
        else:
            logger.info(
                f"Data quality check: All {report['total_records']} records passed validation"
            )

        return valid_df, report


    def check_data_consistency(
        self,
        df: pd.DataFrame
    ) -> Dict:
        """
        Check for data consistency issues.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with consistency check results
        """
        issues = []

        # Check 1: Lag features should be ordered (lag_1 <= lag_4 <= lag_96 approximately)
        # In reality, demand can vary, so we check for extreme inconsistencies
        if 'lag_1' in df.columns and 'lag_96' in df.columns:
            extreme_diff = df['lag_1'] > (df['lag_96'] * 3)
            if extreme_diff.any():
                issues.append(
                    f"Extreme lag inconsistency: {extreme_diff.sum()} records where "
                    f"lag_1 > 3 * lag_96"
                )

        # Check 2: Rolling mean should be close to lag features
        if all(col in df.columns for col in ['rolling_mean_4', 'lag_1', 'lag_4']):
            expected_mean = (df['lag_1'] + df['lag_4']) / 2
            mean_diff = abs(df['rolling_mean_4'] - expected_mean)
            large_diff = mean_diff > 50  # Threshold

            if large_diff.any():
                issues.append(
                    f"Rolling mean inconsistency: {large_diff.sum()} records with "
                    f"large difference between rolling_mean_4 and lag features"
                )

        # Check 3: Temporal consistency (if timestamp available)
        if 'pickup_window' in df.columns:
            df_sorted = df.sort_values('pickup_window')
            time_diffs = df_sorted['pickup_window'].diff()

            # Check for duplicates
            duplicates = df.duplicated(subset=['pickup_window', 'PULocationID'])
            if duplicates.any():
                issues.append(
                    f"Duplicate records: {duplicates.sum()} records with same "
                    f"pickup_window and location"
                )

        return {
            'timestamp': datetime.now().isoformat(),
            'consistency_check': 'failed' if issues else 'passed',
            'issues': issues
        }


    def generate_quality_report(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive data quality report.

        Args:
            df: DataFrame to analyze
            save_path: Optional path to save report

        Returns:
            Quality report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(df),
            'features': list(df.columns),
            'summary_statistics': {},
            'missing_values': {},
            'outliers': {},
            'data_types': {}
        }

        # Summary statistics
        for col in df.columns:
            if col in self.feature_ranges:
                report['summary_statistics'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median())
                }

                # Missing values
                missing = df[col].isna().sum()
                report['missing_values'][col] = {
                    'count': int(missing),
                    'percentage': float(missing / len(df) * 100)
                }

                # Outliers (using IQR method)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()

                report['outliers'][col] = {
                    'count': int(outliers),
                    'percentage': float(outliers / len(df) * 100)
                }

                # Data types
                report['data_types'][col] = str(df[col].dtype)

        # Save report if path provided
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Quality report saved to: {save_path}")

        return report
