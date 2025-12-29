"""
Shared feature engineering utilities for NYC Taxi Pulse.
Contains reusable functions for time-series feature generation.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Manhattan location IDs (major zones)
MANHATTAN_ZONES = [
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90,
    100, 103, 104, 105, 107, 113, 114, 116, 120, 125, 127, 128, 137, 140,
    141, 142, 143, 144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 166,
    170, 186, 194, 202, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236,
    237, 238, 239, 243, 244, 246, 249, 261, 262, 263
]


def add_time_series_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add time-series features for demand prediction.
    
    Generates lag features and rolling features at different time intervals per location:
    - Lag features: Previous demand values (15 mins, 1 hour, 24 hours ago)
    - Rolling features: Moving averages over time windows
    
    Args:
        df: DataFrame with pickup_window, PULocationID, and actual_demand
        verbose: Whether to print detailed progress information
    
    Returns:
        DataFrame with time-series features added
    """
    try:
        if verbose:
            print("\n" + "=" * 70)
            print("TIME-SERIES FEATURE ENGINEERING")
            print("=" * 70)
        
        # Step 1: Ensure data is sorted by location and time for proper temporal ordering
        if verbose:
            print("\nStep 1: Sorting data by PULocationID and pickup_window...")
        df = df.sort_values(['PULocationID', 'pickup_window']).reset_index(drop=True)
        if verbose:
            print(f"  ✓ Data sorted. Shape: {df.shape}")
        
        rows_before = len(df)
        
        # Step 2: Temporal Features (before lag features)
        if verbose:
            print("\nStep 2: Creating temporal features...")

        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df['day_of_month'] = df['pickup_window'].dt.day
        df['week_of_year'] = df['pickup_window'].dt.isocalendar().week.astype(int)

        # Time of day (0: night, 1: morning, 2: afternoon, 3: evening)
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[-1, 6, 12, 18, 24],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # Season (0: winter, 1: spring, 2: summer, 3: fall)
        df['season'] = ((df['month'] % 12) // 3).astype(int)

        # Location features
        df['is_manhattan'] = df['PULocationID'].isin(MANHATTAN_ZONES).astype(int)

        if verbose:
            print("  ✓ Temporal features created (9 features)")

        # Step 3: Create LAG features per PULocationID
        if verbose:
            print("\nStep 3: Creating lag features per location...")

        df['lag_1'] = df.groupby('PULocationID')['actual_demand'].shift(1)
        df['lag_4'] = df.groupby('PULocationID')['actual_demand'].shift(4)
        df['lag_24'] = df.groupby('PULocationID')['actual_demand'].shift(24)   # 6 hours
        df['lag_96'] = df.groupby('PULocationID')['actual_demand'].shift(96)   # 24 hours
        df['lag_672'] = df.groupby('PULocationID')['actual_demand'].shift(672) # 1 week

        if verbose:
            print("  ✓ Lag features created (5 lags)")

        # Step 4: Create ROLLING features per PULocationID
        if verbose:
            print("\nStep 4: Creating rolling statistics per location...")

        grouped = df.groupby('PULocationID')['actual_demand']

        df['rolling_mean_4'] = grouped.transform(
            lambda x: x.rolling(window=4, min_periods=4).mean()
        )
        df['rolling_std_4'] = grouped.transform(
            lambda x: x.rolling(window=4, min_periods=4).std()
        )
        df['rolling_max_24'] = grouped.transform(
            lambda x: x.rolling(window=24, min_periods=24).max()
        )
        df['rolling_min_24'] = grouped.transform(
            lambda x: x.rolling(window=24, min_periods=24).min()
        )

        if verbose:
            print("  ✓ Rolling statistics created (4 features)")
        
        # Step 5: Handle NaNs - Drop rows with any NaN values from feature engineering
        if verbose:
            print("\nStep 5: Handling NaNs from feature engineering...")
        feature_cols = [
            'lag_1', 'lag_4', 'lag_24', 'lag_96', 'lag_672',
            'rolling_mean_4', 'rolling_std_4', 'rolling_max_24', 'rolling_min_24'
        ]
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
        
        rows_after = len(df)
        rows_dropped = rows_before - rows_after
        
        if verbose:
            print(f"  ✓ Rows before: {rows_before:,}")
            print(f"  ✓ Rows after:  {rows_after:,}")
            print(f"  ✓ Dropped:     {rows_dropped:,} ({rows_dropped/rows_before*100:.2f}%)")
            print(f"\n📊 Total Features Created: 18")
            print(f"  • Temporal: 9 (weekend, rush_hour, time_of_day, etc.)")
            print(f"  • Lag: 5 (1, 4, 24, 96, 672 periods)")
            print(f"  • Rolling: 4 (mean, std, max, min)")
            print("=" * 70)
        
        logger.info(f"Time-series features added. Final shape: {df.shape}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error adding time-series features: {str(e)}")
        raise

