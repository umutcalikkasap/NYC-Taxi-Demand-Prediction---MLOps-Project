"""
Aggregator module for NYC Yellow Taxi data.
Aggregates trip data into 15-minute demand windows per PULocationID.
Includes data cleaning to filter out dirty records from incorrect years/months.
"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_month_from_filename(file_path: Path) -> int:
    """
    Extract expected month number from filename.
    
    Args:
        file_path: Path to parquet file (e.g., yellow_tripdata_2024-01.parquet)
    
    Returns:
        Month number (1-12)
    
    Raises:
        ValueError: If month cannot be extracted
    """
    try:
        # Extract month from filename pattern: yellow_tripdata_YYYY-MM.parquet
        match = re.search(r'_(\d{4})-(\d{2})\.parquet', file_path.name)
        if match:
            return int(match.group(2))
        else:
            raise ValueError(f"Cannot extract month from filename: {file_path.name}")
    except Exception as e:
        logger.error(f"Error extracting month from filename: {str(e)}")
        raise


def process_single_file(file_path: Path, target_year: int = 2024) -> Tuple[pd.DataFrame, int, int]:
    """
    Process a single parquet file with robust data cleaning and aggregation.
    
    Applies multiple quality filters:
    - Date filtering (year and month validation)
    - Distance filtering (trip_distance > 0)
    - Duration filtering (positive and realistic trip durations)
    
    Args:
        file_path: Path to the parquet file
        target_year: Year to filter for (default: 2024)
    
    Returns:
        Tuple of (aggregated_df, original_rows, clean_rows)
    
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: For other processing errors
    """
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract expected month from filename
        expected_month = extract_month_from_filename(file_path)
        
        # Load required columns including quality validation fields
        columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'trip_distance']
        df = pd.read_parquet(file_path, columns=columns, engine='pyarrow')
        
        original_rows = len(df)
        
        # Ensure datetime format
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        
        # STEP 1: DATE FILTERING
        # Filter to keep ONLY records from target year (2024)
        df = df[df['tpep_pickup_datetime'].dt.year == target_year].copy()
        
        # Additional filter: Keep ONLY records matching expected month
        df = df[df['tpep_pickup_datetime'].dt.month == expected_month].copy()
        
        rows_after_date_filter = len(df)
        date_filtered = original_rows - rows_after_date_filter
        
        # STEP 2: QUALITY FILTERING (Distance & Duration)
        if rows_after_date_filter > 0:
            # Distance Rule: Keep rows where trip_distance > 0
            df = df[df['trip_distance'] > 0].copy()
            
            # Duration Rule: Calculate trip duration in minutes
            df['duration_minutes'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
            
            # Keep rows where duration is positive (> 1 minute) and realistic (< 5 hours = 300 minutes)
            df = df[(df['duration_minutes'] > 1) & (df['duration_minutes'] < 300)].copy()
            
            rows_after_quality_filter = len(df)
            quality_filtered = rows_after_date_filter - rows_after_quality_filter
        else:
            rows_after_quality_filter = 0
            quality_filtered = 0
        
        clean_rows = rows_after_quality_filter
        total_dropped = original_rows - clean_rows
        
        # Print detailed data cleaning stats
        print(f"  📁 {file_path.name}")
        print(f"     ├─ Original rows:          {original_rows:,}")
        
        if date_filtered > 0:
            print(f"     ├─ Date Filtering:         -{date_filtered:,} ({date_filtered/original_rows*100:.2f}%)")
        
        if quality_filtered > 0:
            print(f"     ├─ Quality Filtering:      -{quality_filtered:,} ({quality_filtered/original_rows*100:.2f}%)")
            print(f"     │   └─ (distance & duration rules)")
        
        print(f"     └─ ✓ Clean rows:           {clean_rows:,} ({clean_rows/original_rows*100:.1f}%)")
        
        # If no data left after filtering, return empty aggregated DataFrame
        if clean_rows == 0:
            logger.warning(f"No valid data in {file_path.name} after filtering")
            return pd.DataFrame(columns=['pickup_window', 'PULocationID', 'actual_demand', 'hour', 'day_of_week', 'month']), original_rows, clean_rows
        
        # Clean up: Drop extra columns before aggregation to save memory
        df = df.drop(columns=['tpep_dropoff_datetime', 'trip_distance', 'duration_minutes'])
        
        # Create pickup_window by flooring to 15-minute intervals
        df['pickup_window'] = df['tpep_pickup_datetime'].dt.floor('15min')
        
        # Group by pickup_window and PULocationID to count trips
        aggregated = df.groupby(
            ['pickup_window', 'PULocationID']
        ).size().reset_index(name='actual_demand')
        
        # Extract time-based features from pickup_window
        aggregated['hour'] = aggregated['pickup_window'].dt.hour
        aggregated['day_of_week'] = aggregated['pickup_window'].dt.dayofweek
        aggregated['month'] = aggregated['pickup_window'].dt.month
        
        return aggregated, original_rows, clean_rows
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        raise


def aggregate_training_data(
    training_dir: Path,
    output_path: Path,
    target_year: int = 2024
) -> pd.DataFrame:
    """
    Aggregate all 2024 training batch files into a single processed dataset.
    Includes data cleaning to remove dirty records from incorrect years/months.
    
    Args:
        training_dir: Directory containing training parquet files (data/training_batch/)
        output_path: Path to save final aggregated data (data/processed_training_data_2024.parquet)
        target_year: Year to filter for (default: 2024)
    
    Returns:
        Final aggregated DataFrame
    
    Raises:
        ValueError: If no parquet files found
        Exception: For processing errors
    """
    try:
        # Get all parquet files in training directory
        parquet_files = sorted(training_dir.glob("*.parquet"))
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in {training_dir}")
        
        print("\n" + "=" * 70)
        print(f"DATA CLEANING & AGGREGATION PIPELINE")
        print("=" * 70)
        print(f"Found {len(parquet_files)} training files to process")
        print(f"Target year: {target_year}")
        print("=" * 70)
        print("\nProcessing files and cleaning data:\n")
        
        # Process each file with data cleaning
        aggregated_chunks: List[pd.DataFrame] = []
        total_original_rows = 0
        total_clean_rows = 0
        
        for file_path in parquet_files:
            chunk, original_rows, clean_rows = process_single_file(file_path, target_year)
            if len(chunk) > 0:
                aggregated_chunks.append(chunk)
            total_original_rows += original_rows
            total_clean_rows += clean_rows
        
        total_dirty_rows = total_original_rows - total_clean_rows
        
        print("\n" + "=" * 70)
        print("DATA CLEANING SUMMARY:")
        print("=" * 70)
        print(f"  Total original rows:        {total_original_rows:,}")
        print(f"  Total clean rows:           {total_clean_rows:,}")
        print(f"  Total rows dropped:         {total_dirty_rows:,} ({total_dirty_rows/total_original_rows*100:.2f}%)")
        print(f"  Data quality pass rate:     {total_clean_rows/total_original_rows*100:.2f}%")
        print("=" * 70)
        print("Quality Filters Applied:")
        print("  ✓ Date: Year = 2024, Month matches filename")
        print("  ✓ Distance: trip_distance > 0")
        print("  ✓ Duration: 1 min < trip duration < 5 hours")
        print("=" * 70)
        
        # Concatenate all chunks into a single DataFrame
        print("\nConcatenating all aggregated chunks...")
        final_df = pd.concat(aggregated_chunks, ignore_index=True)
        
        # Group again in case same pickup_window + PULocationID appear in multiple months
        print("Final aggregation across all months...")
        final_df = final_df.groupby(
            ['pickup_window', 'PULocationID', 'hour', 'day_of_week', 'month']
        )['actual_demand'].sum().reset_index()
        
        # Sort by pickup_window for better organization
        final_df = final_df.sort_values(['pickup_window', 'PULocationID']).reset_index(drop=True)
        
        print("\n" + "=" * 70)
        print("FINAL DATASET STATISTICS:")
        print("=" * 70)
        print(f"  Shape: {final_df.shape}")
        print(f"  Date range: {final_df['pickup_window'].min()} to {final_df['pickup_window'].max()}")
        print(f"  Unique locations: {final_df['PULocationID'].nunique()}")
        print(f"  Total aggregated records: {len(final_df):,}")
        print("=" * 70)
        
        # Save to parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_path, index=False, engine='pyarrow')
        print(f"\n✓ Saved processed data to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        print("=" * 70 + "\n")
        
        return final_df
    
    except Exception as e:
        logger.error(f"Error aggregating training data: {str(e)}")
        raise


def main() -> None:
    """
    Main function to run the aggregation pipeline.
    """
    try:
        # Define paths
        project_root = Path(__file__).parent.parent.parent
        training_dir = project_root / "data" / "training_batch"
        output_path = project_root / "data" / "processed_training_data_2024.parquet"
        
        # Run aggregation with data cleaning
        result_df = aggregate_training_data(training_dir, output_path, target_year=2024)
        
        print("Pipeline completed successfully! ✓\n")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

