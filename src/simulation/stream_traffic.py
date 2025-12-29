"""
Streaming simulation for NYC Taxi demand prediction.
Simulates real-time data streaming by sending requests to the FastAPI endpoint.
"""

import pandas as pd
from pathlib import Path
import requests
import time
import logging
from typing import Dict, Any
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrafficStreamSimulator:
    """
    Simulates real-time traffic streaming by sending predictions to API.
    """
    
    def __init__(
        self,
        data_path: Path,
        api_url: str = "http://localhost:8000",
        delay: float = 0.1
    ):
        """
        Initialize the traffic stream simulator.
        
        Args:
            data_path: Path to processed inference data
            api_url: Base URL of the prediction API
            delay: Delay between requests in seconds (default: 0.1)
        """
        self.data_path = data_path
        self.api_url = api_url
        self.predict_endpoint = f"{api_url}/predict"
        self.health_endpoint = f"{api_url}/health"
        self.delay = delay
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_mae = 0.0
        self.total_squared_error = 0.0
        
    def check_api_health(self) -> bool:
        """
        Check if the API is running and healthy.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('model_loaded', False):
                    logger.info("✓ API is healthy and model is loaded")
                    return True
                else:
                    logger.error("✗ API is running but model is not loaded")
                    return False
            else:
                logger.error(f"✗ API health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Cannot connect to API: {str(e)}")
            logger.error(f"  Make sure the API is running at {self.api_url}")
            return False
    
    def load_inference_data(self) -> pd.DataFrame:
        """
        Load and prepare inference data for streaming.
        
        Returns:
            Sorted DataFrame ready for streaming
        
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            logger.info(f"Loading inference data from {self.data_path}")
            df = pd.read_parquet(self.data_path, engine='pyarrow')
            
            # Sort by pickup_window to simulate chronological streaming
            df = df.sort_values('pickup_window').reset_index(drop=True)
            
            logger.info(f"✓ Loaded {len(df):,} records")
            logger.info(f"  Date range: {df['pickup_window'].min()} to {df['pickup_window'].max()}")
            logger.info(f"  Unique locations: {df['PULocationID'].nunique()}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading inference data: {str(e)}")
            raise
    
    def prepare_prediction_input(self, row: pd.Series) -> Dict[str, Any]:
        """
        Prepare a single row for API request.
        
        Args:
            row: DataFrame row with all features
        
        Returns:
            Dictionary formatted for API request
        """
        return {
            'PULocationID': int(row['PULocationID']),
            'hour': int(row['hour']),
            'day_of_week': int(row['day_of_week']),
            'month': int(row['month']),
            'lag_1': float(row['lag_1']),
            'lag_4': float(row['lag_4']),
            'lag_96': float(row['lag_96']),
            'rolling_mean_4': float(row['rolling_mean_4'])
        }
    
    def send_prediction_request(self, input_data: Dict[str, Any]) -> float:
        """
        Send prediction request to API.
        
        Args:
            input_data: Input features for prediction
        
        Returns:
            Predicted demand value
        
        Raises:
            requests.exceptions.RequestException: If request fails
        """
        try:
            response = requests.post(
                self.predict_endpoint,
                json=input_data,
                timeout=5
            )
            response.raise_for_status()
            
            result = response.json()
            return result['predicted_demand']
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Prediction request failed: {str(e)}")
            raise
    
    def calculate_metrics(self, actual: float, predicted: float):
        """
        Update running metrics.
        
        Args:
            actual: Actual demand value
            predicted: Predicted demand value
        """
        error = actual - predicted
        self.total_mae += abs(error)
        self.total_squared_error += error ** 2
    
    def stream_data(self, max_records: int = None, batch_size: int = 1):
        """
        Stream inference data and send predictions to API.
        
        Args:
            max_records: Maximum number of records to stream (None for all)
            batch_size: Number of records to process per time window (default: 1)
        """
        try:
            # Load data
            df = self.load_inference_data()
            
            # Limit records if specified
            if max_records is not None:
                df = df.head(max_records)
                logger.info(f"Limiting to first {max_records:,} records")
            
            print("\n" + "=" * 90)
            print("STARTING REAL-TIME TRAFFIC STREAM SIMULATION")
            print("=" * 90)
            print(f"API Endpoint: {self.predict_endpoint}")
            print(f"Total Records: {len(df):,}")
            print(f"Delay: {self.delay}s between requests")
            print("=" * 90)
            print("\n{:<20} | {:<8} | {:<10} | {:<10} | {:<10}".format(
                "Time", "Location", "Actual", "Predicted", "Error"
            ))
            print("-" * 90)
            
            start_time = time.time()
            
            # Stream data row by row
            for idx, row in df.iterrows():
                try:
                    # Prepare input
                    input_data = self.prepare_prediction_input(row)
                    
                    # Send prediction request
                    predicted = self.send_prediction_request(input_data)
                    
                    # Get actual value
                    actual = float(row['actual_demand'])
                    
                    # Calculate error
                    error = actual - predicted
                    
                    # Update statistics
                    self.total_requests += 1
                    self.successful_requests += 1
                    self.calculate_metrics(actual, predicted)
                    
                    # Print log line
                    time_str = str(row['pickup_window'])[:16]  # Truncate to minute
                    location = int(row['PULocationID'])
                    
                    print(f"{time_str:<20} | {location:<8} | {actual:<10.2f} | {predicted:<10.2f} | {error:>+10.2f}")
                    
                    # Simulate real-time delay
                    time.sleep(self.delay)
                    
                    # Print progress every 100 requests
                    if self.total_requests % 100 == 0:
                        current_mae = self.total_mae / self.successful_requests
                        print(f"\n  Progress: {self.total_requests:,} requests | Running MAE: {current_mae:.4f}\n")
                
                except requests.exceptions.RequestException as e:
                    self.failed_requests += 1
                    logger.error(f"Request failed for record {idx}: {str(e)}")
                    continue
                
                except KeyboardInterrupt:
                    logger.info("\n\nStreaming interrupted by user")
                    break
                
                except Exception as e:
                    self.failed_requests += 1
                    logger.error(f"Error processing record {idx}: {str(e)}")
                    continue
            
            # Print final statistics
            elapsed_time = time.time() - start_time
            self.print_statistics(elapsed_time)
        
        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}")
            raise
    
    def print_statistics(self, elapsed_time: float):
        """
        Print final streaming statistics.
        
        Args:
            elapsed_time: Total elapsed time in seconds
        """
        print("\n" + "=" * 90)
        print("STREAMING SIMULATION COMPLETED")
        print("=" * 90)
        
        print(f"\n{'Total Requests:':<30} {self.total_requests:,}")
        print(f"{'Successful:':<30} {self.successful_requests:,}")
        print(f"{'Failed:':<30} {self.failed_requests:,}")
        print(f"{'Success Rate:':<30} {self.successful_requests/max(self.total_requests, 1)*100:.2f}%")
        
        if self.successful_requests > 0:
            mae = self.total_mae / self.successful_requests
            rmse = (self.total_squared_error / self.successful_requests) ** 0.5
            
            print(f"\n{'Performance Metrics:'}")
            print(f"{'  MAE (Mean Absolute Error):':<30} {mae:.4f}")
            print(f"{'  RMSE (Root Mean Squared Error):':<30} {rmse:.4f}")
        
        print(f"\n{'Elapsed Time:':<30} {elapsed_time:.2f}s")
        print(f"{'Requests per Second:':<30} {self.total_requests/max(elapsed_time, 1):.2f}")
        
        print("=" * 90 + "\n")


def main():
    """
    Main function to run the streaming simulation.
    """
    try:
        # Define paths
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "data" / "processed_inference_data_2025.parquet"
        
        # Initialize simulator
        simulator = TrafficStreamSimulator(
            data_path=data_path,
            api_url="http://localhost:8000",
            delay=0.1  # 100ms delay between requests
        )
        
        # Check API health
        print("\n" + "=" * 90)
        print("NYC TAXI TRAFFIC STREAM SIMULATOR")
        print("=" * 90)
        print("Checking API connectivity...\n")
        
        if not simulator.check_api_health():
            print("\n" + "=" * 90)
            print("ERROR: Cannot connect to API")
            print("=" * 90)
            print("\nPlease ensure the API server is running:")
            print("  ./start_api.sh")
            print("\nOr manually:")
            print("  cd src/inference && uvicorn app:app --host 0.0.0.0 --port 8000")
            print("=" * 90 + "\n")
            sys.exit(1)
        
        # Run streaming simulation
        # Default: stream first 100 records for demo
        # Remove max_records parameter to stream all data
        simulator.stream_data(max_records=100)
    
    except KeyboardInterrupt:
        logger.info("\n\nSimulation interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

