"""
Prometheus metrics for API monitoring.
Tracks predictions, latency, errors, and model performance.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from functools import wraps
from typing import Callable

# Prediction counters
prediction_counter = Counter(
    'taxi_predictions_total',
    'Total number of predictions made',
    ['endpoint', 'status']
)

batch_size_histogram = Histogram(
    'taxi_batch_prediction_size',
    'Size of batch predictions',
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

# Latency metrics
prediction_latency = Histogram(
    'taxi_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# Model performance metrics (real-time)
prediction_value_histogram = Histogram(
    'taxi_prediction_value',
    'Distribution of predicted demand values',
    buckets=[0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
)

# Error tracking
api_errors = Counter(
    'taxi_api_errors_total',
    'Total number of API errors',
    ['error_type', 'endpoint']
)

# Model info gauge
model_info = Gauge(
    'taxi_model_info',
    'Information about the loaded model',
    ['model_type', 'version']
)

# Active requests gauge
active_requests = Gauge(
    'taxi_active_requests',
    'Number of active prediction requests'
)

# Feature distribution tracking (for drift detection)
feature_value_summary = Summary(
    'taxi_feature_value',
    'Summary statistics of feature values',
    ['feature_name']
)


def track_predictions(endpoint: str):
    """
    Decorator to track prediction metrics.

    Args:
        endpoint: API endpoint name (e.g., 'single', 'batch')
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            active_requests.inc()
            start_time = time.time()

            try:
                # Execute prediction
                result = await func(*args, **kwargs)

                # Track success
                prediction_counter.labels(endpoint=endpoint, status='success').inc()

                # Track latency
                latency = time.time() - start_time
                prediction_latency.labels(endpoint=endpoint).observe(latency)

                # Track prediction values
                if hasattr(result, 'predicted_demand'):
                    prediction_value_histogram.observe(result.predicted_demand)
                elif hasattr(result, 'predictions'):
                    for pred in result.predictions:
                        prediction_value_histogram.observe(pred.predicted_demand)
                    batch_size_histogram.observe(len(result.predictions))

                return result

            except Exception as e:
                # Track errors
                prediction_counter.labels(endpoint=endpoint, status='error').inc()
                api_errors.labels(
                    error_type=type(e).__name__,
                    endpoint=endpoint
                ).inc()
                raise

            finally:
                active_requests.dec()

        return wrapper
    return decorator


def track_feature_distribution(feature_name: str, value: float):
    """
    Track feature value for drift detection.

    Args:
        feature_name: Name of the feature
        value: Feature value
    """
    feature_value_summary.labels(feature_name=feature_name).observe(value)


def set_model_info(model_type: str, version: str = "v1"):
    """
    Set model information metric.

    Args:
        model_type: Type of model (e.g., 'XGBoost')
        version: Model version
    """
    model_info.labels(model_type=model_type, version=version).set(1)
