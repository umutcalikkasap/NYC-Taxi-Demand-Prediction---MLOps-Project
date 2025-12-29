"""
FastAPI application for NYC Taxi demand prediction.
Serves real-time predictions using trained XGBoost model.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Union
import pandas as pd
import xgboost as xgb
from pathlib import Path
import logging
from datetime import datetime
from prometheus_client import make_asgi_app
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from monitoring.metrics import (
    track_predictions,
    track_feature_distribution,
    set_model_info
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NYC Taxi Demand Prediction API",
    description="Real-time taxi demand prediction using XGBoost",
    version="1.0.0"
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Global model variable
MODEL = None
FEATURE_COLUMNS = [
    'PULocationID', 'hour', 'day_of_week', 'month',
    'is_weekend', 'is_rush_hour', 'day_of_month', 'week_of_year',
    'time_of_day', 'season', 'is_manhattan',
    'lag_1', 'lag_4', 'lag_24', 'lag_96', 'lag_672',
    'rolling_mean_4', 'rolling_std_4', 'rolling_max_24', 'rolling_min_24'
]


class PredictionInput(BaseModel):
    """
    Input schema for demand prediction.

    All features must match those used during model training.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "PULocationID": 237,
                "hour": 18,
                "day_of_week": 4,
                "month": 6,
                "is_weekend": 0,
                "is_rush_hour": 1,
                "day_of_month": 15,
                "week_of_year": 24,
                "time_of_day": 2,
                "season": 2,
                "is_manhattan": 1,
                "lag_1": 15.0,
                "lag_4": 18.0,
                "lag_24": 17.5,
                "lag_96": 20.0,
                "lag_672": 18.5,
                "rolling_mean_4": 16.5,
                "rolling_std_4": 3.2,
                "rolling_max_24": 25.0,
                "rolling_min_24": 10.0
            }
        }
    )

    PULocationID: int = Field(..., ge=1, le=263, description="Pickup location ID (1-263)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    is_weekend: int = Field(..., ge=0, le=1, description="Is weekend (0 or 1)")
    is_rush_hour: int = Field(..., ge=0, le=1, description="Is rush hour (0 or 1)")
    day_of_month: int = Field(..., ge=1, le=31, description="Day of month (1-31)")
    week_of_year: int = Field(..., ge=1, le=53, description="Week of year (1-53)")
    time_of_day: int = Field(..., ge=0, le=3, description="Time of day (0=night, 1=morning, 2=afternoon, 3=evening)")
    season: int = Field(..., ge=0, le=3, description="Season (0=winter, 1=spring, 2=summer, 3=fall)")
    is_manhattan: int = Field(..., ge=0, le=1, description="Is Manhattan location (0 or 1)")
    lag_1: float = Field(..., ge=0, description="Demand 15 minutes ago")
    lag_4: float = Field(..., ge=0, description="Demand 1 hour ago")
    lag_24: float = Field(..., ge=0, description="Demand 6 hours ago")
    lag_96: float = Field(..., ge=0, description="Demand 24 hours ago")
    lag_672: float = Field(..., ge=0, description="Demand 1 week ago")
    rolling_mean_4: float = Field(..., ge=0, description="Average demand over last hour")
    rolling_std_4: float = Field(..., ge=0, description="Std deviation of demand over last hour")
    rolling_max_24: float = Field(..., ge=0, description="Max demand over last 6 hours")
    rolling_min_24: float = Field(..., ge=0, description="Min demand over last 6 hours")


class PredictionOutput(BaseModel):
    """Output schema for demand prediction."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_demand": 18.5
            }
        }
    )
    
    predicted_demand: float = Field(..., description="Predicted taxi demand")


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    {"predicted_demand": 18.5},
                    {"predicted_demand": 22.3}
                ],
                "count": 2
            }
        }
    )
    
    predictions: List[PredictionOutput]
    count: int = Field(..., description="Number of predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str


def load_model() -> xgb.XGBRegressor:
    """
    Load the trained XGBoost model from disk.

    Returns:
        Loaded XGBoost model

    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: For other loading errors
    """
    try:
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "models" / "xgb_model.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load booster and wrap in XGBRegressor (compatible with XGBoost 2.0+)
        booster = xgb.Booster()
        booster.load_model(model_path)

        # Create XGBRegressor wrapper
        model = xgb.XGBRegressor()
        model._Booster = booster

        logger.info(f"Model loaded successfully from {model_path}")
        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    global MODEL
    try:
        MODEL = load_model()
        logger.info("=" * 70)
        logger.info("NYC Taxi Demand Prediction API Started")
        logger.info("=" * 70)
        logger.info(f"Model loaded: ✓")
        logger.info(f"Expected features: {FEATURE_COLUMNS}")
        logger.info(f"Prometheus metrics: /metrics")
        logger.info("=" * 70)

        # Set model info for Prometheus
        set_model_info(model_type="XGBoost", version="v1")

    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "NYC Taxi Demand Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict_single": "/predict",
            "predict_batch": "/predict/batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint.
    
    Returns current status and model availability.
    """
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
@track_predictions(endpoint="single")
async def predict_single(input_data: PredictionInput):
    """
    Predict demand for a single input.

    Args:
        input_data: Single prediction input with all required features

    Returns:
        Predicted demand value

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    try:
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert input to dictionary
        input_dict = input_data.dict()

        # Track feature distributions for drift detection
        for feature, value in input_dict.items():
            if feature in ['lag_1', 'lag_4', 'lag_96', 'rolling_mean_4']:
                track_feature_distribution(feature, value)

        # Create DataFrame with correct column order
        df = pd.DataFrame([input_dict], columns=FEATURE_COLUMNS)

        # Ensure proper data types
        df['PULocationID'] = df['PULocationID'].astype(int)

        # Make prediction
        prediction = MODEL.predict(df)[0]

        # Ensure non-negative prediction
        prediction = max(0.0, float(prediction))

        return PredictionOutput(predicted_demand=prediction)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
@track_predictions(endpoint="batch")
async def predict_batch(input_data: List[PredictionInput]):
    """
    Predict demand for multiple inputs (batch prediction).

    Args:
        input_data: List of prediction inputs

    Returns:
        List of predicted demand values with count

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    try:
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if len(input_data) == 0:
            raise HTTPException(status_code=400, detail="Empty input list")

        # Convert list of inputs to DataFrame
        input_dicts = [item.dict() for item in input_data]
        df = pd.DataFrame(input_dicts, columns=FEATURE_COLUMNS)

        # Ensure proper data types
        df['PULocationID'] = df['PULocationID'].astype(int)

        # Make predictions
        predictions = MODEL.predict(df)

        # Ensure non-negative predictions
        predictions = [max(0.0, float(pred)) for pred in predictions]

        # Format output
        prediction_outputs = [
            PredictionOutput(predicted_demand=pred)
            for pred in predictions
        ]

        return BatchPredictionOutput(
            predictions=prediction_outputs,
            count=len(predictions)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Model metadata and feature information
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        return {
            "model_type": "XGBoost Regressor",
            "n_estimators": MODEL.n_estimators,
            "max_depth": MODEL.max_depth,
            "learning_rate": MODEL.learning_rate,
            "expected_features": FEATURE_COLUMNS,
            "n_features": len(FEATURE_COLUMNS)
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

