# Project Structure

```
project-files/
│
├── data/                                    # Data storage
│   ├── training_batch/                     # Raw training data (2024)
│   ├── inference_stream/                   # Raw inference data (2025)
│   ├── processed_training_data_2024.parquet
│   └── processed_inference_data_2025.parquet
│
├── models/                                  # Trained models
│   └── xgb_model.json                      # Production XGBoost model
│
├── src/                                     # Source code
│   ├── __init__.py
│   ├── features.py                         # Shared feature engineering
│   │
│   ├── data_processing/                    # Data pipeline
│   │   ├── __init__.py
│   │   ├── aggregator.py                   # Raw data → aggregated data
│   │   └── prepare_inference_data.py       # Prepare 2025 data
│   │
│   ├── training/                           # Model training
│   │   ├── __init__.py
│   │   └── trainer.py                      # XGBoost training with MLflow
│   │
│   ├── inference/                          # Model serving
│   │   ├── __init__.py
│   │   └── app.py                          # FastAPI application
│   │
│   ├── simulation/                         # Testing & simulation
│   │   ├── __init__.py
│   │   └── stream_traffic.py               # Streaming simulation
│   │
│   └── monitoring/                         # NEW: Production monitoring
│       ├── __init__.py
│       ├── metrics.py                      # Prometheus metrics
│       ├── drift_detector.py               # Statistical drift detection
│       ├── data_quality.py                 # Data validation
│       └── run_drift_detection.py          # Drift analysis script
│
├── monitoring/                             # NEW: Monitoring configuration
│   ├── prometheus/
│   │   ├── prometheus.yml                  # Prometheus config
│   │   └── alerts/
│   │       └── prediction_alerts.yml       # Alert rules
│   │
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   └── taxi_demand_dashboard.json  # Grafana dashboard
│   │   └── datasources/
│   │       └── prometheus.yml              # Grafana datasource config
│   │
│   ├── postgres/
│   │   └── init.sql                        # Database initialization
│   │
│   └── drift_reports/                      # Drift detection reports
│       └── drift_report_*.json
│
├── scripts/                                # NEW: Utility scripts
│   ├── start_production.sh                 # Start all services
│   ├── check_metrics.sh                    # Check current metrics
│   └── run_drift_check.sh                  # Run drift detection
│
├── mlruns/                                 # MLflow tracking data
│   └── [experiment_id]/
│       └── [run_id]/
│           ├── meta.yaml
│           ├── metrics/
│           └── artifacts/
│
├── tests/                                  # Test files
│   ├── test_api.py
│   └── example_usage.py
│
├── venv/                                   # Python virtual environment
│
├── docker-compose.yml                      # Basic Docker setup
├── docker-compose.prod.yml                 # NEW: Production Docker Compose
├── Dockerfile                              # Container definition
├── requirements.txt                        # Python dependencies (updated)
│
├── API_README.md                           # API documentation
├── STREAMING_README.md                     # Streaming simulation docs
├── PRODUCTION_README.md                    # NEW: Production deployment guide
├── MLOPS_SUMMARY.md                        # NEW: Complete MLOps summary
├── PROJECT_STRUCTURE.md                    # This file
│
├── start_api.sh                            # Quick API start
└── demo_stream.sh                          # Demo streaming script
```

## Key Components

### Data Pipeline
- **aggregator.py**: Processes raw NYC taxi data into 15-min windows
- **prepare_inference_data.py**: Prepares 2025 data for drift detection
- **features.py**: Shared time-series feature engineering (lag + rolling)

### Model Training
- **trainer.py**: XGBoost training with temporal split validation
- **MLflow Integration**: Automatic experiment tracking
- **Current Performance**: MAE=3.28, RMSE=6.05

### Model Serving
- **app.py**: FastAPI REST API with Prometheus metrics
- **Endpoints**: /predict, /predict/batch, /health, /model/info, /metrics
- **Validation**: Pydantic schemas with range checking

### Monitoring (NEW)
- **metrics.py**: Prometheus metric definitions and decorators
- **drift_detector.py**: KS test + PSI for drift detection
- **data_quality.py**: Input validation and quality checks
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards

### Infrastructure (NEW)
- **docker-compose.prod.yml**: Complete production stack
  - API (FastAPI)
  - MLflow (Experiment tracking)
  - Prometheus (Metrics)
  - Grafana (Dashboards)
  - PostgreSQL (Logging)
  - Redis (Caching)

## Data Flow

```
1. Raw Data Collection
   └─> data/training_batch/*.parquet (2024)
   └─> data/inference_stream/*.parquet (2025)

2. Data Processing
   └─> src/data_processing/aggregator.py
   └─> data/processed_training_data_2024.parquet

3. Feature Engineering
   └─> src/features.py (lag + rolling features)
   └─> Ready for training/inference

4. Model Training
   └─> src/training/trainer.py
   └─> MLflow tracking
   └─> models/xgb_model.json

5. Production Serving
   └─> src/inference/app.py (FastAPI)
   └─> Prometheus metrics export
   └─> PostgreSQL logging

6. Monitoring & Drift Detection
   └─> src/monitoring/drift_detector.py
   └─> Grafana visualization
   └─> Alert on drift/errors
```

## Quick Start Commands

### Training
```bash
python -m src.training.trainer
mlflow ui  # View experiments
```

### Development API
```bash
./start_api.sh
# or
cd src/inference && uvicorn app:app --reload
```

### Production Stack
```bash
./scripts/start_production.sh
# Opens: API, MLflow, Prometheus, Grafana
```

### Testing
```bash
python test_api.py
./demo_stream.sh
```

### Monitoring
```bash
./scripts/check_metrics.sh
./scripts/run_drift_check.sh
```

## Configuration Files

| File | Purpose |
|------|---------|
| requirements.txt | Python dependencies |
| docker-compose.prod.yml | Production orchestration |
| monitoring/prometheus/prometheus.yml | Metrics collection config |
| monitoring/prometheus/alerts/*.yml | Alert rules |
| monitoring/grafana/dashboards/*.json | Dashboard definitions |
| monitoring/postgres/init.sql | Database schema |

## Environment Variables

Production deployment supports:
- `ENVIRONMENT`: production/development
- `LOG_LEVEL`: INFO/DEBUG/WARNING
- `MODEL_PATH`: Path to model file
- `MLFLOW_TRACKING_URI`: MLflow server URL

## Metrics Endpoints

- **API Metrics**: http://localhost:8000/metrics
- **Prometheus UI**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **MLflow**: http://localhost:5000

## Documentation

- [API_README.md](API_README.md) - API endpoints and usage
- [STREAMING_README.md](STREAMING_README.md) - Streaming simulation
- [PRODUCTION_README.md](PRODUCTION_README.md) - Production deployment
- [MLOPS_SUMMARY.md](MLOPS_SUMMARY.md) - Complete MLOps overview

---

**Last Updated**: 2025-12-27
**Model Version**: v1
**MLflow Experiment**: NYC_Taxi_Demand_v1
