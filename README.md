# 🗽 NYC Taxi Pulse - Demand Prediction MLOps Project

> **Real-time taxi demand prediction system with comprehensive MLOps practices including continual learning, drift detection, and live monitoring.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Dashboard Tabs](#-dashboard-tabs)
- [API Documentation](#-api-documentation)
- [MLOps Components](#-mlops-components)
- [Project Structure](#-project-structure)
- [Demo Guide](#-demo-guide)

---

## 🎯 Overview

NYC Taxi Pulse is a production-ready machine learning system that predicts taxi demand across NYC locations using historical trip data. The project demonstrates end-to-end MLOps practices including:

- **Real-time predictions** via FastAPI
- **Interactive monitoring dashboard** with 7 specialized tabs
- **Continual learning pipeline** for automatic model retraining
- **Drift detection** for data quality monitoring
- **Model versioning** and registry
- **Live prediction streaming** with auto-refresh

### Key Metrics
- **MAE**: 2.32 trips
- **RMSE**: 4.47 trips
- **R² Score**: 0.9626
- **Model**: XGBoost Regressor
- **Features**: 20 engineered features

---

## ✨ Features

### 🤖 Machine Learning
- ✅ XGBoost regression model
- ✅ 20 engineered features (temporal, lag, rolling stats)
- ✅ Multi-model comparison (XGBoost, LightGBM, CatBoost)
- ✅ Feature importance analysis
- ✅ Model persistence and versioning

### 🔄 MLOps Pipeline
- ✅ **Continual Learning**: Automatic retraining based on performance degradation or drift
- ✅ **Drift Detection**: KS test and PSI for feature drift monitoring
- ✅ **Model Registry**: Version control with performance tracking
- ✅ **A/B Testing**: Champion vs challenger model comparison
- ✅ **MLflow Integration**: Experiment tracking

### 📊 Monitoring & Visualization
- ✅ **Real-time Dashboard**: 7-tab Streamlit interface
- ✅ **Live Predictions**: Streaming predictions with auto-refresh
- ✅ **NYC Heatmap**: Interactive demand visualization
- ✅ **Actual vs Predicted**: Time-series comparison
- ✅ **Prometheus Metrics**: Production monitoring

### 🚀 Deployment
- ✅ **FastAPI Backend**: High-performance prediction API
- ✅ **Batch Predictions**: Process multiple requests
- ✅ **Health Checks**: API monitoring endpoints
- ✅ **Docker Ready**: Containerization support

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NYC Taxi Pulse                            │
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Data       │─────▶│   Feature    │─────▶│   Training   │  │
│  │ Ingestion    │      │ Engineering  │      │   Pipeline   │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                      │         │
│         │                      │                      ▼         │
│         │                      │              ┌──────────────┐  │
│         │                      │              │    Model     │  │
│         │                      │              │   Registry   │  │
│         │                      │              └──────────────┘  │
│         │                      │                      │         │
│         ▼                      ▼                      ▼         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Prediction API (FastAPI)                     │  │
│  │  • /predict  • /health  • /model/info  • /metrics       │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                      │                      │         │
│         ▼                      ▼                      ▼         │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │  Dashboard   │      │   Continual  │      │    Drift     │  │
│  │  (Streamlit) │      │   Learning   │      │  Detection   │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Conda (recommended) or virtualenv
- 8GB+ RAM recommended
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd nyc-taxi-pulse-project
   ```

2. **Create conda environment**
   ```bash
   conda create -n mlops python=3.11
   conda activate mlops
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python --version  # Should show Python 3.11+
   ```

### Verify Model

Check that the trained model exists:
```bash
ls -lh models/xgb_model.json
# Should show: -rw-rw-r-- 362K xgb_model.json
```

If model doesn't exist, train it:
```bash
python -m src.training.trainer
```

---

## 💻 Usage

### Start the API

**Terminal 1:**
```bash
./start_api.sh
```

Expected output:
```
=======================================================================
Starting NYC Taxi Demand Prediction API
=======================================================================
Starting server on http://0.0.0.0:8000
Model loaded: ✓
Expected features: [20 features]
```

### Start the Dashboard

**Terminal 2:**
```bash
./start_dashboard.sh
```

Expected output:
```
=======================================================================
NYC Taxi MLOps Dashboard - Unified
=======================================================================
✅ API is running

📊 Dashboard: http://localhost:8501
```

### Access the Applications

- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 📊 Dashboard Tabs

### 1. 🗺️ NYC Map
Interactive map showing demand heatmap across NYC locations.

**Features:**
- Hourly demand visualization
- Location-based filtering
- Live predictions on map
- Demand statistics

### 2. 🔄 Continual Learning
Monitor model performance and retraining events.

**Features:**
- Current model version and metrics
- Model history (last 10 versions)
- Latest CL check report
- Performance degradation alerts
- Retraining triggers

### 3. 📉 Drift Detection
Feature drift analysis and monitoring.

**Features:**
- Drift status dashboard
- 20 features tracked
- KS test statistics
- PSI (Population Stability Index)
- Visual drift indicators

### 4. 🎯 Live Predictions
Manual prediction testing interface.

**Features:**
- Location selection
- Hour selection
- Instant predictions
- Historical comparison

### 5. 📊 Live Monitoring
Production metrics and system health.

**Features:**
- API health status
- Model info display
- Quick load testing
- Recent activity log

### 6. 📈 Actual vs Predicted
Time-series comparison of predictions vs actuals.

**Features:**
- Multi-location comparison
- Configurable time windows (24h, 48h, 7 days)
- 3 chart types (Line, Area, Scatter)
- Performance metrics (MAE, RMSE, MAPE, R²)
- Detailed comparison table
- Interactive Plotly charts

### 7. 🔴 Real-Time Stream
Live prediction streaming with auto-refresh.

**Features:**
- Auto-streaming mode (1-10s refresh)
- Real-time chart updates
- Moving average trend line
- Streaming metrics (Latest, Avg, Min/Max)
- Prediction history table
- CSV export
- Dark theme visualization

---

## 🔌 API Documentation

### Endpoints

#### `POST /predict`
Make a single prediction.

**Request:**
```json
{
  "PULocationID": 237,
  "hour": 18,
  "day_of_week": 4,
  "month": 1,
  "is_weekend": 0,
  "is_rush_hour": 1,
  "day_of_month": 15,
  "week_of_year": 24,
  "time_of_day": 2,
  "season": 0,
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
```

**Response:**
```json
{
  "predicted_demand": 18.5
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-12-29T13:00:00"
}
```

#### `GET /model/info`
Get model information.

**Response:**
```json
{
  "model_type": "XGBoost Regressor",
  "n_features": 20,
  "expected_features": ["PULocationID", "hour", ...]
}
```

#### `GET /metrics`
Prometheus metrics endpoint for monitoring.

---

## 🛠️ MLOps Components

### Continual Learning Pipeline

**Trigger Conditions:**
1. MAE > 2.5 (baseline threshold)
2. Performance degradation > 20%
3. Drift detected in features

**Process:**
```bash
# Dry run (no deployment)
python -m src.continual_learning.run_continual_learning --days-back 7 --dry-run

# Production run (auto-deploy if better)
python -m src.continual_learning.run_continual_learning --days-back 7

# Force retrain
python -m src.continual_learning.run_continual_learning --force-retrain
```

**Output:**
- Report: `monitoring/continual_learning_reports/cl_report_*.json`
- New model: `models/production/xgb_model_vX.X.X.json`
- Updated registry: `models/production/model_registry.json`

### Drift Detection

**Statistical Tests:**
- Kolmogorov-Smirnov (KS) test: p-value < 0.05
- Population Stability Index (PSI): PSI > 0.25

**Run Drift Check:**
```bash
python -m src.monitoring.run_drift_detection --days-back 7
```

**Output:**
- Report: `monitoring/drift_reports/drift_report_*.json`
- Features checked: 20
- Drift indicators per feature

### Model Training

**Single Model:**
```bash
python -m src.training.trainer
```

**Model Comparison:**
```bash
python -m src.training.model_comparison
```

**Output:**
- Model: `models/xgb_model.json`
- MLflow tracking: `mlruns/`
- Performance metrics logged

---

## 📁 Project Structure

```
nyc-taxi-pulse-project/
├── data/
│   ├── processed_training_data_2024.parquet
│   └── processed_inference_data_2025.parquet
├── models/
│   ├── xgb_model.json (current model)
│   └── production/
│       ├── model_registry.json
│       └── xgb_model_v*.json
├── monitoring/
│   ├── continual_learning_reports/
│   └── drift_reports/
├── src/
│   ├── dashboard/
│   │   └── unified_dashboard.py (7 tabs)
│   ├── inference/
│   │   └── app.py (FastAPI)
│   ├── training/
│   │   ├── trainer.py
│   │   └── model_comparison.py
│   ├── continual_learning/
│   │   ├── retraining_pipeline.py
│   │   └── run_continual_learning.py
│   ├── monitoring/
│   │   ├── drift_detector.py
│   │   └── run_drift_detection.py
│   └── features.py (feature engineering)
├── scripts/
│   ├── run_continual_learning.sh
│   └── run_drift_check.sh
├── start_api.sh
├── start_dashboard.sh
├── requirements.txt
└── README.md
```

---

## 🎬 Demo Guide

### Quick Demo (5 minutes)

1. **Start services**
   ```bash
   # Terminal 1
   ./start_api.sh

   # Terminal 2
   ./start_dashboard.sh
   ```

2. **Open Dashboard**: http://localhost:8501

3. **Show features in order:**
   - **Tab 1 (NYC Map)**: Interactive heatmap
   - **Tab 7 (Real-Time Stream)**: Enable auto-stream at 2s refresh
   - **Tab 6 (Actual vs Predicted)**: Show time-series comparison
   - **Tab 3 (Drift Detection)**: Show drift analysis
   - **Tab 2 (Continual Learning)**: Show model registry

### Full Demo (15 minutes)

**Part 1: Predictions (3 min)**
- Tab 1: Show demand heatmap for different hours (9am, 6pm, 11pm)
- Tab 4: Make manual predictions for Times Square vs JFK

**Part 2: Real-Time Stream (4 min)**
- Tab 7: Enable auto-stream
- Show live chart updating
- Explain moving average
- Change locations to compare

**Part 3: MLOps Features (5 min)**
- Tab 3: Explain drift detection (17/20 features drifted)
- Tab 2: Show model versions and CL reports
- Tab 6: Show actual vs predicted performance

**Part 4: API (3 min)**
- Open http://localhost:8000/docs
- Show Swagger UI
- Test /predict endpoint
- Show /metrics endpoint

---

## 🔧 Advanced Usage

### Run Complete MLOps Cycle

```bash
# 1. Check for drift
python -m src.monitoring.run_drift_detection --days-back 7

# 2. Run CL check
python -m src.continual_learning.run_continual_learning --days-back 7

# 3. View reports in dashboard
# Tab 2: Continual Learning
# Tab 3: Drift Detection
```

### Custom Feature Engineering

Edit `src/features.py` to add new features:
```python
def add_custom_features(df):
    df['my_feature'] = ...  # Your logic
    return df
```

### Model Experimentation

Use MLflow UI:
```bash
mlflow ui
# Open http://localhost:5000
```

---

## 📊 Performance Benchmarks

### Model Performance (Test Set)
| Metric | Value |
|--------|-------|
| MAE | 2.32 trips |
| RMSE | 4.47 trips |
| R² Score | 0.9626 |
| MAPE | 29.5% |
| Within ±5 trips | 86.8% |

### API Performance
| Metric | Value |
|--------|-------|
| Response time | ~50ms |
| Throughput | ~200 req/s |
| Model size | 362KB |

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is for educational and demonstration purposes.

---

## 🙏 Acknowledgments

- NYC TLC for trip data
- XGBoost, Streamlit, FastAPI communities
- MLOps best practices from various sources

---

## 📞 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ for MLOps excellence**

🗽 NYC Taxi Pulse - Predicting the future, one ride at a time.
