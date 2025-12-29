#!/bin/bash
# Startup script for NYC Taxi Demand Prediction API

echo "======================================================================="
echo "Starting NYC Taxi Demand Prediction API"
echo "======================================================================="

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mlops

# Navigate to inference directory
cd src/inference

# Start FastAPI server with uvicorn
echo "Starting server on http://0.0.0.0:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "======================================================================="

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

