#!/bin/bash

# Start Unified Dashboard - All-in-One

echo "======================================================================="
echo "NYC Taxi MLOps Dashboard - Unified"
echo "======================================================================="
echo ""

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is running"
else
    echo "⚠️  Warning: API is not running"
    echo "Start API: ./start_api.sh"
    echo ""
fi

echo ""
echo "Launching Unified Dashboard..."
echo ""
echo "📊 Dashboard: http://localhost:8501"
echo ""
echo "Features:"
echo "  🗺️  NYC Map (demand heatmap + live predictions)"
echo "  🔄 Continual Learning monitoring"
echo "  📉 Drift Detection analysis"
echo "  🎯 Live Prediction testing"
echo ""
echo "Press Ctrl+C to stop"
echo "======================================================================="
echo ""

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mlops

# Start Streamlit
streamlit run src/dashboard/unified_dashboard.py
