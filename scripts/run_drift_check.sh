#!/bin/bash

# Run Drift Detection

echo "======================================================================="
echo "Drift Detection Check"
echo "======================================================================="
echo ""

DAYS_BACK=7

# Parse arguments
if [ "$1" == "--days-back" ]; then
    DAYS_BACK=$2
fi

echo "Running drift detection..."
echo "Days back: $DAYS_BACK"
echo ""

python -m src.monitoring.run_drift_detection --days-back $DAYS_BACK

echo ""
echo "✅ Drift check complete!"
echo "Report saved to: monitoring/drift_reports/"
