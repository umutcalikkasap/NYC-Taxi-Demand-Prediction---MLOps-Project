#!/bin/bash

# Run Continual Learning Check

echo "======================================================================="
echo "Continual Learning Check"
echo "======================================================================="
echo ""

DAYS_BACK=7
DRY_RUN=""
FORCE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --days-back)
            DAYS_BACK="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --force)
            FORCE="--force-retrain"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "Running continual learning check..."
echo "Days back: $DAYS_BACK"
echo "Dry run: ${DRY_RUN:-'no'}"
echo "Force retrain: ${FORCE:-'no'}"
echo ""

python -m src.continual_learning.run_continual_learning --days-back $DAYS_BACK $DRY_RUN $FORCE

echo ""
echo "✅ Check complete!"
echo "Report saved to: monitoring/continual_learning_reports/"
