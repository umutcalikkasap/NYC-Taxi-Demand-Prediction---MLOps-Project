# Continual Learning Module

Automated model monitoring, drift detection, and retraining for NYC Taxi Demand Prediction.

## Quick Start

### 1. Demo (For Presentation)

```bash
# Run 4-week simulation
python -m src.continual_learning.demo_continual_learning --weeks 4

# Custom simulation
python -m src.continual_learning.demo_continual_learning --weeks 8 --start-date 2025-01-01
```

### 2. Production Check

```bash
# Check last 7 days
./scripts/run_continual_learning.sh --days-back 7

# Dry run (don't deploy)
./scripts/run_continual_learning.sh --days-back 7 --dry-run

# Force retrain
./scripts/run_continual_learning.sh --force
```

### 3. Background Service

```bash
# Run as daemon (checks every 24h)
python -m src.continual_learning.monitor_service

# Run single check and exit
python -m src.continual_learning.monitor_service --run-once

# Custom interval (1 hour)
python -m src.continual_learning.monitor_service --interval 3600
```

## Components

### ContinualLearningPipeline

Main orchestrator for the continual learning workflow.

```python
from src.continual_learning import ContinualLearningPipeline

pipeline = ContinualLearningPipeline()

# Load production data
prod_data = pipeline.load_production_data(days_back=7)

# Check performance
metrics = pipeline.check_performance(prod_data)

# Check drift
drift = pipeline.check_drift(prod_data, reference_data)

# Decide on retraining
should_retrain, reasons = pipeline.should_retrain(metrics, drift)

if should_retrain:
    # Retrain
    new_model, version, train_metrics = pipeline.retrain_model()

    # Compare models
    comparison = pipeline.compare_models(current_model, new_model, test_data)

    # Deploy if better
    if comparison['winner'] == 'new':
        pipeline.deploy_model(new_model, version)
```

### Retraining Triggers

1. **Performance Degradation**
   - MAE > 2.5 threshold
   - MAE increase > 20% from baseline

2. **Data Drift**
   - KS test p-value < 0.05
   - PSI score > 0.25

3. **Manual**
   - Force retrain flag

### Model Versioning

Models are versioned semantically:
- `v1.0.0` - Initial model
- `v1.1.0` - Minor update
- `v2.0.0` - Major update

Registry stored in: `models/production/model_registry.json`

## Configuration

Edit `src/config.py`:

```python
class Config:
    # Continual learning
    CL_MAE_THRESHOLD = 2.5  # Retrain if MAE > this
    CL_MAE_INCREASE_PCT = 20  # Retrain if MAE increase > this %
    CL_DRIFT_PVALUE_THRESHOLD = 0.05  # KS test threshold
    CL_PSI_THRESHOLD = 0.25  # PSI threshold
```

## Monitoring

### Check Logs

```bash
# Application logs
tail -f logs/continual_learning.log
tail -f logs/continual_learning_monitor.log

# Reports
ls -lh monitoring/continual_learning_reports/
cat monitoring/continual_learning_reports/cl_report_*.json | jq '.'
```

### Model Registry

```bash
# View model history
cat models/production/model_registry.json | jq '.'

# Get current model
cat models/production/model_registry.json | jq '.[-1]'
```

## Deployment

### Systemd Service

```bash
sudo cp deployment/continual_learning_monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable continual_learning_monitor
sudo systemctl start continual_learning_monitor
```

### Cron Job

```cron
# Daily at 2 AM
0 2 * * * cd /path/to/project && python -m src.continual_learning.monitor_service --run-once
```

### Docker

Add to `docker-compose.prod.yml`:

```yaml
continual-learning-monitor:
  build: .
  command: python -m src.continual_learning.monitor_service --interval 86400
  volumes:
    - ./data:/app/data
    - ./models:/app/models
    - ./logs:/app/logs
```

## Files

```
src/continual_learning/
├── __init__.py                      # Module exports
├── README.md                        # This file
├── retraining_pipeline.py           # Main pipeline class
├── demo_continual_learning.py       # Interactive demo
├── run_continual_learning.py        # Production runner
└── monitor_service.py               # Background daemon

scripts/
└── run_continual_learning.sh        # Convenience script

deployment/
├── continual_learning_monitor.service  # Systemd service
└── CONTINUAL_LEARNING_DEPLOYMENT.md    # Deployment guide

monitoring/continual_learning_reports/   # JSON reports
models/production/model_registry.json    # Model versions
```

## Documentation

- [Continual Learning Summary](../../CONTINUAL_LEARNING_SUMMARY.md) - Overview
- [Deployment Guide](../../deployment/CONTINUAL_LEARNING_DEPLOYMENT.md) - Production deployment
- [Presentation Commands](../../PRESENTATION_COMMANDS.md) - Demo commands
- [Demo Guide](../../DEMO_GUIDE.md) - Presentation walkthrough

## Examples

### Example 1: Weekly Production Check

```bash
# Check last 7 days, deploy if needed
./scripts/run_continual_learning.sh --days-back 7
```

### Example 2: Test Before Deployment

```bash
# Dry run - see what would happen
./scripts/run_continual_learning.sh --days-back 7 --dry-run

# Check the report
cat monitoring/continual_learning_reports/cl_report_*.json | jq '.should_retrain, .retrain_reasons'
```

### Example 3: Force Retrain

```bash
# Force retrain regardless of checks
./scripts/run_continual_learning.sh --force

# Check model registry
cat models/production/model_registry.json | jq '.[-2:]'
```

### Example 4: Custom Date Range

```bash
# Check specific period
./scripts/run_continual_learning.sh --days-back 14 --end-date 2025-01-15
```

## API

### ContinualLearningPipeline

```python
class ContinualLearningPipeline:
    def load_production_data(days_back: int, end_date: str = None) -> pd.DataFrame
    def check_performance(production_data: pd.DataFrame) -> dict
    def check_drift(prod_data: pd.DataFrame, ref_data: pd.DataFrame) -> dict
    def should_retrain(performance: dict, drift: dict) -> tuple[bool, list]
    def retrain_model(training_data: pd.DataFrame = None, training_end_date: str = None) -> tuple
    def compare_models(model1_path: Path, model2_path: Path, test_data: pd.DataFrame) -> dict
    def deploy_model(model_path: Path, version: str) -> bool
```

## Troubleshooting

### Issue: No production data found

**Solution:** Check that `data/inference/` contains parquet files with dates in range.

### Issue: Model won't deploy

**Solution:**
1. Check file permissions
2. Verify model path
3. Test with `--dry-run` first

### Issue: Service won't start

**Solution:**
```bash
sudo systemctl status continual_learning_monitor
sudo journalctl -u continual_learning_monitor -n 100
```

### Issue: High memory usage

**Solution:**
- Reduce `--days-back` value
- Increase check interval
- Use data sampling

## Support

For issues:
- Check logs: `logs/continual_learning.log`
- Review reports: `monitoring/continual_learning_reports/`
- See main documentation: [CONTINUAL_LEARNING_SUMMARY.md](../../CONTINUAL_LEARNING_SUMMARY.md)
