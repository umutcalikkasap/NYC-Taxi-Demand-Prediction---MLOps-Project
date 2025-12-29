# Continual Learning Deployment Guide

This guide explains how to deploy the Continual Learning Monitor in production.

## Overview

The Continual Learning Monitor can be deployed in three ways:

1. **Systemd Service** (Recommended) - Runs as a daemon with automatic restarts
2. **Cron Job** - Scheduled periodic checks
3. **Manual/On-Demand** - Run checks manually when needed

---

## Option 1: Systemd Service (Recommended)

### Setup

1. **Copy the service file:**
```bash
sudo cp deployment/continual_learning_monitor.service /etc/systemd/system/
```

2. **Edit the service file with your paths:**
```bash
sudo nano /etc/systemd/system/continual_learning_monitor.service
```

Update these paths:
- `WorkingDirectory`: Your project directory
- `ExecStart`: Path to your Python virtual environment

3. **Reload systemd:**
```bash
sudo systemctl daemon-reload
```

4. **Enable and start the service:**
```bash
# Enable auto-start on boot
sudo systemctl enable continual_learning_monitor

# Start the service
sudo systemctl start continual_learning_monitor
```

### Management Commands

```bash
# Check status
sudo systemctl status continual_learning_monitor

# View logs
sudo journalctl -u continual_learning_monitor -f

# Restart service
sudo systemctl restart continual_learning_monitor

# Stop service
sudo systemctl stop continual_learning_monitor

# Disable auto-start
sudo systemctl disable continual_learning_monitor
```

### Configuration

Edit the service file to change:
- `--interval 86400`: Check interval in seconds (default: 24h)
- `--days-back 7`: Days of production data to check (default: 7)

Example for 12-hour checks:
```
ExecStart=/path/to/venv/bin/python -m src.continual_learning.monitor_service --interval 43200 --days-back 7
```

---

## Option 2: Cron Job

### Setup

1. **Edit crontab:**
```bash
crontab -e
```

2. **Add cron job:**

**Daily at 2 AM:**
```cron
0 2 * * * cd /path/to/project && /path/to/venv/bin/python -m src.continual_learning.monitor_service --run-once >> /path/to/logs/cron_cl.log 2>&1
```

**Every 12 hours:**
```cron
0 */12 * * * cd /path/to/project && /path/to/venv/bin/python -m src.continual_learning.monitor_service --run-once >> /path/to/logs/cron_cl.log 2>&1
```

**Every Monday at 3 AM:**
```cron
0 3 * * 1 cd /path/to/project && /path/to/venv/bin/python -m src.continual_learning.monitor_service --run-once >> /path/to/logs/cron_cl.log 2>&1
```

3. **Alternative: Use the shell script:**
```cron
0 2 * * * cd /path/to/project && /path/to/project/scripts/run_continual_learning.sh --days-back 7 >> /path/to/logs/cron_cl.log 2>&1
```

### View Cron Logs

```bash
# View cron log
tail -f /path/to/logs/cron_cl.log

# View application log
tail -f logs/continual_learning.log
```

---

## Option 3: Manual/On-Demand

### Single Check

```bash
# Run single check
python -m src.continual_learning.run_continual_learning --days-back 7

# Dry run (don't deploy)
python -m src.continual_learning.run_continual_learning --days-back 7 --dry-run

# Force retrain
python -m src.continual_learning.run_continual_learning --force-retrain

# Using shell script
./scripts/run_continual_learning.sh --days-back 7
```

---

## Monitoring & Alerts

### Check Logs

```bash
# Application logs
tail -f logs/continual_learning.log
tail -f logs/continual_learning_monitor.log

# Check reports
ls -lh monitoring/continual_learning_reports/
cat monitoring/continual_learning_reports/cl_report_*.json | jq '.'
```

### Integration with Prometheus

The continual learning check results can be exposed as Prometheus metrics:

```python
# Add to src/continual_learning/run_continual_learning.py
from prometheus_client import Gauge, Counter

cl_last_check_timestamp = Gauge('cl_last_check_timestamp', 'Last check timestamp')
cl_retrain_counter = Counter('cl_retrain_total', 'Total retrains triggered')
cl_deployment_counter = Counter('cl_deployment_total', 'Total model deployments')
cl_drift_detected = Gauge('cl_drift_detected', 'Drift detected (1) or not (0)')
```

### Email Alerts

Add email notifications when retraining is triggered:

```python
import smtplib
from email.mime.text import MIMEText

def send_alert(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'alerts@yourcompany.com'
    msg['To'] = 'ml-team@yourcompany.com'

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login('user', 'password')
        server.send_message(msg)

# In run_continual_learning_check():
if should_retrain:
    send_alert(
        'Model Retraining Triggered',
        f'Reasons: {reasons}\nMAE: {current_mae}'
    )
```

---

## Production Checklist

Before deploying to production:

- [ ] Set appropriate check interval (daily, weekly, etc.)
- [ ] Configure data retention (how many days to check)
- [ ] Set up log rotation
- [ ] Configure alerts (email, Slack, PagerDuty)
- [ ] Test dry-run mode first
- [ ] Set up monitoring dashboards
- [ ] Document rollback procedure
- [ ] Configure backup before deployment

---

## Troubleshooting

### Service won't start

```bash
# Check service status
sudo systemctl status continual_learning_monitor

# View detailed logs
sudo journalctl -u continual_learning_monitor -n 100 --no-pager

# Check Python path
which python
/path/to/venv/bin/python --version
```

### No production data found

- Verify data files exist in `data/inference/`
- Check date range (adjust `--days-back`)
- Verify data format

### Model not deploying

- Check file permissions
- Verify model path in config
- Check deployment logs
- Test with `--dry-run` first

### High memory usage

- Reduce `--days-back` value
- Use data sampling
- Increase check interval

---

## Advanced Configuration

### Docker Deployment

Add to `docker-compose.prod.yml`:

```yaml
services:
  continual-learning-monitor:
    build: .
    container_name: cl-monitor
    command: python -m src.continual_learning.monitor_service --interval 86400
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./monitoring:/app/monitoring
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    depends_on:
      - api
```

### Kubernetes CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: continual-learning-check
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: cl-check
            image: nyc-taxi-prediction:latest
            command:
            - python
            - -m
            - src.continual_learning.run_continual_learning
            - --days-back
            - "7"
          restartPolicy: OnFailure
```

---

## Best Practices

1. **Start Conservative**
   - Begin with dry-run mode
   - Monitor for 1-2 weeks before enabling auto-deployment

2. **Appropriate Interval**
   - Daily checks for high-traffic systems
   - Weekly checks for moderate traffic
   - Monthly checks for stable systems

3. **Data Window**
   - Use 7-14 days for performance checks
   - Use 30+ days for drift detection reference

4. **Thresholds**
   - Adjust MAE threshold based on your requirements
   - Set drift detection sensitivity appropriately

5. **Monitoring**
   - Set up alerts for check failures
   - Monitor deployment success rate
   - Track model performance over time

6. **Safety**
   - Always keep model backups
   - Test in staging first
   - Have rollback procedure ready

---

## Support

For issues or questions:
- Check logs: `logs/continual_learning.log`
- Review reports: `monitoring/continual_learning_reports/`
- Contact: ml-team@yourcompany.com
