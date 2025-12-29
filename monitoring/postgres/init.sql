-- Initialize PostgreSQL database for MLOps
-- Stores predictions, actuals, and feedback for monitoring

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    location_id INTEGER NOT NULL,
    hour INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,
    month INTEGER NOT NULL,
    lag_1 REAL NOT NULL,
    lag_4 REAL NOT NULL,
    lag_96 REAL NOT NULL,
    rolling_mean_4 REAL NOT NULL,
    predicted_demand REAL NOT NULL,
    actual_demand REAL,
    error REAL,
    latency_ms REAL,
    model_version VARCHAR(50) DEFAULT 'v1',
    endpoint VARCHAR(50),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create indexes for common queries
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp DESC);
CREATE INDEX idx_predictions_location ON predictions(location_id);
CREATE INDEX idx_predictions_model_version ON predictions(model_version);

-- Drift reports table
CREATE TABLE IF NOT EXISTS drift_reports (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    report_type VARCHAR(50) NOT NULL,
    drift_detected BOOLEAN NOT NULL,
    drift_score REAL,
    features_affected TEXT[],
    report_json JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_drift_reports_timestamp ON drift_reports(timestamp DESC);
CREATE INDEX idx_drift_reports_type ON drift_reports(report_type);

-- Model performance metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    model_version VARCHAR(50) DEFAULT 'v1',
    data_window VARCHAR(50),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_model_metrics_timestamp ON model_metrics(timestamp DESC);
CREATE INDEX idx_model_metrics_name ON model_metrics(metric_name);

-- API logs table
CREATE TABLE IF NOT EXISTS api_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    endpoint VARCHAR(100) NOT NULL,
    status_code INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_api_logs_timestamp ON api_logs(timestamp DESC);
CREATE INDEX idx_api_logs_endpoint ON api_logs(endpoint);
CREATE INDEX idx_api_logs_status ON api_logs(status_code);

-- Create views for common queries
CREATE OR REPLACE VIEW recent_performance AS
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as prediction_count,
    AVG(predicted_demand) as avg_predicted,
    AVG(actual_demand) as avg_actual,
    AVG(ABS(error)) as mae,
    SQRT(AVG(error * error)) as rmse,
    AVG(latency_ms) as avg_latency_ms
FROM predictions
WHERE actual_demand IS NOT NULL
AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;

CREATE OR REPLACE VIEW hourly_predictions AS
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    location_id,
    AVG(predicted_demand) as avg_predicted_demand,
    COUNT(*) as prediction_count
FROM predictions
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', timestamp), location_id
ORDER BY hour DESC, location_id;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlops;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlops;
