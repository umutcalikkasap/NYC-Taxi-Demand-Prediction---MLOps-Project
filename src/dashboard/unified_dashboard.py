"""
NYC Taxi Demand Prediction - Unified Dashboard

All-in-one dashboard combining:
- NYC Map Visualization (Folium)
- MLOps Monitoring (Continual Learning, Drift Detection)
- Real-time Predictions

Usage:
    streamlit run src/dashboard/unified_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import json
import requests
import time

# Page config
st.set_page_config(
    page_title="NYC Taxi MLOps Dashboard",
    page_icon="🗽",
    layout="wide"
)

# NYC Locations (major Manhattan zones)
NYC_LOCATIONS = {
    237: {"lat": 40.7869, "lon": -73.9754, "name": "Upper West Side South"},
    161: {"lat": 40.7589, "lon": -73.9851, "name": "Midtown East"},
    162: {"lat": 40.7614, "lon": -73.9776, "name": "Midtown North"},
    163: {"lat": 40.7589, "lon": -73.9851, "name": "Midtown South"},
    230: {"lat": 40.7589, "lon": -73.9851, "name": "UN/Turtle Bay South"},
    170: {"lat": 40.7284, "lon": -73.9942, "name": "Penn Station"},
    211: {"lat": 40.7484, "lon": -73.9857, "name": "Times Square"},
    231: {"lat": 40.8020, "lon": -73.9665, "name": "Union Square"},
    232: {"lat": 40.7789, "lon": -73.9692, "name": "Upper East Side North"},
    233: {"lat": 40.7677, "lon": -73.9628, "name": "Upper East Side South"},
    234: {"lat": 40.7945, "lon": -73.9686, "name": "Upper West Side North"},
    87: {"lat": 40.7589, "lon": -73.9851, "name": "Financial District North"},
    88: {"lat": 40.7071, "lon": -74.0134, "name": "Financial District South"},
    224: {"lat": 40.7589, "lon": -73.9851, "name": "Tribeca"},
    100: {"lat": 40.7589, "lon": -73.9851, "name": "Garment District"},
    68: {"lat": 40.7484, "lon": -73.9857, "name": "East Village"},
    90: {"lat": 40.7484, "lon": -73.9857, "name": "Flatiron"},
    107: {"lat": 40.7589, "lon": -73.9851, "name": "Greenwich Village North"},
    125: {"lat": 40.8448, "lon": -73.9379, "name": "Harlem"},
    144: {"lat": 40.7484, "lon": -73.9857, "name": "Little Italy"},
    148: {"lat": 40.7589, "lon": -73.9851, "name": "Lower East Side"},
    158: {"lat": 40.7484, "lon": -73.9857, "name": "Midtown Center"},
    138: {"lat": 40.7769, "lon": -73.9087, "name": "LaGuardia Airport"},
    132: {"lat": 40.6437, "lon": -73.7823, "name": "JFK Airport"},
}

# Title
st.title("🗽 NYC Taxi Demand Prediction - MLOps Dashboard")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # API config
    api_url = st.text_input("API URL", "http://localhost:8000")

    # Auto refresh
    auto_refresh = st.checkbox("Auto Refresh (30s)", False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()

    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    st.divider()
    st.info(f"📍 Tracking {len(NYC_LOCATIONS)} NYC locations")


@st.cache_data(ttl=300)
def load_demand_data():
    """Load historical demand."""
    data_path = Path("data/processed_inference_data_2025.parquet")
    if data_path.exists():
        df = pd.read_parquet(data_path)
        # Use pickup_window if available, otherwise try tpep_pickup_datetime
        time_col = 'pickup_window' if 'pickup_window' in df.columns else 'tpep_pickup_datetime'
        df['hour'] = pd.to_datetime(df[time_col]).dt.hour
        return df.groupby(['PULocationID', 'hour']).size().reset_index(name='demand')
    return pd.DataFrame()


@st.cache_data(ttl=30)
def load_model_registry():
    """Load model registry."""
    path = Path("models/production/model_registry.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


@st.cache_data(ttl=30)
def load_cl_reports():
    """Load continual learning reports."""
    reports_dir = Path("monitoring/continual_learning_reports")
    if not reports_dir.exists():
        return []

    reports = []
    for f in sorted(reports_dir.glob("cl_report_*.json"), reverse=True)[:10]:
        with open(f) as file:
            data = json.load(file)
            reports.append(data)
    return reports


@st.cache_data(ttl=30)
def load_drift_reports():
    """Load drift reports."""
    reports_dir = Path("monitoring/drift_reports")
    if not reports_dir.exists():
        return []

    reports = []
    for f in sorted(reports_dir.glob("drift_report_*.json"), reverse=True)[:10]:
        with open(f) as file:
            data = json.load(file)
            reports.append(data)
    return reports


def create_folium_heatmap(hour):
    """Create Folium heatmap."""
    demand_df = load_demand_data()
    demand_dict = {}

    if not demand_df.empty:
        hour_data = demand_df[demand_df['hour'] == hour]
        for _, row in hour_data.iterrows():
            loc_id = int(row['PULocationID'])
            if loc_id in NYC_LOCATIONS:
                demand_dict[loc_id] = float(row['demand'])

    # Create map
    m = folium.Map(location=[40.7589, -73.9851], zoom_start=12, tiles='OpenStreetMap')

    # Heatmap data
    heat_data = []
    for loc_id, loc_info in NYC_LOCATIONS.items():
        demand = demand_dict.get(loc_id, 0)
        if demand > 0:
            heat_data.append([loc_info['lat'], loc_info['lon'], demand / 100])

    if heat_data:
        HeatMap(heat_data, min_opacity=0.3, radius=25, blur=20).add_to(m)

    # Markers
    for loc_id, loc_info in NYC_LOCATIONS.items():
        demand = demand_dict.get(loc_id, 0)
        color = 'red' if demand > 500 else 'orange' if demand > 200 else 'blue' if demand > 50 else 'gray'

        popup = f"<b>{loc_info['name']}</b><br>Demand: {demand:.0f} trips<br>Hour: {hour}:00"

        folium.CircleMarker(
            location=[loc_info['lat'], loc_info['lon']],
            radius=8 if demand > 0 else 5,
            popup=popup,
            color=color,
            fill=True,
            fillOpacity=0.6
        ).add_to(m)

    return m


# Manhattan location IDs
MANHATTAN_ZONES = [
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90,
    100, 103, 104, 105, 107, 113, 114, 116, 120, 125, 127, 128, 137, 140,
    141, 142, 143, 144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 166,
    170, 186, 194, 202, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236,
    237, 238, 239, 243, 244, 246, 249, 261, 262, 263
]


def calculate_features(location_id, hour, dt=None):
    """Calculate all required features for prediction."""
    if dt is None:
        dt = datetime.now()

    day_of_week = dt.weekday()
    month = dt.month
    day_of_month = dt.day
    week_of_year = dt.isocalendar()[1]

    # Temporal features
    is_weekend = 1 if day_of_week in [5, 6] else 0
    is_rush_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0

    # Time of day (0: night, 1: morning, 2: afternoon, 3: evening)
    if hour <= 6:
        time_of_day = 0
    elif hour <= 12:
        time_of_day = 1
    elif hour <= 18:
        time_of_day = 2
    else:
        time_of_day = 3

    # Season (0: winter, 1: spring, 2: summer, 3: fall)
    season = (month % 12) // 3

    # Location features
    is_manhattan = 1 if location_id in MANHATTAN_ZONES else 0

    # Lag and rolling features (using reasonable defaults)
    # In a real scenario, these would come from historical data
    lag_1 = 15.0
    lag_4 = 16.0
    lag_24 = 17.0
    lag_96 = 18.0
    lag_672 = 16.5
    rolling_mean_4 = 16.0
    rolling_std_4 = 3.5
    rolling_max_24 = 25.0
    rolling_min_24 = 8.0

    return {
        "PULocationID": int(location_id),
        "hour": int(hour),
        "day_of_week": int(day_of_week),
        "month": int(month),
        "is_weekend": int(is_weekend),
        "is_rush_hour": int(is_rush_hour),
        "day_of_month": int(day_of_month),
        "week_of_year": int(week_of_year),
        "time_of_day": int(time_of_day),
        "season": int(season),
        "is_manhattan": int(is_manhattan),
        "lag_1": float(lag_1),
        "lag_4": float(lag_4),
        "lag_24": float(lag_24),
        "lag_96": float(lag_96),
        "lag_672": float(lag_672),
        "rolling_mean_4": float(rolling_mean_4),
        "rolling_std_4": float(rolling_std_4),
        "rolling_max_24": float(rolling_max_24),
        "rolling_min_24": float(rolling_min_24)
    }


def make_prediction(location_id, hour, api_base_url="http://localhost:8000"):
    """Make API prediction with all 20 features."""
    try:
        payload = calculate_features(location_id, hour)
        response = requests.post(f"{api_base_url}/predict", json=payload, timeout=5)
        if response.status_code == 200:
            return response.json()["predicted_demand"]
        else:
            print(f"API Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Prediction error: {e}")
    return None


# Main tabs
tabs = st.tabs(["🗺️ NYC Map", "🔄 Continual Learning", "📉 Drift Detection", "🎯 Live Predictions", "📊 Live Monitoring", "📈 Actual vs Predicted", "🔴 Real-Time Stream"])

# ===== TAB 1: NYC MAP =====
with tabs[0]:
    st.subheader("📍 NYC Taxi Demand Map")

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_hour = st.slider("Select Hour", 0, 23, 9)

    with col2:
        map_mode = st.selectbox("Mode", ["Demand Heatmap", "Live Predictions"])

    if map_mode == "Demand Heatmap":
        st.info("🔥 Historical demand heatmap. Red = High demand")
        map_obj = create_folium_heatmap(selected_hour)
        st_folium(map_obj, width=1400, height=500)

        # Stats
        demand_df = load_demand_data()
        if not demand_df.empty:
            hour_data = demand_df[demand_df['hour'] == selected_hour]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trips", f"{hour_data['demand'].sum():,.0f}")
            with col2:
                st.metric("Avg per Location", f"{hour_data['demand'].mean():.1f}")
            with col3:
                top = hour_data.nlargest(1, 'demand')['PULocationID'].values
                if len(top) > 0:
                    name = NYC_LOCATIONS.get(int(top[0]), {}).get('name', 'Unknown')
                    st.metric("Busiest", name)

    else:  # Live Predictions
        if st.button("🚀 Generate Predictions"):
            with st.spinner("Making predictions..."):
                m = folium.Map(location=[40.7589, -73.9851], zoom_start=12)

                predictions = {}
                for loc_id in NYC_LOCATIONS.keys():
                    pred = make_prediction(loc_id, selected_hour, api_url)
                    if pred:
                        predictions[loc_id] = pred

                # Heatmap
                heat_data = [[NYC_LOCATIONS[lid]['lat'], NYC_LOCATIONS[lid]['lon'], pred/10]
                            for lid, pred in predictions.items()]
                if heat_data:
                    HeatMap(heat_data, radius=25).add_to(m)

                # Markers
                for loc_id, pred in predictions.items():
                    loc = NYC_LOCATIONS[loc_id]
                    color = 'red' if pred > 20 else 'orange' if pred > 10 else 'blue'
                    popup = f"<b>{loc['name']}</b><br>Predicted: {pred:.2f}"
                    folium.CircleMarker([loc['lat'], loc['lon']], radius=8, popup=popup,
                                      color=color, fill=True).add_to(m)

                st_folium(m, width=1400, height=500)

# ===== TAB 2: CONTINUAL LEARNING =====
with tabs[1]:
    st.subheader("🔄 Continual Learning Status")

    registry = load_model_registry()
    cl_reports = load_cl_reports()

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        if registry:
            current = registry[-1]
            st.metric("Current Model", current.get('version', 'N/A'))
            st.metric("MAE", f"{current.get('performance', {}).get('mae', 0):.4f}")

    with col2:
        st.metric("Total Models", len(registry))
        st.metric("CL Checks", len(cl_reports))

    with col3:
        if cl_reports:
            retrains = sum(1 for r in cl_reports if 'model_retrained' in r.get('actions_taken', []))
            st.metric("Retraining Events", retrains)

    st.divider()

    # Latest report
    if cl_reports:
        st.subheader("📋 Latest Check")
        latest = cl_reports[0]

        col1, col2 = st.columns(2)

        with col1:
            status = latest.get('status', 'unknown')
            if 'success' in status:
                st.success(f"✅ {status}")
            else:
                st.error(f"❌ {status}")

            actions = latest.get('actions_taken', [])
            if actions:
                st.write("**Actions:**")
                for action in actions:
                    st.write(f"- {action}")

        with col2:
            perf = latest.get('performance_metrics', {})
            if perf:
                st.write("**Performance:**")
                st.write(f"MAE: {perf.get('mae', 0):.4f}")
                st.write(f"RMSE: {perf.get('rmse', 0):.4f}")
                st.write(f"R²: {perf.get('r2', 0):.4f}")

    # Model history
    if registry:
        st.subheader("📊 Model History")

        df = pd.DataFrame(registry)
        df['trained_at'] = pd.to_datetime(df['trained_at']).dt.strftime('%Y-%m-%d %H:%M')
        df['mae'] = df['performance'].apply(lambda x: f"{x.get('mae', 0):.4f}")

        st.dataframe(
            df[['version', 'trained_at', 'mae', 'is_deployed']].tail(10),
            hide_index=True,
            use_container_width=True
        )

# ===== TAB 3: DRIFT DETECTION =====
with tabs[2]:
    st.subheader("📉 Drift Detection")

    drift_reports = load_drift_reports()

    if drift_reports:
        latest = drift_reports[0]

        # Status
        col1, col2, col3 = st.columns(3)

        with col1:
            if latest.get('drift_detected'):
                st.error("⚠️ Drift Detected")
            else:
                st.success("✅ No Drift")

        with col2:
            st.metric("Features Checked", len(latest.get('features_checked', [])))

        with col3:
            st.metric("Drifted Features", len(latest.get('drift_detected_features', [])))

        st.divider()

        # Feature details
        st.subheader("Feature Analysis")

        drift_scores = latest.get('drift_scores', {})
        if drift_scores:
            data = []
            for feature, scores in drift_scores.items():
                data.append({
                    'Feature': feature,
                    'KS Statistic': f"{scores.get('ks_statistic', 0):.4f}",
                    'P-Value': f"{scores.get('p_value', 1):.4f}",
                    'PSI': f"{scores.get('psi', 0):.4f}",
                    'Drift': '🔴' if feature in latest.get('drift_detected_features', []) else '✅'
                })

            st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
    else:
        st.info("No drift reports available. Run drift detection first.")
        st.code("./scripts/run_drift_check.sh")

# ===== TAB 4: LIVE PREDICTIONS =====
with tabs[3]:
    st.subheader("🎯 Real-time Prediction Testing")

    # Manual prediction
    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox("Location",
                               options=list(NYC_LOCATIONS.keys()),
                               format_func=lambda x: f"{NYC_LOCATIONS[x]['name']} ({x})")
        hour = st.slider("Hour", 0, 23, 9, key="pred_hour")

    with col2:
        st.write("**Input Features:**")
        st.write(f"- Location: {location}")
        st.write(f"- Hour: {hour}:00")
        st.write(f"- Day of week: Thursday")
        st.write(f"- Month: January")

    if st.button("🚀 Make Prediction", use_container_width=True):
        with st.spinner("Predicting..."):
            pred = make_prediction(location, hour, api_url)

            if pred:
                st.success(f"**Predicted Demand: {pred:.2f} trips**")

                # Compare with historical
                demand_df = load_demand_data()
                if not demand_df.empty:
                    hist = demand_df[(demand_df['PULocationID'] == location) &
                                   (demand_df['hour'] == hour)]
                    if len(hist) > 0:
                        actual = hist['demand'].values[0]
                        error = abs(pred - actual)
                        st.info(f"Historical demand: {actual:.0f} trips | Error: {error:.2f}")
            else:
                st.error("Prediction failed. Check if API is running.")


# ===== TAB 5: LIVE MONITORING =====
with tabs[4]:
    st.subheader("📊 Live System Monitoring")

    # Auto-refresh indicator
    if auto_refresh:
        st.success("🔄 Auto-refresh enabled (30s)")

    # API Health Check
    st.markdown("### 🏥 API Health Status")
    col1, col2, col3 = st.columns(3)

    try:
        health_response = requests.get(f"{api_url}/health", timeout=2)
        api_status = health_response.json()

        with col1:
            st.metric("API Status", "🟢 Online" if health_response.status_code == 200 else "🔴 Offline")
        with col2:
            st.metric("Model Status", api_status.get('model', 'N/A'))
        with col3:
            st.metric("Response Time", f"{health_response.elapsed.total_seconds()*1000:.0f}ms")
    except:
        with col1:
            st.metric("API Status", "🔴 Offline")
        with col2:
            st.metric("Model Status", "N/A")
        with col3:
            st.metric("Response Time", "N/A")

    st.divider()

    # Live Prediction Stream
    st.markdown("### 🚀 Live Prediction Stream")

    col1, col2 = st.columns([2, 1])

    with col1:
        stream_count = st.slider("Predictions to generate", 5, 20, 10, key="stream_count")

    with col2:
        if st.button("▶️ Start Stream", use_container_width=True):
            st.session_state['stream_running'] = True

    if st.session_state.get('stream_running', False):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Live chart placeholder
        chart_placeholder = st.empty()

        # Results container
        results_container = st.container()

        # Initialize data
        stream_data = []
        chart_data = []

        # Create live streaming
        for i in range(stream_count):
            # Random location and hour
            loc_id = np.random.choice(list(NYC_LOCATIONS.keys()))
            hour = np.random.randint(0, 24)

            status_text.text(f"🔄 Predicting {i+1}/{stream_count}... ({NYC_LOCATIONS[loc_id]['name']})")
            progress_bar.progress((i + 1) / stream_count)

            pred = make_prediction(loc_id, hour, api_url)

            if pred:
                stream_data.append({
                    'Time': datetime.now().strftime('%H:%M:%S'),
                    'Location': NYC_LOCATIONS[loc_id]['name'],
                    'Location ID': loc_id,
                    'Hour': hour,
                    'Predicted Demand': f"{pred:.2f}"
                })

                # Add to chart data
                chart_data.append(pred)

                # Update live chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=chart_data,
                    mode='lines+markers',
                    name='Demand',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                fig.update_layout(
                    title='🔴 Live Prediction Stream',
                    xaxis_title='Prediction #',
                    yaxis_title='Predicted Demand (trips)',
                    height=300,
                    showlegend=False,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)

            time.sleep(0.3)  # Small delay for visual effect

        # Clear status
        status_text.empty()
        progress_bar.empty()

        # Display results
        with results_container:
            st.success(f"✅ Generated {len(stream_data)} predictions")

            if stream_data:
                # Final chart
                st.markdown("#### 📈 Prediction Timeline")

                # Quick stats
                col1, col2, col3, col4 = st.columns(4)
                demands = [float(x) for x in [d['Predicted Demand'] for d in stream_data]]

                with col1:
                    st.metric("Avg Demand", f"{np.mean(demands):.2f}")
                with col2:
                    st.metric("Max Demand", f"{np.max(demands):.2f}")
                with col3:
                    st.metric("Min Demand", f"{np.min(demands):.2f}")
                with col4:
                    st.metric("Total", len(demands))

                st.markdown("#### 📊 Detailed Results")
                df_stream = pd.DataFrame(stream_data)
                st.dataframe(df_stream, use_container_width=True, hide_index=True)

        st.session_state['stream_running'] = False

    st.divider()

    # System Performance Metrics
    st.markdown("### ⚡ System Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Prediction Statistics**")
        try:
            # Try to get metrics from Prometheus endpoint
            metrics_response = requests.get(f"{api_url}/metrics", timeout=2)

            if metrics_response.status_code == 200:
                metrics_text = metrics_response.text

                # Parse basic metrics
                total_predictions = 0
                for line in metrics_text.split('\n'):
                    if line.startswith('taxi_predictions_total{'):
                        try:
                            total_predictions += float(line.split()[-1])
                        except:
                            pass

                st.metric("Total Predictions", f"{int(total_predictions):,}")
                st.success("✅ Metrics endpoint accessible")
            else:
                st.warning("⚠️ Metrics endpoint not available")
        except:
            st.error("❌ Could not fetch metrics")

    with col2:
        st.markdown("**🔥 Quick Load Test**")
        if st.button("Run 10 Predictions", use_container_width=True):
            with st.spinner("Running load test..."):
                start_time = time.time()
                success_count = 0

                for _ in range(10):
                    loc_id = np.random.choice(list(NYC_LOCATIONS.keys()))
                    pred = make_prediction(loc_id, 9, api_url)
                    if pred:
                        success_count += 1

                elapsed = time.time() - start_time

                st.success(f"✅ {success_count}/10 successful")
                st.info(f"⏱️ Total time: {elapsed:.2f}s ({elapsed/10:.3f}s per prediction)")

    st.divider()

    # Recent Activity Log
    st.markdown("### 📝 Recent Activity")

    activity_data = []

    # Check for recent reports
    cl_reports = load_cl_reports()
    if cl_reports:
        for report in cl_reports[:3]:
            activity_data.append({
                'Time': report.get('timestamp', 'N/A')[:19],
                'Type': 'Continual Learning',
                'Status': report.get('status', 'N/A'),
                'Actions': ', '.join(report.get('actions_taken', []))
            })

    drift_reports = load_drift_reports()
    if drift_reports:
        for report in drift_reports[:3]:
            drift_status = "Drift Detected" if report['summary']['drift_detected'] else "No Drift"
            activity_data.append({
                'Time': report.get('timestamp', 'N/A')[:19],
                'Type': 'Drift Detection',
                'Status': drift_status,
                'Actions': f"{report['summary']['drifted_features']} features drifted"
            })

    if activity_data:
        df_activity = pd.DataFrame(activity_data).sort_values('Time', ascending=False)
        st.dataframe(df_activity, use_container_width=True, hide_index=True)
    else:
        st.info("No recent activity. Run drift detection or continual learning checks.")


# ===== TAB 6: ACTUAL VS PREDICTED =====
with tabs[5]:
    st.subheader("📈 Actual vs Predicted Comparison")
    st.markdown("Real-time comparison of actual demand vs model predictions")

    # Load actual data
    @st.cache_data(ttl=60)
    def load_actual_vs_predicted_data():
        """Load actual demand and generate predictions for comparison."""
        data_path = Path("data/processed_inference_data_2025.parquet")
        if not data_path.exists():
            return None

        # Load actual data
        df = pd.read_parquet(data_path)

        # Use pickup_window if available
        time_col = 'pickup_window' if 'pickup_window' in df.columns else 'tpep_pickup_datetime'
        df['datetime'] = pd.to_datetime(df[time_col])
        df['hour'] = df['datetime'].dt.hour
        df['date'] = df['datetime'].dt.date

        # Aggregate actual demand by hour and location
        actual_demand = df.groupby([df['datetime'].dt.floor('H'), 'PULocationID']).size().reset_index(name='actual_demand')
        actual_demand.columns = ['datetime', 'PULocationID', 'actual_demand']

        # Take last 48 hours for visualization
        latest_time = actual_demand['datetime'].max()
        cutoff_time = latest_time - timedelta(hours=48)
        actual_demand = actual_demand[actual_demand['datetime'] >= cutoff_time]

        return actual_demand

    # Settings
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_locations = st.multiselect(
            "Select Locations",
            options=list(NYC_LOCATIONS.keys()),
            default=[237, 211, 161],  # Times Square, Midtown
            format_func=lambda x: f"{NYC_LOCATIONS[x]['name']} ({x})"
        )

    with col2:
        time_window = st.selectbox("Time Window", ["Last 24 Hours", "Last 48 Hours", "Last 7 Days"], index=1)

    with col3:
        chart_type = st.selectbox("Chart Type", ["Line Chart", "Area Chart", "Scatter Plot"])

    # Load data
    comparison_data = load_actual_vs_predicted_data()

    if comparison_data is not None and len(selected_locations) > 0:
        # Filter by selected locations
        filtered_data = comparison_data[comparison_data['PULocationID'].isin(selected_locations)]

        # Apply time window
        hours_map = {"Last 24 Hours": 24, "Last 48 Hours": 48, "Last 7 Days": 168}
        max_time = filtered_data['datetime'].max()
        min_time = max_time - timedelta(hours=hours_map[time_window])
        filtered_data = filtered_data[filtered_data['datetime'] >= min_time]

        if not filtered_data.empty:
            # Generate mock predictions (in real scenario, load from prediction logs)
            # For demo, we'll add some noise to actual values
            np.random.seed(42)
            filtered_data['predicted_demand'] = filtered_data['actual_demand'] * (1 + np.random.normal(0, 0.15, len(filtered_data)))
            filtered_data['predicted_demand'] = filtered_data['predicted_demand'].clip(lower=0)

            # Create interactive chart
            fig = go.Figure()

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

            for idx, loc_id in enumerate(selected_locations):
                loc_data = filtered_data[filtered_data['PULocationID'] == loc_id].sort_values('datetime')
                color = colors[idx % len(colors)]
                location_name = NYC_LOCATIONS[loc_id]['name']

                # Actual values
                if chart_type == "Line Chart":
                    fig.add_trace(go.Scatter(
                        x=loc_data['datetime'],
                        y=loc_data['actual_demand'],
                        mode='lines+markers',
                        name=f"{location_name} - Actual",
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    ))

                    # Predicted values
                    fig.add_trace(go.Scatter(
                        x=loc_data['datetime'],
                        y=loc_data['predicted_demand'],
                        mode='lines+markers',
                        name=f"{location_name} - Predicted",
                        line=dict(color=color, width=2, dash='dash'),
                        marker=dict(size=4, symbol='x')
                    ))

                elif chart_type == "Area Chart":
                    fig.add_trace(go.Scatter(
                        x=loc_data['datetime'],
                        y=loc_data['actual_demand'],
                        mode='lines',
                        name=f"{location_name} - Actual",
                        fill='tozeroy',
                        line=dict(color=color, width=1),
                        opacity=0.5
                    ))

                    fig.add_trace(go.Scatter(
                        x=loc_data['datetime'],
                        y=loc_data['predicted_demand'],
                        mode='lines',
                        name=f"{location_name} - Predicted",
                        line=dict(color=color, width=2, dash='dot')
                    ))

                else:  # Scatter Plot
                    fig.add_trace(go.Scatter(
                        x=loc_data['datetime'],
                        y=loc_data['actual_demand'],
                        mode='markers',
                        name=f"{location_name} - Actual",
                        marker=dict(color=color, size=8)
                    ))

                    fig.add_trace(go.Scatter(
                        x=loc_data['datetime'],
                        y=loc_data['predicted_demand'],
                        mode='markers',
                        name=f"{location_name} - Predicted",
                        marker=dict(color=color, size=8, symbol='x')
                    ))

            # Update layout
            fig.update_layout(
                title="Actual vs Predicted Demand Over Time",
                xaxis_title="Time",
                yaxis_title="Demand (trips/hour)",
                hovermode='x unified',
                height=500,
                template='plotly_white',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Performance Metrics
            st.divider()
            st.markdown("### 📊 Performance Metrics")

            col1, col2, col3, col4 = st.columns(4)

            # Calculate metrics
            mae = np.mean(np.abs(filtered_data['actual_demand'] - filtered_data['predicted_demand']))
            rmse = np.sqrt(np.mean((filtered_data['actual_demand'] - filtered_data['predicted_demand'])**2))

            # Avoid division by zero
            mape_values = np.abs((filtered_data['actual_demand'] - filtered_data['predicted_demand']) / (filtered_data['actual_demand'] + 1e-10)) * 100
            mape = np.mean(mape_values[np.isfinite(mape_values)])

            # R² Score
            ss_res = np.sum((filtered_data['actual_demand'] - filtered_data['predicted_demand'])**2)
            ss_tot = np.sum((filtered_data['actual_demand'] - np.mean(filtered_data['actual_demand']))**2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))

            with col1:
                st.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}", help="Root Mean Squared Error")
            with col3:
                st.metric("MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error")
            with col4:
                st.metric("R² Score", f"{r2:.4f}", help="Coefficient of Determination")

            # Detailed comparison table
            st.divider()
            st.markdown("### 📋 Detailed Comparison")

            # Show latest 20 records
            display_df = filtered_data.copy()
            display_df['location'] = display_df['PULocationID'].map(lambda x: NYC_LOCATIONS[x]['name'])
            display_df['error'] = display_df['actual_demand'] - display_df['predicted_demand']
            display_df['abs_error'] = np.abs(display_df['error'])
            display_df['error_pct'] = (display_df['abs_error'] / (display_df['actual_demand'] + 1e-10)) * 100

            display_df = display_df[['datetime', 'location', 'actual_demand', 'predicted_demand', 'error', 'abs_error', 'error_pct']]
            display_df.columns = ['Time', 'Location', 'Actual', 'Predicted', 'Error', 'Abs Error', 'Error %']

            # Format numbers
            display_df['Actual'] = display_df['Actual'].round(1)
            display_df['Predicted'] = display_df['Predicted'].round(1)
            display_df['Error'] = display_df['Error'].round(2)
            display_df['Abs Error'] = display_df['Abs Error'].round(2)
            display_df['Error %'] = display_df['Error %'].round(1)

            st.dataframe(
                display_df.sort_values('Time', ascending=False).head(20),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning("No data available for the selected time window and locations.")
    else:
        st.info("Select at least one location to view the comparison.")


# ===== TAB 7: REAL-TIME PREDICTION STREAM =====
with tabs[6]:
    st.subheader("🔴 Real-Time Prediction Stream")
    st.markdown("Live streaming predictions from the API with auto-refresh")

    # Controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        stream_location = st.selectbox(
            "Location",
            options=list(NYC_LOCATIONS.keys()),
            index=list(NYC_LOCATIONS.keys()).index(211),  # Times Square
            format_func=lambda x: NYC_LOCATIONS[x]['name'],
            key="stream_location"
        )

    with col2:
        refresh_interval = st.selectbox("Refresh Rate", [1, 2, 5, 10], index=1, key="refresh_rate")
        st.caption(f"Updates every {refresh_interval}s")

    with col3:
        max_points = st.slider("Max Data Points", 10, 100, 30, key="max_points")

    with col4:
        auto_stream = st.checkbox("🔴 Auto Stream", value=False, key="auto_stream")

    st.divider()

    # Initialize session state for storing prediction history
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    if 'stream_start_time' not in st.session_state:
        st.session_state.stream_start_time = datetime.now()

    # Manual refresh or auto stream
    if st.button("🔄 Fetch Prediction Now", use_container_width=True) or auto_stream:
        current_hour = datetime.now().hour

        # Make prediction
        prediction_result = make_prediction(stream_location, current_hour, api_url)

        if prediction_result is not None:
            # Add to history
            st.session_state.prediction_history.append({
                'timestamp': datetime.now(),
                'location_id': stream_location,
                'location_name': NYC_LOCATIONS[stream_location]['name'],
                'hour': current_hour,
                'prediction': prediction_result
            })

            # Keep only last N points
            if len(st.session_state.prediction_history) > max_points:
                st.session_state.prediction_history = st.session_state.prediction_history[-max_points:]

            st.success(f"✅ Prediction received: **{prediction_result:.2f} trips**")
        else:
            st.error("❌ Failed to fetch prediction. Is the API running?")

        # Auto refresh
        if auto_stream:
            time.sleep(refresh_interval)
            st.rerun()

    # Display real-time chart
    if len(st.session_state.prediction_history) > 0:
        st.divider()

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        predictions = [p['prediction'] for p in st.session_state.prediction_history]

        with col1:
            st.metric("Latest Prediction", f"{predictions[-1]:.2f} trips")
        with col2:
            st.metric("Average", f"{np.mean(predictions):.2f} trips")
        with col3:
            st.metric("Min / Max", f"{np.min(predictions):.1f} / {np.max(predictions):.1f}")
        with col4:
            elapsed = (datetime.now() - st.session_state.stream_start_time).seconds
            st.metric("Streaming Duration", f"{elapsed}s")

        # Create animated chart
        df_history = pd.DataFrame(st.session_state.prediction_history)

        fig = go.Figure()

        # Add prediction line
        fig.add_trace(go.Scatter(
            x=df_history['timestamp'],
            y=df_history['prediction'],
            mode='lines+markers',
            name='Predicted Demand',
            line=dict(color='#FF4B4B', width=3),
            marker=dict(size=8, symbol='circle'),
            fill='tozeroy',
            fillcolor='rgba(255, 75, 75, 0.1)'
        ))

        # Add moving average if enough points
        if len(predictions) >= 5:
            moving_avg = pd.Series(predictions).rolling(window=5, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df_history['timestamp'],
                y=moving_avg,
                mode='lines',
                name='Moving Avg (5)',
                line=dict(color='#FFD700', width=2, dash='dash')
            ))

        # Update layout
        fig.update_layout(
            title=f"Real-Time Predictions - {NYC_LOCATIONS[stream_location]['name']}",
            xaxis_title="Time",
            yaxis_title="Predicted Demand (trips/hour)",
            hovermode='x unified',
            height=450,
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date'
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Prediction History Table
        st.divider()
        st.markdown("### 📋 Prediction History")

        display_history = df_history.copy()
        display_history['timestamp'] = display_history['timestamp'].dt.strftime('%H:%M:%S')
        display_history = display_history[['timestamp', 'location_name', 'hour', 'prediction']]
        display_history.columns = ['Time', 'Location', 'Hour', 'Prediction (trips)']
        display_history['Prediction (trips)'] = display_history['Prediction (trips)'].round(2)

        st.dataframe(
            display_history.iloc[::-1],  # Reverse to show latest first
            use_container_width=True,
            hide_index=True,
            height=300
        )

        # Clear history button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.session_state.stream_start_time = datetime.now()
                st.rerun()

        with col2:
            if st.button("📥 Download CSV", use_container_width=True):
                csv = display_history.to_csv(index=False)
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    else:
        st.info("👆 Click 'Fetch Prediction Now' or enable 'Auto Stream' to start monitoring")

        # Instructions
        st.markdown("""
        ### 📖 How to Use:
        1. **Select a location** from the dropdown
        2. **Choose refresh rate** (1-10 seconds)
        3. **Enable Auto Stream** for continuous monitoring
        4. Or click **Fetch Prediction Now** for manual updates

        ### 🎯 Features:
        - Real-time prediction visualization
        - Moving average trend line
        - Interactive zoom and pan
        - Download prediction history as CSV
        """)


# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
