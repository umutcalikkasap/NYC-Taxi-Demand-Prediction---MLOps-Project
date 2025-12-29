# ⚡ Quick Start Guide

Get NYC Taxi Pulse up and running in **3 minutes**.

---

## 🎯 Prerequisites

- Python 3.11+ with conda
- ~1GB free disk space
- Web browser

---

## 🚀 Steps

### 1. Activate Environment
```bash
conda activate mlops
```

### 2. Navigate to Project
```bash
cd /Users/umut/Desktop/25-26\ Fall/MLOPS/project/nyc-taxi-pulse-project
```

### 3. Start API (Terminal 1)
```bash
./start_api.sh
```

**Wait for:**
```
Model loaded: ✓
Uvicorn running on http://0.0.0.0:8000
```

### 4. Start Dashboard (Terminal 2)
```bash
./start_dashboard.sh
```

**Wait for:**
```
Local URL: http://localhost:8501
```

### 5. Open Dashboard
```
http://localhost:8501
```

---

## ✅ Verify

**Check API:**
```bash
curl http://localhost:8000/health
```

**Should return:**
```json
{"status":"healthy","model_loaded":true}
```

---

## 🎬 First Demo

1. **Go to Tab 7**: 🔴 Real-Time Stream
2. **Location**: Times Square
3. **Refresh Rate**: 2 seconds
4. **Enable**: 🔴 Auto Stream checkbox
5. **Watch**: Live predictions streaming!

---

## 📊 Explore Other Tabs

- **Tab 1**: 🗺️ NYC demand heatmap
- **Tab 6**: 📈 Actual vs Predicted comparison
- **Tab 3**: 📉 Drift detection analysis
- **Tab 2**: 🔄 Continual learning status

---

## 🛑 Stop Services

**In each terminal:**
```
Ctrl + C
```

**Or kill all:**
```bash
pkill -f uvicorn
pkill -f streamlit
```

---

## 🔧 Troubleshooting

### API won't start
```bash
# Check port
lsof -i :8000

# Kill if busy
pkill -f uvicorn

# Restart
./start_api.sh
```

### Dashboard errors
```bash
# Check API is running
curl http://localhost:8000/health

# Restart dashboard
pkill -f streamlit
./start_dashboard.sh
```

### Predictions fail
1. Check sidebar API URL: `http://localhost:8000`
2. Verify API is running
3. Try manual prediction in Tab 4 first

---

## 📚 Next Steps

- **Full README**: [README.md](README.md)
- **Demo Guide**: [DEMO_GUIDE.md](DEMO_GUIDE.md)
- **API Docs**: http://localhost:8000/docs

---

**That's it! You're ready to go! 🎉**
