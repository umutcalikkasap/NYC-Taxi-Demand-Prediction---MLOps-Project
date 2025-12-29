# 🎬 NYC Taxi Pulse - Demo Guide

**Quick reference for presenting the project**

---

## ⚡ Quick Setup (5 minutes before demo)

### Step 1: Start API
```bash
cd /Users/umut/Desktop/25-26\ Fall/MLOPS/project/nyc-taxi-pulse-project
./start_api.sh
```

**Wait for:**
```
Model loaded: ✓
Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Start Dashboard
```bash
# New terminal window
./start_dashboard.sh
```

**Wait for:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### Step 3: Open in Browser
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs

### Step 4: Pre-check
- [ ] API health: `curl http://localhost:8000/health`
- [ ] Dashboard loads
- [ ] All 7 tabs visible

---

## 🎯 Demo Flow (15 minutes)

### PART 1: Introduction (2 min)

**What to say:**
> "This is NYC Taxi Pulse, a production-ready ML system for predicting taxi demand across NYC. It demonstrates end-to-end MLOps practices including real-time predictions, continual learning, and drift detection."

**Show:**
- Browser: Dashboard home page
- 7 tabs overview

---

### PART 2: Real-Time Predictions (4 min)

#### Tab 7: 🔴 Real-Time Stream

**What to do:**
1. Click **Tab 7: Real-Time Stream**
2. Select **Times Square** location
3. Set **Refresh Rate: 2 seconds**
4. Check **🔴 Auto Stream**

**What to say:**
> "This tab shows live predictions streaming from the API. The red line shows predicted demand updating every 2 seconds, and the yellow dashed line shows a moving average trend."

**What to show:**
- ✅ Watch graph update automatically
- ✅ Point to metrics: Latest, Average, Min/Max
- ✅ Scroll down to Prediction History table
- ✅ Note: "In production, this would monitor real taxi demand 24/7"

**Pro tip:** Let it run for ~30 seconds while you talk

---

### PART 3: Data Visualization (3 min)

#### Tab 1: 🗺️ NYC Map

**What to do:**
1. Click **Tab 1: NYC Map**
2. Move **hour slider**: 8am → 12pm → 6pm → 11pm
3. Show heatmap changes

**What to say:**
> "The heatmap shows historical demand patterns. Red areas indicate high demand - notice how it shifts from residential areas in the morning to business districts during the day, and entertainment areas at night."

**What to show:**
- ✅ Morning (8am): Residential areas active
- ✅ Noon (12pm): Midtown busy
- ✅ Evening (6pm): Times Square, business districts
- ✅ Night (11pm): Concentrated in entertainment zones

---

### PART 4: Performance Analysis (3 min)

#### Tab 6: 📈 Actual vs Predicted

**What to do:**
1. Click **Tab 6: Actual vs Predicted**
2. Locations: Times Square, Midtown East (already selected)
3. Time Window: **Last 48 Hours**
4. Chart Type: **Line Chart**

**What to say:**
> "This shows how accurate our predictions are. The solid lines are actual demand, dashed lines are predictions. We achieve 96% R² score with MAE of only 2.32 trips."

**What to show:**
- ✅ Point to metrics: MAE 2.32, R² 0.96
- ✅ Show how lines follow each other closely
- ✅ Switch to **Area Chart** for different view
- ✅ Scroll to comparison table

---

### PART 5: MLOps Features (3 min)

#### Tab 3: 📉 Drift Detection

**What to do:**
1. Click **Tab 3: Drift Detection**
2. Show drift summary

**What to say:**
> "This is our data quality monitoring. We detected drift in 17 out of 20 features - this means the data distribution has changed over time. This triggers our continual learning pipeline."

**What to show:**
- ✅ Drift Status: **Drift Detected**
- ✅ Features Checked: **20**
- ✅ Drifted Features: **17**
- ✅ Scroll to feature table
- ✅ Point to KS statistic and P-value columns

#### Tab 2: 🔄 Continual Learning

**What to do:**
1. Click **Tab 2: Continual Learning**

**What to say:**
> "When drift is detected or performance degrades, our continual learning pipeline automatically retrains the model. This ensures predictions stay accurate as taxi demand patterns evolve."

**What to show:**
- ✅ Current Model: v1.0.0
- ✅ Performance Metrics: MAE 2.32
- ✅ Latest Check report
- ✅ Decision: Performance healthy despite drift

---

### PART 6: API Demo (2 min - Optional)

**What to do:**
1. Open http://localhost:8000/docs in new tab
2. Click **POST /predict**
3. Click **Try it out**
4. Use example payload
5. Click **Execute**

**What to say:**
> "The system exposes a production-ready API. Here's the Swagger documentation. I can make a prediction request with all 20 features and get an instant response."

**What to show:**
- ✅ Request body (20 features)
- ✅ Response: predicted_demand
- ✅ Response time (~50ms)

---

## 💡 Key Talking Points

### Technical Highlights
- **Model**: XGBoost with 20 engineered features
- **Performance**: 96.26% R² score, 2.32 MAE
- **MLOps**: Continual learning, drift detection, model registry
- **Architecture**: FastAPI backend + Streamlit dashboard
- **Real-time**: Auto-refreshing predictions every 2 seconds

### Business Value
- **Predictive**: Helps taxi companies optimize fleet allocation
- **Adaptive**: Model automatically retrains when patterns change
- **Monitored**: Drift detection ensures data quality
- **Scalable**: API handles 200+ requests/second
- **Actionable**: Real-time insights for operational decisions

---

## 🎨 Visual Demo Tips

### Dashboard Navigation
1. **Start with Tab 7** (Real-Time Stream) - most impressive
2. **Show Tab 1** (Map) - most visual
3. **Demonstrate Tab 6** (Actual vs Predicted) - shows accuracy
4. **Explain Tab 3 & 2** (MLOps) - shows sophistication

### Timing
- Spend most time on **Tab 7** (Real-Time Stream)
- Keep Tab 1 (Map) fast - just slide through hours
- Tab 6 should focus on metrics
- MLOps tabs - explain concepts, don't dive deep

### Common Questions

**Q: How often does the model retrain?**
> "Automatically when MAE exceeds 2.5, performance drops 20%, or drift is detected. We use a sliding window of the last 30 days of production data."

**Q: What features are most important?**
> "Lag features (past demand) and temporal features (hour, is_rush_hour, is_weekend) are the top predictors."

**Q: How do you handle real-time data?**
> "We stream predictions through the API. The dashboard connects via WebSocket for live updates."

**Q: What about deployment?**
> "We use FastAPI for the backend, which is production-ready. It can be containerized with Docker and deployed on any cloud platform."

---

## 🚨 Troubleshooting

### Issue: API not responding
```bash
# Check if running
curl http://localhost:8000/health

# Restart
pkill -f uvicorn
./start_api.sh
```

### Issue: Dashboard shows error
```bash
# Check API URL in sidebar
# Should be: http://localhost:8000

# Restart dashboard
pkill -f streamlit
./start_dashboard.sh
```

### Issue: Auto Stream not working
- **Check:** API is running
- **Check:** API URL is correct in sidebar
- **Try:** Manual "Fetch Prediction Now" first

---

## 📝 Presentation Script (30 seconds)

> "NYC Taxi Pulse predicts taxi demand across New York City using machine learning. What makes it special is the complete MLOps pipeline: real-time predictions via API, automatic model retraining through continual learning, and drift detection for data quality. As you can see in the Real-Time Stream tab, predictions update every 2 seconds. Our XGBoost model achieves 96% accuracy, and when performance degrades or data drifts, the system automatically retrains itself. This demonstrates production-ready machine learning with monitoring, versioning, and adaptive learning - everything you need for a real-world ML system."

---

## ✅ Pre-Demo Checklist

**5 minutes before:**
- [ ] Close all unnecessary browser tabs
- [ ] Increase browser zoom to 125% for visibility
- [ ] Open terminal windows (API + Dashboard)
- [ ] Start API
- [ ] Start Dashboard
- [ ] Load http://localhost:8501
- [ ] Test one prediction manually
- [ ] Prepare to open Tab 7 first

**During demo:**
- [ ] Speak clearly and confidently
- [ ] Point to specific metrics on screen
- [ ] Don't rush through tabs
- [ ] Pause for questions
- [ ] Keep it under 15 minutes

---

**Good luck with your demo! 🎯**

Remember: The auto-streaming feature (Tab 7) is your show-stopper. Start there to grab attention!
