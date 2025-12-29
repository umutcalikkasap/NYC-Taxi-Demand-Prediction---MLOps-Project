# 🎬 Sunum için Komutlar - Hızlı Referans

## 🚀 Hızlı Başlangıç (Sunum Öncesi Hazırlık)

```bash
# 1. Production stack'i başlat
./scripts/start_production.sh

# 2. Tarayıcıda şu sekmeleri aç:
# http://localhost:8000/docs     (API)
# http://localhost:3000           (Grafana - admin/admin)
# http://localhost:9090           (Prometheus)
# http://localhost:5000           (MLflow)
```

---

## 📊 Demo 1: Model Eğitimi & Metrikler

### Geliştirilmiş Metriklerle Eğitim
```bash
python -m src.training.trainer
```

**Göstereceğiniz metrikler:**
- ✅ MAE (Mean Absolute Error)
- ✅ RMSE (Root Mean Squared Error)
- ✅ MAPE (Mean Absolute Percentage Error)
- ✅ R² Score (Coefficient of Determination)
- ✅ MedAE (Median Absolute Error)
- ✅ Bias (Overestimation/Underestimation)
- ✅ Within ±5 trips accuracy
- ✅ Within ±10% accuracy

---

## 🏆 Demo 2: Model Karşılaştırması

### 3 Model Baseline Comparison
```bash
python -m src.training.model_comparison
```

**Karşılaştırılan modeller:**
- XGBoost (Current)
- LightGBM (Alternative)
- CatBoost (Alternative)

**Sonuç:** Tablo formatında tüm metriklerde karşılaştırma + winner

### MLflow UI
```bash
mlflow ui
# http://localhost:5000
```

**MLflow'da göstereceğiniz:**
- Tüm experiment runs
- Metric comparison charts
- Model parameters
- Best model selection

---

## 🔥 Demo 3: Real-time Streaming Dashboard (EN ETKİLEYİCİ!)

### Dashboard'u Başlat
```bash
./scripts/demo_dashboard.sh
# veya
streamlit run src/dashboard/realtime_dashboard.py
# http://localhost:8501
```

### Mode 1: Manual Input
1. Değerleri manuel gir
2. "Make Prediction" butonuna tıkla
3. Sonuçları göster

### Mode 2: Auto Simulation ⭐⭐⭐
1. "Auto Simulation" seç
2. Speed: 3-5 predictions/sec
3. Max predictions: 40-50
4. "Start Simulation" → **CANLI GRAFİKLERİ İZLET!**

**Canlı grafiklerde göstereceğiniz:**
- ✅ Prediction vs Actual (time series)
- ✅ Error over time (error chart)
- ✅ Demand distribution (histogram)
- ✅ Live MAE/RMSE updates

**Bu çok etkileyici olacak!** 🎯

---

## 🗽 Demo 3B: NYC Map Dashboard ⭐⭐⭐ (YENİ - ÇOK ETKİLEYİCİ!)

### Dashboard'u Başlat
```bash
./scripts/start_map_dashboard.sh
# http://localhost:8501
```

**4 Farklı Görselleştirme Modu:**

### 1. Demand Heatmap 🔥
- NYC haritası üzerinde location-based demand
- 60+ Manhattan lokasyonu + airports
- Hover ile her location için detay
- Saat bazlı demand değişimi
- Top 10 en yoğun lokasyonlar listesi

### 2. Live Predictions 🔮
- Tüm NYC lokasyonları için real-time tahmin
- API ile entegre predictions
- Harita üzerinde tahmin değerleri (bubble size)
- Interactive Plotly visualization

### 3. Drift Analysis 🔍
- Son drift raporlarını harita üzerinde göster
- Lokasyon bazlı drift analizi

### 4. Comparison ⚖️
- Predicted vs Actual scatter plot
- Location-based karşılaştırma
- Perfect prediction line ile comparison

**Ek Features:**
- 📊 Analytics tab: Hourly demand trends, top locations
- 📈 Time Series tab: Location-specific demand patterns
- 🗺️ Borough filtering (Manhattan, Queens, Brooklyn, Bronx)

**Bu MUTlaka gösterin!** 🗽🔥
- Interactive NYC haritası
- Real-time visualization
- Professional görünüm
- Taksi projesi için perfect!

---

## 📊 Demo 3C: MLOps Monitoring Dashboard ⭐⭐⭐ (YENİ!)

### Dashboard'u Başlat
```bash
./scripts/start_monitoring_dashboard.sh
# http://localhost:8502 (farklı port!)
```

**4 Ana Tab:**

### 1. Overview 🏠
- Current model version & MAE
- Total models & improvement trends
- Recent model versions table
- Recent alerts

### 2. Continual Learning 🔄
- Check timeline (color-coded status)
- Retraining events statistics
- Latest check report details
- A/B testing results
- Performance metrics comparison

### 3. Drift Detection 📉
- Latest drift status (detected/stable)
- Drift heatmap (features x dates)
- Feature-level drift analysis table
- KS statistics, P-values, PSI scores

### 4. Model Performance 📊
- Performance trends (MAE, RMSE, R² over time)
- Model version timeline
- Model comparison table
- All models side-by-side

**Bu monitoring dashboard çok professional!** 📈
- Web-based continual learning monitoring
- Drift detection visualization
- Model performance tracking
- Production-ready dashboard

---

## 🌐 Demo 4: API & Swagger UI

### Swagger UI'da Test
```
http://localhost:8000/docs
```

### Test Endpoints:
1. **GET /health** - Health check
2. **GET /model/info** - Model bilgileri
3. **POST /predict** - Single prediction

Example request:
```json
{
  "PULocationID": 237,
  "hour": 18,
  "day_of_week": 4,
  "month": 6,
  "lag_1": 15.0,
  "lag_4": 18.0,
  "lag_96": 20.0,
  "rolling_mean_4": 16.5
}
```

4. **POST /predict/batch** - Batch prediction

---

## 📈 Demo 5: Prometheus & Grafana

### Prometheus Metrics
```
http://localhost:8000/metrics
```

**Query örnekleri (Prometheus UI - http://localhost:9090):**
```promql
# Request rate
rate(taxi_predictions_total{status="success"}[5m])

# P95 Latency
histogram_quantile(0.95, rate(taxi_prediction_latency_seconds_bucket[5m]))

# Active requests
taxi_active_requests
```

### Grafana Dashboard
```
http://localhost:3000
Login: admin / admin
```

**Dashboard'da göstereceğiniz:**
- Request rate & success rate
- Latency percentiles (p50, p95, p99)
- Prediction distribution
- Error rates
- Active requests gauge

---

## 🔍 Demo 6: Drift Detection

### Drift Analysis Çalıştır
```bash
./scripts/run_drift_check.sh
# veya
python -m src.monitoring.run_drift_detection --days-back 7
```

**Göstereceğiniz:**
- Feature-by-feature drift analysis
- KS test p-values
- PSI scores
- Distribution shifts (mean, std)
- Drift detected features

**Rapor:**
```bash
cat monitoring/drift_reports/drift_report_*.json | jq '.'
```

---

## 🔄 Demo 7: Continual Learning Pipeline ⭐⭐⭐ (EN YENİ!)

### Quick Demo (4 hafta simülasyon)
```bash
python -m src.continual_learning.demo_continual_learning --weeks 4
```

**Göstereceğiniz:**
- 📊 Haftalık performance monitoring
- 🔍 Drift detection (KS test, PSI)
- 🚨 Automatic retraining triggers
- 🏆 A/B testing (old vs new model)
- 🚀 Automated deployment
- 📈 Model versioning & registry

**Demo akışı:**
1. Her hafta için production data yüklenir
2. Performance metrics hesaplanır (MAE, RMSE, R²)
3. Drift detection yapılır (2024 train vs 2025 prod)
4. Retraining gerekirse:
   - Yeni model eğitilir
   - A/B test yapılır
   - Daha iyi model deploy edilir

### Production Continual Learning Check
```bash
# Last 7 days check
./scripts/run_continual_learning.sh --days-back 7

# Dry run (don't deploy)
./scripts/run_continual_learning.sh --days-back 7 --dry-run

# Force retrain
./scripts/run_continual_learning.sh --force

# Specific date
./scripts/run_continual_learning.sh --days-back 7 --end-date 2025-01-15
```

**Göstereceğiniz:**
- Real-world continual learning check
- Performance degradation detection
- Drift-triggered retraining
- Model comparison & deployment decision
- JSON reports (monitoring/continual_learning_reports/)

**Bu çok etkileyici olacak!** 🔥
- Otomatik model monitoring
- Drift detection ile proactive retraining
- A/B testing ile safe deployment
- Full MLOps lifecycle

---

## 📊 Demo 8: Streaming Simulation (Original)

### Klasik Streaming Demo
```bash
./demo_stream.sh
```

**Göstereceğiniz:**
- 100 kayıtlık simulation
- Real-time request/response
- MAE/RMSE hesaplama
- Terminal-based output

---

## 🔧 Demo 9: Monitoring & Health Checks

### Metrics Check
```bash
./scripts/check_metrics.sh
```

**Göstereceğiniz:**
- API health status
- Model info
- Current metrics snapshot
- Service status

### Docker Services
```bash
docker-compose -f docker-compose.prod.yml ps
```

---

## 🎯 Demo Sırası (25-30 dakika)

```
1. Giriş & Problem (2 dk)
   └─> Slides

2. Model Training (4 dk)
   └─> python -m src.training.trainer
   └─> python -m src.training.model_comparison
   └─> mlflow ui

3. Real-time Dashboard ⭐ (6 dk)
   └─> ./scripts/demo_dashboard.sh
   └─> Auto simulation çalıştır
   └─> Live graphs göster

4. API Demo (3 dk)
   └─> Swagger UI
   └─> /predict test
   └─> /metrics göster

5. Monitoring (3 dk)
   └─> Prometheus queries
   └─> Grafana dashboard

6. Drift Detection (2 dk)
   └─> ./scripts/run_drift_check.sh
   └─> Results explain

7. Continual Learning ⭐⭐⭐ (6 dk) - EN ETKİLEYİCİ!
   └─> python -m src.continual_learning.demo_continual_learning --weeks 4
   └─> Haftalık monitoring göster
   └─> Retraining triggers
   └─> A/B testing & deployment

8. Production Setup (2 dk)
   └─> Docker Compose
   └─> Services overview

9. Q&A (3-5 dk)
```

---

## 🗣️ Söyleyebileceğiniz Şeyler

### Model Performance:
```
"Modelimiz comprehensive metrics ile evaluate edildi.
MAE 3.28 ile mükemmel performans gösteriyor.
R² score ile tahmin gücümüz de oldukça güçlü.
Tahminlerimizin %X'i ±5 trip hata payı içinde."
```

### Model Comparison:
```
"3 farklı gradient boosting algoritmasını karşılaştırdık.
XGBoost, LightGBM ve CatBoost'u aynı hiperparametrelerle eğittik.
Sonuçlar tabloda görüldüğü gibi, [model] en iyi performansı gösterdi."
```

### Real-time Dashboard:
```
"Şimdi en etkileyici kısmı görelim: Real-time streaming dashboard.
Burada gerçek 2025 production verisini kullanarak
canlı tahminler yapıyoruz.
Grafiklerde predicted vs actual'i anlık olarak görebilirsiniz.
MAE ve RMSE değerleri her tahminle birlikte güncelleniyor."
```

### Drift Detection:
```
"Production'da model drift önemli bir risk.
2024 training verisi ile 2025 production verisini
istatistiksel testlerle karşılaştırıyoruz.
KS test ve PSI ile distribution shift'leri tespit ediyoruz."
```

### Production Monitoring:
```
"Tam bir MLOps pipeline kurduk.
Prometheus ile metrikleri topluyoruz,
Grafana ile visualize ediyoruz,
Alert kurallarıyla proaktif monitoring yapıyoruz.
Her şey production-ready."
```

### Continual Learning:
```
"En önemli özelliğimiz: Continual Learning Pipeline.
Production'da model performansı sürekli izleniyor.
Her hafta otomatik olarak:
  - Performance metrics kontrol ediliyor (MAE threshold)
  - Data drift detection yapılıyor (KS test, PSI)
  - Eğer drift veya performance degradation varsa,
    otomatik olarak model yeniden eğitiliyor.
  - A/B testing ile eski ve yeni model karşılaştırılıyor.
  - Daha iyi olan model otomatik deploy ediliyor.

Şimdi 4 haftalık bir simülasyon gösteriyorum.
Her hafta için drift check yapıyor, gerekirse retrain ediyor.
Bu tam otomatik, human-in-the-loop yok.
Gerçek bir MLOps production sistemi!"
```

---

## 📌 Önemli Notlar

### Sunum Öncesi Checklist:
- [ ] Production stack çalışıyor
- [ ] Tüm servisler healthy
- [ ] Data dosyaları yerinde
- [ ] Requirements install edildi
- [ ] Tarayıcı sekmeleri açık
- [ ] Terminal'ler hazır

### Eğer Bir Şey Çalışmazsa:
```bash
# Servisleri restart et
docker-compose -f docker-compose.prod.yml restart

# API'yi tek başına başlat
./start_api.sh

# Logları kontrol et
docker logs taxi-prediction-api
```

---

## 🎬 Final Kontrol

**5 dakika önce:**
1. Production stack çalışıyor mu? ✓
2. Dashboard açılıyor mu? ✓
3. MLflow UI erişilebilir mi? ✓
4. Grafana açılıyor mu? ✓
5. API Swagger UI çalışıyor mu? ✓

**Başarılar!** 🚀

---

## 🔥 Pro Tips

1. **Real-time dashboard'u muhakkahazırlık göster!** En etkileyici kısım.

2. **Model comparison'da tabloyu büyük font ile göster.** Winner belli olsun.

3. **Grafana dashboard'da refresh rate'i 5 saniye yap.** Canlı görünsün.

4. **Terminal font size'ı büyüt.** Herkes görsün.

5. **Demo sırasında API'nin response time'ını vurgula.** Hızlı!

6. **MAE/RMSE değerlerini yorumla.** "Sadece 3-4 yolcu hata!"

7. **Production stack'i gösterirken "tek komutla 6 servis" vurgula.**

8. **Drift detection sonuçlarını yorumla.** Teknik detaya girme.

9. **MLflow'da experiment comparison'ı grafik olarak göster.**

10. **Soru gelirse hazır ol:** Metrics, model choice, deployment strategy

---

**Hazırsın! Başarılar!** 🎯✨
