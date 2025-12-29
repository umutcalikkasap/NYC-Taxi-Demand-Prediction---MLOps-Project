# 🎯 SUNUM KILAVUZU - Hızlı Referans

## 📁 DOSYALAR VE İŞLEVLERİ

### 🔵 CORE DOSYALAR (Dokunma!)

#### API ve Serving
```
src/inference/app.py
├─ İşlevi: FastAPI prediction server
├─ Endpoint'ler: /predict, /health, /model/info, /metrics
├─ Port: 8000
└─ Çıktı: JSON predictions
```

#### Model Training
```
src/training/trainer.py
├─ İşlevi: Model eğitimi (XGBoost)
├─ Çıktı: models/xgb_model.json
├─ Metrics: 9 farklı metrik (MAE, RMSE, R², vb.)
└─ Süre: ~2-3 dakika
```

```
src/training/model_comparison.py
├─ İşlevi: 3 model karşılaştırması (XGBoost, LightGBM, CatBoost)
├─ Çıktı: Terminal'de comparison table
└─ Süre: ~5-6 dakika
```

#### Feature Engineering
```
src/features.py
├─ İşlevi: 25 feature oluşturma
├─ Temporal: hour, day_of_week, is_weekend, is_rush_hour, vb.
├─ Lag: lag_1, lag_4, lag_24, lag_96, lag_672
├─ Rolling: rolling_mean_4, rolling_std_4, vb.
└─ Kullanım: Trainer ve API tarafından import edilir
```

#### Continual Learning
```
src/continual_learning/retraining_pipeline.py
├─ İşlevi: CL pipeline orchestration (main class)
├─ Fonksiyonlar:
│  ├─ check_performance(): MAE kontrolü
│  ├─ check_drift(): KS test, PSI
│  ├─ should_retrain(): Karar logic
│  ├─ retrain_model(): Yeni model eğitimi
│  ├─ compare_models(): A/B testing
│  └─ deploy_model(): Model deployment
└─ Çıktı: Yeni model + registry update
```

```
src/continual_learning/run_continual_learning.py
├─ İşlevi: Production CL check runner
├─ CLI: --days-back, --dry-run, --force-retrain
└─ Çıktı: JSON report (monitoring/continual_learning_reports/)
```

```
src/continual_learning/demo_continual_learning.py
├─ İşlevi: 4-haftalık CL simülasyonu
├─ Kullanım: python -m src.continual_learning.demo_continual_learning --weeks 4
└─ Çıktı: Terminal'de beautiful tables (Rich library)
```

#### Drift Detection
```
src/monitoring/drift_detector.py
├─ İşlevi: Statistical drift detection
├─ Testler: KS test, PSI
├─ Threshold: p-value < 0.05, PSI > 0.25
└─ Kullanım: CL pipeline tarafından import
```

```
src/monitoring/run_drift_detection.py
├─ İşlevi: Standalone drift check
├─ CLI: --days-back 7
└─ Çıktı: JSON report (monitoring/drift_reports/)
```

---

## 🔄 RETRAİNİNG VERİ STRATEJİSİ (ÖNEMLİ!)

### Ne Zaman Retrain Edilir?

**3 Trigger:**
1. MAE > 2.5 (baseline threshold)
2. MAE artışı > %20
3. Drift tespit edildi (p-value < 0.05 veya PSI > 0.25)

### Hangi Veri Kullanılır?

**SLİDİNG WINDOW (Kayar Pencere) Yaklaşımı:**

```python
# Son 30 günlük production verisi kullanılır
retrain_data = pipeline.load_production_data(
    start_date="2025-01-01",
    end_date="2025-01-30",
    max_rows=None
)
```

**Önemli Noktalar:**

✅ **SADECE son 30 günlük production data** (`processed_inference_data_2025.parquet`)
❌ **Eski 2024 training data ile birleştirilmez**
✅ **Model sıfırdan eğitilir** (incremental learning değil)
✅ **Yeni data üzerinde 80/20 temporal split** yapılır

### Neden Bu Strateji?

1. **Adaptasyon**: Model en güncel trendlere adapte olur
2. **Drift'e Karşı**: Eski pattern'lar kaybolup yeni pattern'lar öğrenilir
3. **Performance**: Eski büyük datasete ihtiyaç yok, daha hızlı training
4. **Fresh Data**: Production'daki gerçek davranışları yansıtır

### Süreç Akışı:

```
Week 1: Drift tespit edildi (örn: "lag_96" ve "hour" feature'larında)
   ↓
Week 2-5'in son 30 günlük datası yükleniyor
   ↓
Bu 30 günlük data üzerinde yeni model eğitiliyor (sıfırdan)
   ↓
Yeni data için feature engineering (25 features)
   ↓
80% training, 20% validation split
   ↓
XGBoost model training
   ↓
A/B Test: Eski model vs Yeni model (production data üzerinde)
   ↓
Yeni model daha iyiyse → Deploy (xgb_model.json güncellenir)
```

### Kodda Nerede?

**[retraining_pipeline.py:539-546](src/continual_learning/retraining_pipeline.py#L539-L546)**

```python
# Retrain on last 30 days
retrain_data = pipeline.load_production_data(
    start_date="2025-01-01",
    end_date="2025-01-30",
    max_rows=None
)

new_model, metrics, version = pipeline.retrain_model(retrain_data)
```

**SUNUM SIRASINDA:** Bu stratejiyi açıklarken vurgula:
- "2024 eski data ile karıştırmıyoruz"
- "Son 30 günün taze verisini kullanıyoruz"
- "Sıfırdan öğreniyor, böylece yeni pattern'lara adapte oluyor"

---

#### Dashboard
```
src/dashboard/unified_dashboard.py
├─ İşlevi: ALL-IN-ONE web dashboard
├─ Tabs: NYC Map, CL Monitoring, Drift Detection, Live Predictions
├─ Port: 8501
└─ Teknoloji: Streamlit + Folium + Plotly
```

### 🟢 SCRIPT'LER (Hızlı Başlatma)

```
start_api.sh
├─ İşlev: API'yi başlat
├─ Komut: ./start_api.sh
└─ Port: 8000
```

```
start_dashboard.sh
├─ İşlev: Unified dashboard'u başlat
├─ Komut: ./start_dashboard.sh
└─ Port: 8501
```

```
scripts/run_continual_learning.sh (YOKSA OLUŞTUR!)
├─ İşlev: CL check çalıştır
└─ Komut: ./scripts/run_continual_learning.sh --days-back 7
```

```
scripts/run_drift_check.sh (YOKSA OLUŞTUR!)
├─ İşlev: Drift detection çalıştır
└─ Komut: ./scripts/run_drift_check.sh
```

---

## 📂 ÇIKTILAR VE KONUMLARI

### 🔴 MODEL DOSYALARI

```
models/
├── xgb_model.json              ← CURRENT production model
├── production/
│   ├── xgb_model.json          ← Symlink to current
│   ├── xgb_model_backup.json   ← Backup before deployment
│   ├── xgb_model_v1.1.0.json   ← Versioned models
│   └── model_registry.json     ← MODEL METADATA (ÖNEMLİ!)
```

**model_registry.json içeriği:**
```json
[
  {
    "version": "v1.0.0",
    "trained_at": "2025-01-15T10:30:00",
    "model_path": "models/production/xgb_model.json",
    "performance": {
      "mae": 2.32,
      "rmse": 4.47,
      "r2": 0.9626
    },
    "is_deployed": true
  }
]
```

**SUNUM SIRASINDA:**
- Dashboard Tab 2'de gösterilir
- `cat models/production/model_registry.json | jq '.'` ile terminal'de gösterebilirsin

---

### 🟡 CONTINUAL LEARNING RAPORLARI

```
monitoring/continual_learning_reports/
├── cl_report_20250115_103000.json
├── cl_report_20250122_103000.json
└── cl_report_20250129_103000.json
```

**Report içeriği:**
```json
{
  "timestamp": "2025-01-15T10:30:00",
  "status": "success_deployed",
  "actions_taken": ["retrain_triggered", "model_retrained", "model_deployed"],
  "should_retrain": true,
  "retrain_reasons": ["Drift detected in 2 features"],
  "performance_metrics": {
    "mae": 2.45,
    "rmse": 4.58,
    "r2": 0.9612
  },
  "drift_results": {
    "drift_detected": true,
    "drift_detected_features": ["lag_96", "hour"]
  },
  "model_comparison": {
    "current_model": {"mae": 2.45},
    "new_model": {"mae": 2.18},
    "winner": "new"
  },
  "new_model_version": "v1.1.0"
}
```

**SUNUM SIRASINDA:**
- Dashboard Tab 2 (Continual Learning) otomatik gösterir
- En son raporu görmek için: `cat monitoring/continual_learning_reports/cl_report_*.json | tail -1 | jq '.'`

---

### 🟠 DRIFT DETECTION RAPORLARI

```
monitoring/drift_reports/
├── drift_report_20250115.json
├── drift_report_20250122.json
└── drift_report_20250129.json
```

**Report içeriği:**
```json
{
  "timestamp": "2025-01-15T12:00:00",
  "drift_detected": true,
  "features_checked": ["hour", "lag_1", "lag_4", "lag_96", "rolling_mean_4", ...],
  "drift_detected_features": ["lag_96", "hour"],
  "drift_scores": {
    "lag_96": {
      "ks_statistic": 0.0856,
      "p_value": 0.0012,
      "psi": 0.3245
    },
    "hour": {
      "ks_statistic": 0.0423,
      "p_value": 0.0234,
      "psi": 0.2789
    }
  }
}
```

**SUNUM SIRASINDA:**
- Dashboard Tab 3 (Drift Detection) otomatik gösterir
- Terminal'de: `cat monitoring/drift_reports/drift_report_*.json | tail -1 | jq '.drift_detected_features'`

---

### 🟣 MLFLOW TRACKING

```
mlruns/
├── 0/                          ← Experiment ID
│   ├── meta.yaml
│   └── <run_id>/
│       ├── metrics/
│       │   ├── mae
│       │   ├── rmse
│       │   └── r2
│       ├── params/
│       │   ├── n_estimators
│       │   └── learning_rate
│       └── artifacts/
│           └── model/
```

**SUNUM SIRASINDA:**
- MLflow UI: `mlflow ui` → http://localhost:5000
- Göstereceksin: Experiment runs, metric comparison charts

---

### 🔵 DATA KONUMLARI

```
data/
├── training/
│   └── yellow_tripdata_2024-*.parquet   ← Training data (2024)
│
├── inference/
│   └── yellow_tripdata_2025-01.parquet  ← Production data (2025)
│
└── processed/
    └── aggregated_*.parquet             ← Feature-engineered data
```

**SUNUM SIRASINDA:**
- Bu dosyaları göstermene gerek yok
- Eğer sorarlarsa: "2024 training, 2025 production data kullanıyoruz"

---

## 🎬 SUNUM AKIŞI - HANGİ DOSYA NE ZAMAN?

### 1. Başlangıç (5 dk önce)

```bash
# Terminal 1: API başlat
./start_api.sh

# Terminal 2: Dashboard başlat
./start_dashboard.sh

# Tarayıcı:
# - http://localhost:8501 (Dashboard)
# - http://localhost:8000/docs (Swagger - optional)
```

**Hangi dosyalar çalışıyor:**
- `src/inference/app.py` → API server
- `src/dashboard/unified_dashboard.py` → Dashboard
- `models/xgb_model.json` → Loaded by API

---

### 2. NYC Map Demo (5 dk)

**Dashboard Tab 1: NYC Map**

**Ne gösteriyor:**
- `src/dashboard/unified_dashboard.py` → Tab 1
- Veri kaynağı: `data/inference/yellow_tripdata_2025-01.parquet`
- Harita teknolojisi: Folium

**Hangi dosyalar aktif:**
- Dashboard kodu: `unified_dashboard.py` içindeki `create_folium_heatmap()` fonksiyonu
- NYC lokasyonları: `NYC_LOCATIONS` dictionary (hardcoded)

**Göstereceksin:**
1. Hour slider → demand değişimi
2. Marker'lara tıkla → popup göster
3. "Live Predictions" butonuna bas
   - API'ye istek atar: `make_prediction()` → `/predict` endpoint
   - `src/inference/app.py` içindeki `predict_single()` çalışır
   - Model: `models/xgb_model.json` kullanılır

**Eğer çalışmazsa:**
- API çalışıyor mu? → `curl http://localhost:8000/health`
- Data var mı? → `ls -lh data/inference/`

---

### 3. Continual Learning Demo (4 dk)

**Dashboard Tab 2: Continual Learning**

**Ne gösteriyor:**
- Model registry: `models/production/model_registry.json` okunur
- CL reports: `monitoring/continual_learning_reports/*.json` okunur

**Hangi dosyalar okunuyor:**
- Dashboard: `load_model_registry()` fonksiyonu
- Dashboard: `load_cl_reports()` fonksiyonu

**Göstereceksin:**
1. Current model version: Registry'den gelir
2. Model history table: Son 10 model
3. Latest check report:
   - Status
   - Actions taken
   - Performance metrics
   - Retraining reasons (varsa)
   - A/B test results (varsa)

**Eğer boşsa:**
- İlk kez CL check çalıştır:
  ```bash
  python -m src.continual_learning.run_continual_learning --days-back 7 --dry-run
  ```
- Dashboard'u refresh et (F5)
- Dosya oluşacak: `monitoring/continual_learning_reports/cl_report_*.json`

---

### 4. Drift Detection Demo (3 dk)

**Dashboard Tab 3: Drift Detection**

**Ne gösteriyor:**
- Drift reports: `monitoring/drift_reports/*.json` okunur

**Hangi dosyalar okunuyor:**
- Dashboard: `load_drift_reports()` fonksiyonu

**Göstereceksin:**
1. Drift status: Detected/Stable
2. Features checked count
3. Drifted features list
4. Feature analysis table:
   - Feature name
   - KS Statistic
   - P-Value
   - PSI
   - Drift indicator (🔴/✅)

**Eğer boşsa:**
- Drift detection çalıştır:
  ```bash
  python -m src.monitoring.run_drift_detection --days-back 7
  ```
- Dashboard refresh (F5)
- Dosya oluşacak: `monitoring/drift_reports/drift_report_*.json`

---

### 5. Live Predictions Demo (2 dk)

**Dashboard Tab 4: Live Predictions**

**Ne gösteriyor:**
- Manual prediction form
- API integration

**Hangi dosyalar çalışıyor:**
- Dashboard: `make_prediction()` fonksiyonu → API'ye POST request
- API: `src/inference/app.py` → `predict_single()` endpoint
- Model: `models/xgb_model.json`

**Göstereceksin:**
1. Location seç (örn: Times Square - 211)
2. Hour seç (örn: 18)
3. "Make Prediction" butonuna bas
4. Sonuç gösterilir: "Predicted Demand: 15.2 trips"
5. Historical comparison (varsa)

**Eğer hata alırsan:**
- API çalışıyor mu? → Terminal 1'e bak
- Model yüklü mü? → `curl http://localhost:8000/model/info`

---

## 🔧 OLASI SORUNLAR VE HIZLI ÇÖZÜMLER

### Problem 1: Dashboard boş gözüküyor

**Sebep:** Reports yok

**Çözüm:**
```bash
# CL report oluştur
python -m src.continual_learning.run_continual_learning --days-back 7 --dry-run

# Drift report oluştur
python -m src.monitoring.run_drift_detection --days-back 7

# Dashboard refresh (F5)
```

**Hangi dosyalar oluşacak:**
- `monitoring/continual_learning_reports/cl_report_*.json`
- `monitoring/drift_reports/drift_report_*.json`

---

### Problem 2: API çalışmıyor

**Kontrol:**
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info
```

**Hata varsa:**
```bash
# API'yi restart et
Ctrl+C (Terminal 1'de)
./start_api.sh
```

**Hangi dosya çalışıyor:**
- `src/inference/app.py`
- Model: `models/xgb_model.json`

---

### Problem 3: Haritada data yok

**Sebep:** Data file yok

**Kontrol:**
```bash
ls -lh data/inference/
```

**Çözüm:**
- `yellow_tripdata_2025-01.parquet` olmalı
- Yoksa demo mode'da çalış (sadece Live Predictions kullan)

---

### Problem 4: Model yok

**Kontrol:**
```bash
ls -lh models/xgb_model.json
```

**Çözüm:**
```bash
# Model train et
python -m src.training.trainer
```

**Hangi dosyalar oluşacak:**
- `models/xgb_model.json`
- MLflow run: `mlruns/0/<run_id>/`

---

## 📋 HIZLI KOMUT REFERANSI

### API İşlemleri
```bash
# API başlat
./start_api.sh

# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Manuel prediction (test)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID": 237, "hour": 18, "day_of_week": 4, "month": 1, "lag_1": 15, "lag_4": 18, "lag_96": 20, "rolling_mean_4": 16.5}'
```

### Dashboard İşlemleri
```bash
# Dashboard başlat
./start_dashboard.sh

# Port kontrolü
lsof -i :8501
```

### Continual Learning
```bash
# CL check (dry-run)
python -m src.continual_learning.run_continual_learning --days-back 7 --dry-run

# CL check (production - deploy eder!)
python -m src.continual_learning.run_continual_learning --days-back 7

# Demo (4 hafta)
python -m src.continual_learning.demo_continual_learning --weeks 4

# Force retrain
python -m src.continual_learning.run_continual_learning --force-retrain
```

### Drift Detection
```bash
# Drift check
python -m src.monitoring.run_drift_detection --days-back 7

# Drift report oku
cat monitoring/drift_reports/drift_report_*.json | tail -1 | jq '.'
```

### Model Training
```bash
# Train model
python -m src.training.trainer

# Model comparison
python -m src.training.model_comparison

# MLflow UI
mlflow ui
# http://localhost:5000
```

### Reports
```bash
# CL reports
ls -lh monitoring/continual_learning_reports/
cat monitoring/continual_learning_reports/cl_report_*.json | tail -1 | jq '.'

# Drift reports
ls -lh monitoring/drift_reports/
cat monitoring/drift_reports/drift_report_*.json | tail -1 | jq '.'

# Model registry
cat models/production/model_registry.json | jq '.'
```

---

## 📊 HANGİ METRIKLER NEREDE?

### Dashboard'da Gösterilen Metrikler

**Tab 2 (Continual Learning):**
- Current Model Version
- MAE
- Total Models
- CL Checks
- Retraining Events

**Tab 3 (Drift Detection):**
- Drift Status
- Features Checked
- Drifted Features
- KS Statistic (per feature)
- P-Value (per feature)
- PSI (per feature)

**Tab 4 (Live Predictions):**
- Predicted Demand
- Historical Demand (comparison)
- Error

### Terminal'de Gösterilen Metrikler

**Model Training (trainer.py):**
- MAE: 2.32
- RMSE: 4.47
- R² Score: 0.9626
- MAPE: 29%
- MedAE: 0.93
- Bias: +0.15
- Within ±5 trips: 86.8%
- Within ±10%: 64.2%
- Max Error: 48.7

**Model Comparison:**
- XGBoost vs LightGBM vs CatBoost
- Tüm metrikler yan yana
- Winner işaretli (✅)

---

## 🎯 SUNUM SIRASIYLA KULLANILACAK DOSYALAR

1. **Giriş:**
   - README.md (opsiyonel)

2. **NYC Map Demo:**
   - `src/dashboard/unified_dashboard.py` (Tab 1)
   - `data/inference/yellow_tripdata_2025-01.parquet` (veri kaynağı)
   - `src/inference/app.py` (Live Predictions için)
   - `models/xgb_model.json` (tahminler için)

3. **CL Demo:**
   - `src/dashboard/unified_dashboard.py` (Tab 2)
   - `models/production/model_registry.json` (model history)
   - `monitoring/continual_learning_reports/*.json` (reports)

4. **Drift Demo:**
   - `src/dashboard/unified_dashboard.py` (Tab 3)
   - `monitoring/drift_reports/*.json` (reports)

5. **Live Predictions:**
   - `src/dashboard/unified_dashboard.py` (Tab 4)
   - `src/inference/app.py` (API)
   - `models/xgb_model.json` (model)

6. **Kapanış:**
   - PROJECT_STRUCTURE.md (opsiyonel)
   - PRESENTATION_COMMANDS.md (opsiyonel)

---

## ✅ SON KONTROL LİSTESİ (Sunum 5 Dakika Önce)

- [ ] API çalışıyor (`./start_api.sh` + `curl http://localhost:8000/health`)
- [ ] Dashboard açık (`./start_dashboard.sh` + http://localhost:8501)
- [ ] Model yüklü (`ls models/xgb_model.json`)
- [ ] Data var (`ls data/inference/*.parquet`)
- [ ] CL report var (`ls monitoring/continual_learning_reports/`)
- [ ] Drift report var (`ls monitoring/drift_reports/`)
- [ ] Model registry var (`cat models/production/model_registry.json`)
- [ ] Tarayıcı full-screen (F11)

**Eğer raporlar yoksa:**
```bash
# 1 dakikalık hızlı fix:
python -m src.continual_learning.run_continual_learning --days-back 7 --dry-run
python -m src.monitoring.run_drift_detection --days-back 7
# Dashboard refresh (F5)
```

---

## 💡 PRO TİPLER

1. **Terminal fontunu büyüt** - Herkes görsün
2. **Dashboard full-screen** - F11
3. **2 tarayıcı sekmesi aç** - Dashboard + Swagger UI
4. **Terminal'leri organize et** - API, Dashboard, Extra
5. **Backup plan** - API çalışmazsa, sadece heatmap modunu göster

---

**Bu kılavuzu sunum sırasında yanında tut!** 📖🚀
